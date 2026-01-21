import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment


def calculate_correlation_unit(df_ev_generation: pd.DataFrame, df_charger_generation: pd.DataFrame):
    """
    Compute r_ij for one EV–charger pair using ±5 sec nearest-timestamp pairing,
    then return the paired samples used to compute r_ij.
    """
    charger_times = df_charger_generation.index.astype("int64") // 10**9
    charger_current = df_charger_generation["charger_current"]

    ev_times = df_ev_generation.index.astype("int64") // 10**9
    ev_current = df_ev_generation["ev_current"]

    if not np.all(np.diff(charger_times) >= 0):
        s = np.argsort(charger_times)
        charger_times = charger_times[s]
        charger_current = charger_current.iloc[s]

    if not np.all(np.diff(ev_times) >= 0):
        s = np.argsort(ev_times)
        ev_times = ev_times[s]
        ev_current = ev_current.iloc[s]

    idx = np.searchsorted(charger_times, ev_times, side="left")
    idx_prev = np.clip(idx - 1, 0, len(charger_times) - 1)
    idx_next = np.clip(idx, 0, len(charger_times) - 1)

    time_prev = charger_times[idx_prev]
    time_next = charger_times[idx_next]
    diff_prev = np.abs(ev_times - time_prev)
    diff_next = np.abs(ev_times - time_next)

    choose_prev = diff_prev <= diff_next
    choose_next = ~choose_prev

    matched_charger_current = np.full_like(ev_current.to_numpy(), np.nan, dtype=np.float32)
    within_prev = diff_prev <= 5
    within_next = diff_next <= 5

    mask_prev = choose_prev & within_prev
    mask_next = choose_next & within_next

    matched_charger_current[mask_prev] = charger_current.iloc[idx_prev[mask_prev]].to_numpy(dtype=np.float32)
    matched_charger_current[mask_next] = charger_current.iloc[idx_next[mask_next]].to_numpy(dtype=np.float32)

    valid = ~np.isnan(matched_charger_current)
    x = matched_charger_current[valid]
    y = ev_current.to_numpy(dtype=np.float32)[valid]

    if len(x) > 1:
        correlation = np.corrcoef(x, y)[0, 1]
    else:
        correlation = np.nan

    merged_df = pd.DataFrame(
        {"charger_current": x, "ev_current": y},
        index=pd.to_datetime(ev_times[valid], unit="s"),
    )
    return correlation, merged_df


def calculate_correlation_euclidean(df_chargers: pd.DataFrame, df_evs: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the composite evaluation matrix E using Correlation–Euclidean.

    Steps follow the paper:
      A) Similarity matrices:
         - Correlation coefficient r_ij (overall similarity)
         - Euclidean distance d_ij^E (point-to-point deviation) computed on paired samples
      B) Normalization:
         - Normalize correlation to [0,1]: r_tilde = (r + 1) / 2
         - Normalize distance to [0,1] with min-max and one-minus so that larger is better
      C) Hadamard product:
         - E_ij = r_tilde_ij ⊙ d_s_ij

    Returns
    -------
    evaluation_scores : pd.DataFrame
        Composite evaluation matrix E (higher is better).
    """
    charger_cols = df_chargers.columns
    ev_cols = df_evs.columns

    num_chargers = len(charger_cols)
    num_evs = len(ev_cols)

    correlations = np.empty((num_chargers, num_evs), dtype=np.float32)
    distances = np.empty((num_chargers, num_evs), dtype=np.float32)

    charger_data = {c: df_chargers[c].astype(np.float32).values for c in charger_cols}
    ev_data = {e: df_evs[e].astype(np.float32).values for e in ev_cols}

    def compute_one(i, j, c_vals, e_vals, c_ts, e_ts):
        df_c = pd.DataFrame({"charger_current": c_vals}, index=c_ts)
        df_e = pd.DataFrame({"ev_current": e_vals}, index=e_ts)

        r_ij, paired = calculate_correlation_unit(df_e, df_c)
        if np.isnan(r_ij):
            r_ij = 0.0

        # Euclidean distance on the paired samples (Eq. (4) in the paper)
        if len(paired) == 0:
            d_ij = np.inf
        else:
            diff = paired["charger_current"].to_numpy(dtype=np.float32) - paired["ev_current"].to_numpy(dtype=np.float32)
            d_ij = np.sqrt(np.sum(diff * diff))

        return i, j, r_ij, d_ij

    tasks = []
    c_ts = df_chargers.index
    e_ts = df_evs.index
    for i, c_col in enumerate(charger_cols):
        for j, e_col in enumerate(ev_cols):
            tasks.append((i, j, charger_data[c_col], ev_data[e_col], c_ts, e_ts))

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_one)(i, j, c_vals, e_vals, c_ts, e_ts) for (i, j, c_vals, e_vals, c_ts, e_ts) in tasks
    )

    for i, j, r_ij, d_ij in results:
        correlations[i, j] = r_ij
        distances[i, j] = d_ij

    corr_df = pd.DataFrame(correlations, index=charger_cols, columns=ev_cols)
    dist_df = pd.DataFrame(distances, index=charger_cols, columns=ev_cols)

    # Normalize correlation coefficients to [0,1] (Eq. (9))
    r_tilde = (corr_df + 1.0) / 2.0

    # Normalize distance metrics to [0,1] (Eq. (10)-style): larger is better
    d_min = dist_df.min().min()
    d_max = dist_df.max().max()
    if d_max == d_min:
        d_s = dist_df * 0.0
    else:
        d_s = 1.0 - (dist_df - d_min) / (d_max - d_min)

    # Hadamard product (Eq. (11)): Correlation–Euclidean
    evaluation_scores = r_tilde * d_s
    return evaluation_scores


def perform_matching_correlation_euclidean(evaluation_scores_df: pd.DataFrame):
    """
    Perform optimal one-to-one assignment using the Hungarian method.

    This function converts the evaluation matrix to a cost matrix for minimization
    and then solves the assignment problem.

    Returns
    -------
    matching : dict
        Mapping {charger_name: ev_name}.
    accuracy : float
        Simple accuracy assuming Charger_k ↔ EV_k is the correct ground truth.
    """
    scores = evaluation_scores_df.astype(float).values
    cost_matrix = np.max(scores) - scores

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    charger_names = evaluation_scores_df.index.tolist()
    ev_names = evaluation_scores_df.columns.tolist()

    matching = {charger_names[i]: ev_names[j] for i, j in zip(row_ind, col_ind)}

    correct_matches = np.sum(row_ind == col_ind)
    accuracy = correct_matches / len(row_ind)

    return matching, accuracy
