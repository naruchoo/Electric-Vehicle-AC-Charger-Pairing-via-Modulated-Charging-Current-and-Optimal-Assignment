import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment


def calculate_correlation_unit(df_ev_generation: pd.DataFrame, df_charger_generation: pd.DataFrame):
    """
    Compute the Pearson correlation coefficient r_ij for one EV–charger pair.

    Time alignment rule (as in the paper):
      - When the sequence lengths differ, use the shorter series length as the standard.
      - Pair samples that are temporally closest within a tolerance range of ±5 sec.
      - Compute the correlation coefficient using the paired samples.

    Parameters
    ----------
    df_ev_generation : pd.DataFrame
        Index: timestamps, Column: 'ev_current'
    df_charger_generation : pd.DataFrame
        Index: timestamps, Column: 'charger_current'

    Returns
    -------
    correlation : float
        Pearson correlation coefficient. If insufficient valid pairs exist, returns NaN.
    merged_df : pd.DataFrame
        Paired samples used for computing correlation (reproducibility / inspection).
    """
    # Convert timestamps to Unix epoch seconds
    charger_times = df_charger_generation.index.astype("int64") // 10**9
    charger_current = df_charger_generation["charger_current"]

    ev_times = df_ev_generation.index.astype("int64") // 10**9
    ev_current = df_ev_generation["ev_current"]

    # Ensure timestamps are sorted (required by np.searchsorted)
    if not np.all(np.diff(charger_times) >= 0):
        sorted_idx = np.argsort(charger_times)
        charger_times = charger_times[sorted_idx]
        charger_current = charger_current.iloc[sorted_idx]

    if not np.all(np.diff(ev_times) >= 0):
        sorted_idx = np.argsort(ev_times)
        ev_times = ev_times[sorted_idx]
        ev_current = ev_current.iloc[sorted_idx]

    # For each EV timestamp, find the nearest charger timestamp index
    idx = np.searchsorted(charger_times, ev_times, side="left")
    idx_prev = np.clip(idx - 1, 0, len(charger_times) - 1)
    idx_next = np.clip(idx, 0, len(charger_times) - 1)

    # Choose the nearer of the previous/next charger timestamps
    time_prev = charger_times[idx_prev]
    time_next = charger_times[idx_next]
    diff_prev = np.abs(ev_times - time_prev)
    diff_next = np.abs(ev_times - time_next)

    choose_prev = diff_prev <= diff_next
    choose_next = ~choose_prev

    # Pair samples only if the nearest timestamp is within ±5 sec
    matched_charger_current = np.full_like(ev_current.to_numpy(), np.nan, dtype=np.float32)
    within_prev = diff_prev <= 5
    within_next = diff_next <= 5

    mask_prev = choose_prev & within_prev
    mask_next = choose_next & within_next

    matched_charger_current[mask_prev] = charger_current.iloc[idx_prev[mask_prev]].to_numpy(dtype=np.float32)
    matched_charger_current[mask_next] = charger_current.iloc[idx_next[mask_next]].to_numpy(dtype=np.float32)

    # Keep only valid paired samples
    valid = ~np.isnan(matched_charger_current)
    x = matched_charger_current[valid]
    y = ev_current.to_numpy(dtype=np.float32)[valid]

    # Pearson correlation coefficient (Eq. (2) in the paper)
    if len(x) > 1:
        correlation = np.corrcoef(x, y)[0, 1]
    else:
        correlation = np.nan

    merged_df = pd.DataFrame(
        {"charger_current": x, "ev_current": y},
        index=pd.to_datetime(ev_times[valid], unit="s"),
    )
    return correlation, merged_df


def calculate_correlation(df_chargers: pd.DataFrame, df_evs: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the correlation similarity matrix R (rows: chargers, columns: EVs).

    For each (charger_i, EV_j):
      - Compute r_ij using calculate_correlation_unit() with ±5 sec pairing.
      - If r_ij is NaN (no valid pairs), set r_ij = 0.0.

    Returns
    -------
    correlations_df : pd.DataFrame
        Similarity matrix R where entry (i, j) is the correlation coefficient r_ij.
    """
    charger_cols = df_chargers.columns
    ev_cols = df_evs.columns

    num_chargers = len(charger_cols)
    num_evs = len(ev_cols)

    correlations = np.empty((num_chargers, num_evs), dtype=np.float32)

    # Column-wise NumPy arrays for memory efficiency
    charger_data = {c: df_chargers[c].astype(np.float32).values for c in charger_cols}
    ev_data = {e: df_evs[e].astype(np.float32).values for e in ev_cols}

    def compute_one(i, j, c_vals, e_vals, c_ts, e_ts):
        df_c = pd.DataFrame({"charger_current": c_vals}, index=c_ts)
        df_e = pd.DataFrame({"ev_current": e_vals}, index=e_ts)
        r_ij, _ = calculate_correlation_unit(df_e, df_c)
        if np.isnan(r_ij):
            r_ij = 0.0
        return i, j, r_ij

    tasks = []
    c_ts = df_chargers.index
    e_ts = df_evs.index
    for i, c_col in enumerate(charger_cols):
        for j, e_col in enumerate(ev_cols):
            tasks.append((i, j, charger_data[c_col], ev_data[e_col], c_ts, e_ts))

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_one)(i, j, c_vals, e_vals, c_ts, e_ts) for (i, j, c_vals, e_vals, c_ts, e_ts) in tasks
    )

    for i, j, r_ij in results:
        correlations[i, j] = r_ij

    return pd.DataFrame(correlations, index=charger_cols, columns=ev_cols)


def perform_matching_correlation(evaluation_scores_df: pd.DataFrame):
    """
    Solve the optimal one-to-one assignment via the Hungarian method.

    This function takes an evaluation (similarity) matrix and converts it into a cost matrix
    (lower cost = higher pairing likelihood), then applies the Hungarian method.

    Notes
    -----
    - In the paper, the cost matrix is obtained by the one-minus transformation (C_ij = 1 - E_ij).
    - Here we use a monotone transformation (max - score) so that Hungarian minimization
      prefers higher similarity scores.

    Returns
    -------
    matching : dict
        Mapping {charger_name: ev_name}.
    accuracy : float
        Simple accuracy assuming Charger_k ↔ EV_k is the correct ground truth.
    """
    scores = evaluation_scores_df.astype(float).values
    cost_matrix = np.max(scores) - scores  # cost matrix for minimization

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    charger_names = evaluation_scores_df.index.tolist()
    ev_names = evaluation_scores_df.columns.tolist()

    matching = {charger_names[i]: ev_names[j] for i, j in zip(row_ind, col_ind)}

    correct_matches = np.sum(row_ind == col_ind)
    accuracy = correct_matches / len(row_ind)

    return matching, accuracy