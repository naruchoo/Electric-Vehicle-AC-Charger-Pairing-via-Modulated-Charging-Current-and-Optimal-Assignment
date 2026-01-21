import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment


def calculate_correlation_unit(df_ev_generation: pd.DataFrame, df_charger_generation: pd.DataFrame):
    """
    Compute r_ij for one EV–charger pair using ±5 sec nearest-timestamp pairing.
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


def calculate_dtw_unit(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Compute the DTW distance d_ij^DTW between two sequences.

    Definition follows the paper:
      - Local distance: |x_p - y_q|
      - Dynamic programming recursion for cumulative distance
      - Normalize by the warping path length (average local distance)
    """
    n = len(seq1)
    m = len(seq2)

    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    total_cost = dtw[n, m]

    # Backtrack to estimate warping path length K_ij
    i, j = n, m
    path_length = 0
    while i > 0 or j > 0:
        path_length += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_val = min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
            if dtw[i - 1, j - 1] == min_val:
                i -= 1
                j -= 1
            elif dtw[i - 1, j] == min_val:
                i -= 1
            else:
                j -= 1

    return total_cost / path_length


def calculate_correlation_dtw(df_chargers: pd.DataFrame, df_evs: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the cost matrix C for Correlation–DTW and return it.

    Paper-aligned steps (conceptually):
      A) Similarity matrices:
         - Correlation coefficients r_ij
         - DTW distances d_ij^DTW
      B) Normalization:
         - r_tilde_ij = (r_ij + 1)/2
         - Normalize DTW distance to [0,1] (smaller distance -> higher similarity)
      C) Hadamard product:
         - E_ij = r_tilde_ij ⊙ d_s_ij  (Correlation–DTW composite)
      D) One-minus transformation:
         - C_ij = 1 - E_ij  (cost matrix for Hungarian minimization)

    Returns
    -------
    cost_matrix_df : pd.DataFrame
        Cost matrix C (rows: chargers, columns: EVs), lower is better.
    """
    charger_cols = df_chargers.columns
    ev_cols = df_evs.columns
    num_chargers = len(charger_cols)
    num_evs = len(ev_cols)

    correlations = np.empty((num_chargers, num_evs), dtype=np.float32)
    dtw_distances = np.empty((num_chargers, num_evs), dtype=np.float32)

    charger_data = {c: df_chargers[c].astype(np.float32).values for c in charger_cols}
    ev_data = {e: df_evs[e].astype(np.float32).values for e in ev_cols}

    def compute_one(i, j, c_vals, e_vals, c_ts, e_ts):
        # Correlation coefficient (paired within ±5 sec)
        df_c = pd.DataFrame({"charger_current": c_vals}, index=c_ts)
        df_e = pd.DataFrame({"ev_current": e_vals}, index=e_ts)
        r_ij, _ = calculate_correlation_unit(df_e, df_c)
        if np.isnan(r_ij):
            r_ij = 0.0

        # DTW distance on the raw sampled sequences
        d_ij = calculate_dtw_unit(c_vals, e_vals)
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
        dtw_distances[i, j] = d_ij

    corr_df = pd.DataFrame(correlations, index=charger_cols, columns=ev_cols)
    dtw_df = pd.DataFrame(dtw_distances, index=charger_cols, columns=ev_cols)

    # Normalize correlation coefficients to [0,1] (Eq. (9))
    r_tilde = (corr_df + 1.0) / 2.0

    # Normalize DTW distance to [0,1] and convert to "similarity-like" score (larger is better)
    d_min = dtw_df.min().min()
    d_max = dtw_df.max().max()
    if d_max == d_min:
        d_s = dtw_df * 0.0
    else:
        d_s = 1.0 - (dtw_df - d_min) / (d_max - d_min)

    # Hadamard product (Eq. (11)): Correlation–DTW composite evaluation
    evaluation = r_tilde * d_s

    # One-minus transformation (Eq. (12)): cost matrix
    cost_matrix_df = 1.0 - evaluation
    return cost_matrix_df


def perform_matching_correlation_dtw(cost_matrix_df: pd.DataFrame):
    """
    Solve the optimal one-to-one assignment using the Hungarian method on the cost matrix.

    Returns
    -------
    matching : dict
        Mapping {charger_name: ev_name}.
    accuracy : float
        Accuracy computed by matching name suffixes: Charger_k ↔ EV_k is treated as correct.
    """
    cost_matrix = cost_matrix_df.values
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    charger_names = cost_matrix_df.index
    ev_names = cost_matrix_df.columns
    matching = {charger_names[r]: ev_names[c] for r, c in zip(row_ind, col_ind)}

    correct = 0
    for ch, ev in matching.items():
        if ch.replace("Charger_", "") == ev.replace("EV_", ""):
            correct += 1
    accuracy = correct / len(matching) if matching else 0.0

    return matching, accuracy
