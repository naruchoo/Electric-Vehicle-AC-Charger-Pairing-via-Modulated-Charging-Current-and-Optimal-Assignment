import math
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# Parameters (grouped at top for readability)
# =============================================================================
# --- Randomness ---------------------------------------------------------------
DEFAULT_RANDOM_SEED = 42
np.random.seed(DEFAULT_RANDOM_SEED)

# --- Command pattern settings (Table I: command values) -----------------------
COMMAND_MIN_A = 6
COMMAND_MAX_A = 30
NUM_COMMAND_UPDATES = 5  # number of updates after the initial 0 A
COMMAND_PATTERN_LENGTH = 1 + NUM_COMMAND_UPDATES  # includes the initial 0 A

# --- Simulation step (Algorithm 1: Δt) ---------------------------------------
SIM_STEP_S = 1

# --- Stage I: Charger-current generation model (Algorithm 1) -----------------
# Change-start delay (Algorithm 1: first/next change-start delays d1, d)
FIRST_CHANGE_START_DELAY_S = 15
NEXT_CHANGE_START_DELAY_S = 5

# Fitted parameters (Algorithm 1: b(c), r↑(Δc), r↓)
OFFSET_COEF = 0.0221
OFFSET_INTERCEPT = 0.2831

RAMP_UP_COEF = 0.0327       # NOTE: keep as-is to preserve existing behavior
RAMP_UP_INTERCEPT = 0.3787  # A/sec
RAMP_DOWN_A_PER_S = 4.0     # A/sec

# Quantization (Algorithm 1: charger 0.1 A)
CHARGER_QUANTIZATION_DECIMALS = 1  # round(., 1) -> 0.1 A

# --- Stage II: EV-current generation model (Algorithm 1) ---------------------
# Charger -> EV time offset (Algorithm 1: Δt_off = 5 sec)
CHARGER_TO_EV_TIME_OFFSET_S = 5


# =============================================================================
# Helper functions (internal)
# =============================================================================
_ONE_SEC = pd.Timedelta(seconds=SIM_STEP_S)


def _append_constant_current(
    records: List[Tuple[pd.Timestamp, float]],
    current_time: pd.Timestamp,
    end_time: pd.Timestamp,
    current_value: float,
) -> pd.Timestamp:
    """
    Append (time, current_value) each second until current_time reaches end_time.
    This preserves the original behavior:
      while current_time < end_time:
          current_time += 1 sec
          append(current_time, current_value)
    """
    while current_time < end_time:
        current_time += _ONE_SEC
        records.append((current_time, current_value))
    return current_time


def _ramp_up_to_target(
    records: List[Tuple[pd.Timestamp, float]],
    current_time: pd.Timestamp,
    current_value: float,
    target_value: float,
    ramp_rate_a_per_s: float,
) -> Tuple[pd.Timestamp, float]:
    """Ramp up by ramp_rate each second until reaching target_value (inclusive)."""
    while current_value < target_value:
        current_value = min(current_value + ramp_rate_a_per_s, target_value)
        current_time += _ONE_SEC
        records.append((current_time, current_value))
    return current_time, current_value


def _ramp_down_to_target(
    records: List[Tuple[pd.Timestamp, float]],
    current_time: pd.Timestamp,
    current_value: float,
    target_value: float,
    ramp_rate_a_per_s: float,
) -> Tuple[pd.Timestamp, float]:
    """Ramp down by ramp_rate each second until reaching target_value (inclusive)."""
    while current_value > target_value:
        current_value = max(current_value - ramp_rate_a_per_s, target_value)
        current_time += _ONE_SEC
        records.append((current_time, current_value))
    return current_time, current_value


# =============================================================================
# Public functions (API)
# =============================================================================
def get_command_data(initial_timestamp, command_values, tau=60):
    """
    Build the command timeline (Algorithm 1: Build command timeline).

    Parameters
    ----------
    initial_timestamp : str | pd.Timestamp
        Start time t0.
    command_values : Sequence[int]
        Command sequence (e.g., [0, 15, 30, 22, 18, 6]).
    tau : int
        Command update period [sec] (Algorithm 1: τ). Default is 60 sec.

    Returns
    -------
    pd.DataFrame
        Columns: ["timestamp", "command_value"].
    """
    timestamps = pd.date_range(start=initial_timestamp, periods=len(command_values), freq=f"{tau}s")
    return pd.DataFrame({"timestamp": timestamps, "command_value": command_values})


def generate_command_patterns(num_patterns=10, seed=42):
    """
    Generate unique command patterns for chargers.

    Each pattern has 6 values:
      - The 1st value is 0 A
      - The remaining 5 values are unique integers in [6, 30]

    Note
    ----
    This follows the paper’s setting where command values are integers (1 A resolution),
    with the lower bound fixed to 6 A and default upper bound 30 A.:contentReference[oaicite:1]{index=1}
    """
    rnd = np.random.RandomState(seed)

    patterns = set()
    max_attempts = num_patterns * 10
    attempts = 0

    while len(patterns) < num_patterns and attempts < max_attempts:
        pattern = [0] + rnd.choice(range(COMMAND_MIN_A, COMMAND_MAX_A + 1), size=NUM_COMMAND_UPDATES, replace=False).tolist()
        patterns.add(tuple(pattern))
        attempts += 1

    if len(patterns) < num_patterns:
        raise ValueError("ユニークなパターンを生成できませんでした。")

    # Keep behavior: set -> list conversion (order is not guaranteed)
    return [list(p) for p in patterns]


def generate_charger_current(df_command: pd.DataFrame) -> pd.DataFrame:
    """
    Stage I — Generate charger current from command values (Algorithm 1: Stage I).

    This function simulates the 1-second-step evolution of charger current:
      (i) hold until change starts (with first/next change-start delays),
      (ii) compute target current and ramp up/down,
      (iii) hold until next command timestamp (or until the end).

    Implementation note
    -------------------
    - This implementation records 1-second values and then quantizes (rounds) to 0.1 A.
      (The paper also discusses sampling intervals; here we keep the original code’s behavior.)
      :contentReference[oaicite:2]{index=2}
    """
    records: List[Tuple[pd.Timestamp, float]] = []

    charger_current = 0.0
    current_time = df_command.at[0, "timestamp"]
    n_steps = len(df_command)

    for i in range(n_steps):
        # ------------------------------------------------------------
        # i == 0: initial hold until the next command timestamp
        # ------------------------------------------------------------
        if i == 0:
            next_time = df_command.at[i + 1, "timestamp"]
            current_time = _append_constant_current(records, current_time, next_time, charger_current)
            continue

        # ------------------------------------------------------------
        # (i) hold until change starts (delay depends on whether it's the first change)
        # ------------------------------------------------------------
        delay_s = FIRST_CHANGE_START_DELAY_S if i == 1 else NEXT_CHANGE_START_DELAY_S
        change_start_time = df_command.at[i, "timestamp"] + pd.Timedelta(seconds=delay_s)
        current_time = _append_constant_current(records, current_time, change_start_time, charger_current)

        # ------------------------------------------------------------
        # (ii) compute target and ramp
        #   target = c - b(c), where b(c) = a*c + b
        # ------------------------------------------------------------
        command_value = df_command.at[i, "command_value"]
        offset = OFFSET_COEF * command_value + OFFSET_INTERCEPT
        target_current = command_value - offset

        if charger_current < target_current:
            # Ramp-up rate depends on command increment Δc
            command_diff = command_value - df_command.at[i - 1, "command_value"]
            ramp_up_rate = RAMP_UP_COEF * command_diff + RAMP_UP_INTERCEPT
            current_time, charger_current = _ramp_up_to_target(
                records, current_time, charger_current, target_current, ramp_up_rate
            )
        else:
            # Ramp-down uses a constant rate
            current_time, charger_current = _ramp_down_to_target(
                records, current_time, charger_current, target_current, RAMP_DOWN_A_PER_S
            )

        # ------------------------------------------------------------
        # (iii) hold until next command timestamp (or finish at +60 sec from last command)
        # ------------------------------------------------------------
        if i < n_steps - 1:
            next_time = df_command.at[i + 1, "timestamp"]
            current_time = _append_constant_current(records, current_time, next_time, charger_current)
        else:
            end_time = df_command.at[i, "timestamp"] + pd.Timedelta(seconds=60)
            current_time = _append_constant_current(records, current_time, end_time, charger_current)

    df = pd.DataFrame(records, columns=["timestamp", "charger_current"]).set_index("timestamp")
    df["charger_current"] = df["charger_current"].round(CHARGER_QUANTIZATION_DECIMALS)
    return df


def generate_ev_current(df_charger_generation: pd.DataFrame) -> pd.DataFrame:
    """
    Stage II — Generate EV current from charger current (Algorithm 1: Stage II).

    EV current at time t is obtained by:
      - reading the charger current at (t + 5 sec),
      - applying ceiling (quantization to 1 A).

    Note
    ----
    The original code uses a 1-second index and checks exact existence of (t + 5 sec).
    This behavior is preserved as-is.
    """
    records = []
    offset = pd.Timedelta(seconds=CHARGER_TO_EV_TIME_OFFSET_S)

    for t in df_charger_generation.index:
        t_shifted = t + offset

        if t_shifted in df_charger_generation.index:
            charger_current = df_charger_generation.at[t_shifted, "charger_current"]
            ev_current = math.ceil(charger_current)  # 1 A quantization (ceiling)
        else:
            ev_current = records[-1]["ev_current"] if records else 0

        records.append({"timestamp": t, "ev_current": ev_current})

    return pd.DataFrame(records).set_index("timestamp")


def generate_charger_ev_current(command_patterns, initial_timestamp, tau=60):
    """
    Generate 1-second-resolution charger and EV current time series for each command pattern.

    Parameters
    ----------
    command_patterns : Sequence[Sequence[int]]
        List of command sequences (each sequence corresponds to one charger).
    initial_timestamp : str | pd.Timestamp
        Start time for command patterns.
    tau : int
        Command update period [sec].

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        df_charger_generations:
          index: timestamp, columns: Charger_1, Charger_2, ...
        df_ev_generations:
          index: timestamp, columns: EV_1, EV_2, ...
    """
    charger_series_list = []
    ev_series_list = []
    charger_columns = []
    ev_columns = []

    df_charger_generation = None  # keep last one for index extraction (preserves original behavior)

    for idx, command_values in enumerate(command_patterns):
        charger_columns.append(f"Charger_{idx + 1}")
        ev_columns.append(f"EV_{idx + 1}")

        df_command = get_command_data(initial_timestamp, command_values, tau=tau)

        df_charger_generation = generate_charger_current(df_command)
        charger_series_list.append(df_charger_generation.values.flatten())

        df_ev_generation = generate_ev_current(df_charger_generation)
        ev_series_list.append(df_ev_generation["ev_current"].values)

    index = df_charger_generation.index

    charger_array = np.column_stack(charger_series_list) if charger_series_list else np.empty((0, 0))
    ev_array = np.column_stack(ev_series_list) if ev_series_list else np.empty((0, 0))

    df_charger_generations = pd.DataFrame(charger_array, index=index, columns=charger_columns)
    df_ev_generations = pd.DataFrame(ev_array, index=index, columns=ev_columns)

    return df_charger_generations, df_ev_generations