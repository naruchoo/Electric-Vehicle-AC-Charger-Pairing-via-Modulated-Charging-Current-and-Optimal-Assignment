import sys
import pandas as pd
import numpy as np
import os

# Set random seed to ensure reproducibility of command-pattern generation
np.random.seed(42)

cwd = os.getcwd()

# --- charging-current generation model ---
GEN_DIR = f"{cwd}/charging_current_generation"
sys.path.append(GEN_DIR)

from charging_current_generation_model import (
    generate_command_patterns,
    generate_charger_ev_current,
)

# --- correlation & pair-identification (matching) ---
CORR_DIR = f"{cwd}/pair_identification"
sys.path.append(CORR_DIR)

from Correlation import (
    calculate_correlation,
    perform_matching_correlation
)

# Set output directory
out_dir = os.path.join(
    os.getcwd(),
    "case_study/sensitivity_analysis/i_a_command_update_interval/accuracy/Correlation"
)

if __name__ == "__main__":
    # Set a common initial timestamp for issuing the command pattern
    initial_timestamp = "2024-05-23 15:59:13"

    # Generate command patterns (i.e., unique command patterns assigned to chargers)
    num_patterns = 300  # number of EV–charger pairs
    command_patterns = generate_command_patterns(num_patterns)

    # Set charger-side and EV-side sampling intervals
    charger_interval = 5
    ev_intervals = [30, 60]

    # Candidate command update periods (τ) [sec]
    taus = list(range(60, 29, -5))  # [60, 55, 50, 45, 40, 35, 30]

    for ev_interval in ev_intervals:
        for tau in taus:
            # Generate charger current and EV current using the charging-current generation model,
            # with command update period τ
            df_charger_generation, df_ev_generation = generate_charger_ev_current(
                command_patterns, initial_timestamp, tau=tau
            )

            results = []

            # Sweep the time delay from command value to EV current measurement and evaluate pairing accuracy
            for offset in range(30):
                print(
                    f"[EV sampling interval={ev_interval}s, τ={tau}s, "
                    f"time delay to EV measurement={offset}s] Computing..."
                )

                # Subsample from the 1-second simulation step to the specified sampling intervals
                df_charger = df_charger_generation.iloc[0::charger_interval]
                df_ev = df_ev_generation.iloc[offset::ev_interval]

                # Compute the correlation coefficient matrix (similarity)
                df_corr = calculate_correlation(df_charger, df_ev)

                # Perform pair-identification (matching) and evaluate pairing (matching) accuracy
                _, acc = perform_matching_correlation(df_corr)

                print(f"    -> Pairing (matching) accuracy: {acc:.4f}")
                results.append({
                    "time_delay_command_to_ev_measurement_s": offset,
                    "pairing_accuracy": acc
                })

            # Save results to CSV
            df_result = pd.DataFrame(results)
            fname = f"{num_patterns}p_evint{ev_interval}_tau{tau}_CorrAcc.csv"
            df_result.to_csv(os.path.join(out_dir, fname), index=False)
            print(f"-> Saved: {fname}")
