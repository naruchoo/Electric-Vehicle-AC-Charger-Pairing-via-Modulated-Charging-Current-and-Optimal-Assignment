import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Constants and configuration ─────────────────────────
ROOT = os.getcwd()
BASE_DIR = os.path.join(ROOT, "case_study/sensitivity_analysis/i_a_command_update_interval/accuracy")
OUT_DIR = os.path.join(ROOT, "case_study/sensitivity_analysis/i_a_command_update_interval/figures")
os.makedirs(OUT_DIR, exist_ok=True)

METHOD_DIRS = {
    'Corr':   'Correlation',
    'DTW':    'Correlation_DTW',
    'CorEuc': 'Correlation_Euclidean'
}

METHODS = ['CorEuc', 'CorrDTW', 'Corr']
LABELS = {
    'CorEuc':  'Correlation-Euclidean',
    'CorrDTW': 'Correlation-DTW',
    'Corr':    'Baseline (Correlation)'
}
COLORS = {
    'CorEuc':  '#228b22',
    'CorrDTW': '#ff8c00',
    'Corr':    '#376ea4'
}
MARKERS = {'CorEuc': 's', 'CorrDTW': 'o', 'Corr': '^'}
LINESTYLES = {'CorEuc': '-', 'CorrDTW': '-', 'Corr': 'dashed'}

TAU_LIST = [30, 35, 40, 45, 50, 55, 60]
EV_INTERVALS = [30, 60]
NUM_PATTERNS = 300


def get_filename(method: str, ev_int: int, tau: int) -> str:
    """Return the CSV filename for each method."""
    if method == 'Corr':
        return f"{NUM_PATTERNS}p_evint{ev_int}_tau{tau}_CorrAcc.csv"
    if method == 'CorrDTW':
        return f"{NUM_PATTERNS}p_evint{ev_int}_tau{tau}_CorrDTWAcc.csv"
    # CorEuc
    return f"{NUM_PATTERNS}_{ev_int}sInterval_CorEuc_tau{tau}s_accuracy.csv"


def load_accuracy(ev_int: int) -> dict[str, list[float]]:
    """Collect the mean accuracy (%) for each method and each τ."""
    avg_pct = {m: [] for m in METHODS}
    for tau in TAU_LIST:
        for m in METHODS:
            key = 'DTW' if m == 'CorrDTW' else m
            dir_path = os.path.join(BASE_DIR, METHOD_DIRS[key])
            fname = get_filename(m, ev_int, tau)
            path = os.path.join(dir_path, fname)
            if not os.path.exists(path):
                print(f"!! missing: {path}")
                avg_pct[m].append(np.nan)
                continue

            df = pd.read_csv(path)
            col = 'accuracy' if 'accuracy' in df.columns else 'Matching_accuracy'
            avg_pct[m].append(df[col].mean() * 100.0)
    return avg_pct


def plot_bar(avg_pct: dict[str, list[float]], ev_int: int) -> None:
    x = np.arange(len(TAU_LIST))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, m in enumerate(METHODS):
        ax.bar(
            x + (i - 1) * width,
            avg_pct[m],
            width,
            label=LABELS[m],
            color=COLORS[m],
            zorder=3
        )
    ax.set_axisbelow(True)
    ax.set_xlabel('Interval for updating command value [s]', fontsize=26)
    ax.set_ylabel('Accuracy [%]', fontsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels(TAU_LIST, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(0, 102)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=20, loc='best')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"i_a_accuracy_ev_interval:{ev_int}s_bar.png")
    fig.savefig(path)
    plt.show()
    plt.close(fig)
    print(f"Saved bar plot: {os.path.basename(path)}")


def plot_line(avg_pct: dict[str, list[float]], ev_int: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_axisbelow(True)
    for m in METHODS:
        ax.plot(
            TAU_LIST,
            avg_pct[m],
            marker=MARKERS[m],
            linestyle=LINESTYLES[m],
            linewidth=2,
            color=COLORS[m],
            label=LABELS[m],
            zorder=5
        )
    ax.set_xlabel('Interval for updating command value [sec]', fontsize=26, labelpad=10)
    ax.set_ylabel('Accuracy [%]', fontsize=26)
    ax.set_xticks(TAU_LIST)
    ax.tick_params(axis='x', labelsize=20, length=5)
    ax.tick_params(axis='y', labelsize=20, length=5)
    ax.set_ylim(0, 102)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.grid(True, linestyle='-', linewidth=0.8, alpha=1.0)
    ax.xaxis.grid(True, linestyle='-', alpha=0.5)
    ax.legend(fontsize=20, loc='lower right')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"i_a_accuracy_ev_interval:{ev_int}s_line.png")
    fig.savefig(path)
    plt.show()
    plt.close(fig)
    print(f"Saved line plot: {os.path.basename(path)}")


def main():
    for ev in EV_INTERVALS:
        avg = load_accuracy(ev)
        df = pd.DataFrame(avg, index=TAU_LIST)
        print(f"\n[EV interval = {ev}s]\n", df, "\n")
        plot_bar(avg, ev)
        plot_line(avg, ev)


if __name__ == "__main__":
    main()