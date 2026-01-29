# Electric Vehicle–AC Charger Pairing via Modulated Charging Current and Optimal Assignment

This repository contains the official implementation of the paper:

> **Electric Vehicle-AC Charger Pairing via Modulated Charging Current and Optimal Assignment**  
> *Author: Narutaka Nomura, Masaki Imanaka, Jiseok Yang, Hiroyuki Baba, Daisuke Kodaira*  
> *Submitted to IEEE Transactions on Transportation Electrification*

## 📝 Abstract

Bidirectional communication between electric vehicles (EVs) and chargers is essential for enabling plug-and-charge services that support automatic vehicle identification and payment and for leveraging EVs as a flexible resource to balance fluctuations in renewable energy generation. However, many existing AC charging standards lack mechanisms for exchanging data such as state of charge, leaving service providers and grid operators blind to which EV is connected to which charger.  

We address this gap by scaling Modulated Charging Current Technology (MCCT)–which identifies a connected EV–charger pair by comparing their measured current waveforms without any additional hardware–to settings with multiple simultaneous connections. Our framework constructs a cost matrix from two complementary waveform-similarity metrics and then solves the resulting optimal assignment with the Hungarian method to obtain one-to-one pairing.  

Simulations using 300 EV–charger pairs show that the proposed framework maintains pairing accuracy above 99% when both the charger‑ and EV‑side sampling intervals are ≤ 30 sec, demonstrating its practicality for today’s widely deployed AC infrastructure.

## 📂 Repository Structure

The directory structure corresponds directly to the sections of the manuscript:

```text
.
├── charging_current_generation/   # [Section III] DATASET
│   └── charging_current_generation_model.py  # [III-B] Charging-current generation model
├── pair_identification/           # [Section IV] PAIR-IDENTIFICATION METHOD
│   ├── Correlation.py             # Baseline Metric (Correlation only)
│   ├── Correlation_Euclidean.py   # Proposed Metric 1 (Correlation + Euclidean)
│   └── Correlation_DTW.py         # Proposed Metric 2 (Correlation + DTW)
├── case_study/                    # [Section V] CASE STUDY
│   ├── sensitivity_analysis/      # [V-B] Sensitivity Analysis of the Charging-Current Generation Model Parameters
│   │   ├── i_a_command_update_interval/             # [V-B.1] Interval for Updating Command Value (I-a)
│   │   ├── ii_ab_Charger_and_EV_sampling_interval/  # [V-B.2] Charger‑side and EV‑side Sampling Interval (II‑a, II-b)
│   │   └── ii-d_Time_delay_.../                     # [V-B.3] Time delay from command value to EV current measurement (II-d)
│   └── scalability_analysis/      # [V-C] Scalability with the Number of EV–Charger Pairs
└── requirements.txt

```

---

## 💻 Installation

### 1. Prerequisites

* Python 3.10 (or higher)
* Note: We recommend using Python 3.10 for stability and reproducibility. Higher versions may encounter compatibility issues with specific dependencies.

#### Option A: Manual Installation (Local Environment)

1. **Clone the repository:**
```bash
git clone https://github.com/naruchoo/Electric-Vehicle-AC-Charger-Pairing-via-Modulated-Charging-Current-and-Optimal-Assignment.git
cd Electric-Vehicle-AC-Charger-Pairing-via-Modulated-Charging-Current-and-Optimal-Assignment

```

2. **Create and activate a virtual environment:**
```bash
# Using venv
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# OR Using Conda
conda create -n ev-pairing python=3.10.14
conda activate ev-pairing

```

3. **Install dependencies:**
```bash
pip install -r requirements.txt

```

#### Option B: Development Container (VS Code + Docker)

This is the **recommended method** to ensure a consistent environment without polluting your local system.

1. In VS Code, open the **Command Palette**:
* **Mac**: `Shift` + `Command` + `P`
* **Windows/Linux**: `Ctrl` + `Shift` + `P`


2. Search for and select **"Dev Containers: Clone Repository in Container Volume..."**
3. Paste the following repository URL:
`https://github.com/naruchoo/Electric-Vehicle-AC-Charger-Pairing-via-Modulated-Charging-Current-and-Optimal-Assignment.git`
4. VS Code will automatically:
* Build a container based on `Python 3.10-bullseye`.
* Install all required dependencies from `requirements.txt` inside the container.

---

## 🚀 Usage

**⚠️ Important:** All scripts must be executed from the **root directory** of this repository. The scripts rely on `os.getcwd()` to resolve module paths.

### 1. Charging-current generation model (Section III-B)

The data generation logic is defined in `charging_current_generation/`. This model generates the charger sequence  and EV sequence  based on the command patterns and simulation parameters defined in **Table I** of the manuscript.

### 2. Sensitivity Analysis (Section V-B)

You can reproduce the case study results presented in **Section V** by running the scripts in `case_study/`.

#### B.1 Interval for Updating Command Value (I-a) (Figs. 11 & 12)

This experiment evaluates pairing accuracy by sweeping the **interval for updating command value (I-a)** and comparing different sampling configurations.

```bash
# Run the simulation
python case_study/sensitivity_analysis/i_a_command_update_interval/i_a_Correlation_DTW.py
python case_study/sensitivity_analysis/i_a_command_update_interval/i_a_Correlation_Euclidean.py
python case_study/sensitivity_analysis/i_a_command_update_interval/i_a_Correlation.py

```
*Output:* CSVs are saved in `case_study/sensitivity_analysis/i_a_command_update_interval/accuracy/`.

```bash
# Visualize the results
python case_study/sensitivity_analysis/i_a_command_update_interval/i_a_Accuracy_visualization.py

```
*Output:* Figures are saved in `case_study/sensitivity_analysis/i_a_command_update_interval/figures/`.

#### B.2 Charger-side and EV-side Sampling Interval (II-a, II-b) (Fig. 13)

This experiment performs a grid search over the **charger-side (II-a)** and **EV-side (II-b)** sampling intervals (5, 10, 30, 60 sec) to determine the conditions for robust assignment.

```bash
# Run batch simulations for both metrics
python case_study/sensitivity_analysis/ii_ab_Charger_and_EV_sampling_interval/ii_ab_batch_run.py

```

*Output:* Results are saved in `case_study/sensitivity_analysis/ii_ab_Charger_and_EV_sampling_interval/accuracy/`.

```bash
# Visualize the heatmaps
python case_study/sensitivity_analysis/ii_ab_Charger_and_EV_sampling_interval/ii_ab_Accuracy_visualization.py

```

*Output:* Heatmaps are saved in `case_study/sensitivity_analysis/ii_ab_Charger_and_EV_sampling_interval/figures/`.

#### B.3 Time delay from command value to EV current measurement (II-d) (Fig. 14)
Making...

### 3. Scalability Analysis (Section V-C) (Fig. 15)
Making...

---

## 📊 Methodology

The proposed **Pair-identification method** consists of the following steps, corresponding to the workflow described in **Section IV** (Fig. 6):

1. **Similarity Matrix Calculation** (Section IV-A):
* **Correlation coefficient**: Captures the overall similarity of current waveforms.
* **Distance-based metrics**: Includes **Euclidean distance** (point-to-point deviation) and **Dynamic Time Warping (DTW)** (robust to time shifts and non-uniform sampling).


2. **Construction of Evaluation Matrix** (Section IV-B & C):
* **Normalization**: Scaling similarity metrics to the range 0 to 1.
* **Hadamard product**: Combining the normalized correlation matrix and the distance-based matrix to produce the evaluation matrix.


3. **Optimal Assignment** (Section IV-D & E):
* **One-minus transformation**: Converting the evaluation matrix into a cost matrix.
* **Hungarian method**: Solving the assignment problem to minimize total cost, yielding the globally optimal one-to-one pairing.

---

## 📚 Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{Nomura2025MCCT,
  title={Electric Vehicle-AC Charger Pairing via Modulated Charging Current and Optimal Assignment},
  author={Nomura, Narutaka and Imanaka, Masaki and Yang, Jiseok and Baba, Hiroyuki and Kodaira, Daisuke},
  journal={Submitted to IEEE Transactions on Transportation Electrification},
  year={2025}
}

```
