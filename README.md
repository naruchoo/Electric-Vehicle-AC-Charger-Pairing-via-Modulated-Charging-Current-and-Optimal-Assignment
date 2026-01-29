Markdown# Electric Vehicle-AC Charger Pairing via Modulated Charging Current and Optimal Assignment

This repository contains the official implementation of the paper:

> **Electric Vehicle-AC Charger Pairing via Modulated Charging Current and Optimal Assignment** > Narutaka Nomura, Masaki Imanaka, Jiseok Yang, Hiroyuki Baba, Daisuke Kodaira  
> *Submitted to IEEE Transactions on Transportation Electrification*

## 📝 Abstract

[cite_start]Bidirectional communication between electric vehicles (EVs) and chargers is essential for enabling plug-and-charge services and leveraging EVs as flexible grid resources[cite: 5]. [cite_start]However, many existing AC charging standards lack mechanisms for exchanging identification data[cite: 6].

[cite_start]We address this gap by scaling **Modulated Charging Current Technology (MCCT)**, which identifies a connected EV-charger pair by comparing their measured current waveforms without any additional hardware[cite: 7]. [cite_start]Our framework constructs a cost matrix from two complementary waveform-similarity metrics (**Correlation-Euclidean** and **Correlation-DTW**) and solves the optimal assignment problem using the **Hungarian method**[cite: 8, 56, 58, 652, 653].

[cite_start]Simulations using 300 EV-charger pairs demonstrate that the proposed framework maintains pairing accuracy above 99% when both charger- and EV-side sampling intervals are $\le 30$ sec[cite: 9, 620].

## 📂 Repository Structure

The directory structure corresponds directly to the sections of the manuscript:

```text
.
├── charging_current_generation/   # [Section III] Data Generation Model
[cite_start]│   └── charging_current_generation_model.py  # Generates synthetic current waveforms [cite: 147]
├── pair_identification/           # [Section IV] Pairing Algorithm & Metrics
│   ├── Correlation.py             # Baseline Metric (Correlation only)
[cite_start]│   ├── Correlation_Euclidean.py   # Proposed Metric 1 (Correlation + Euclidean) [cite: 378, 652]
[cite_start]│   └── Correlation_DTW.py         # Proposed Metric 2 (Correlation + DTW) [cite: 378, 652]
[cite_start]├── case_study/                    # [Section V] Simulation & Analysis [cite: 400]
│   ├── sensitivity_analysis/
[cite_start]│   │   ├── i_a_command_update_interval/             # [V-B.1] Effect of Command Update Interval [cite: 472]
[cite_start]│   │   ├── ii_ab_Charger_and_EV_sampling_interval/  # [V-B.2] Effect of Sampling Intervals [cite: 558]
[cite_start]│   │   └── ii-d_Time_delay_.../                     # [V-B.3] Effect of Time Delay (Work in Progress) [cite: 588]
[cite_start]│   └── scalability_analysis/      # [V-C] Scalability Analysis (Work in Progress) [cite: 608]
└── requirements.txt
💻 Installation1. PrerequisitesPython 3.10 or higherWe recommend creating a virtual environment (e.g., via conda or venv) to manage dependencies.2. Install DependenciesInstall the required libraries using pip:Bashpip install -r requirements.txt
Key Dependencies:numpy, pandas: Data manipulationscipy: Optimization (Hungarian method)fastdtw: Dynamic Time Warping calculationmatplotlib, seaborn, japanize-matplotlib: Visualization🚀 Usage⚠️ Important: All scripts must be executed from the root directory of this repository. The scripts rely on os.getcwd() to resolve module paths.1. Data Generation Model (Section III)The core simulation logic is defined in charging_current_generation/. You can import the model in your own scripts as follows:Pythonfrom charging_current_generation.charging_current_generation_model import generate_charger_ev_current
2. Sensitivity Analysis (Section V-B)You can reproduce the experimental results presented in the paper by running the scripts in case_study/.V-B.1 Effect of Command Update Interval (Figs. 11 & 12)This experiment evaluates pairing accuracy by sweeping the command update interval ($\tau$) and time delays.Bash# Run the simulation for Correlation-DTW
python case_study/sensitivity_analysis/i_a_command_update_interval/i_a_Correlation_DTW.py

# Run for Correlation-Euclidean and Baseline (Correlation)
python case_study/sensitivity_analysis/i_a_command_update_interval/i_a_Correlation_Euclidean.py
python case_study/sensitivity_analysis/i_a_command_update_interval/i_a_Correlation.py

# Visualize the results
python case_study/sensitivity_analysis/i_a_command_update_interval/i_a_Accuracy_visualization.py
Output: Results (CSVs) are saved in case_study/sensitivity_analysis/i_a_command_update_interval/accuracy/.V-B.2 Effect of Sampling Intervals (Fig. 13)This experiment performs a grid search over charger-side and EV-side sampling intervals (5, 10, 30, 60 sec).Bash# Run batch simulations for both metrics
python case_study/sensitivity_analysis/ii_ab_Charger_and_EV_sampling_interval/ii_ab_batch_run.py

# Visualize the heatmaps
python case_study/sensitivity_analysis/ii_ab_Charger_and_EV_sampling_interval/ii_ab_Accuracy_visualization.py
Output: Results are saved in case_study/sensitivity_analysis/ii_ab_Charger_and_EV_sampling_interval/accuracy/.📊 MethodologyThe pairing framework consists of three main steps implemented in pair_identification/:Similarity Calculation:Correlation-Euclidean: Combines Pearson correlation (global shape) and Euclidean distance (local deviation).Correlation-DTW: Uses Dynamic Time Warping (DTW) to robustly handle time offsets and non-uniform sampling.Cost Matrix Construction: Similarities are normalized and converted into a cost matrix.Optimal Assignment: The Hungarian method (via scipy.optimize.linear_sum_assignment) is applied to find the global optimal one-to-one matching between $N$ chargers and $N$ EVs.🛡️ LicenseThis project is licensed under the MIT License - see the LICENSE file for details.📚 CitationIf you use this code or dataset in your research, please cite our paper:コード スニペット@article{Nomura2025MCCT,
  title={Electric Vehicle-AC Charger Pairing via Modulated Charging Current and Optimal Assignment},
  author={Nomura, Narutaka and Imanaka, Masaki and Yang, Jiseok and Baba, Hiroyuki and Kodaira, Daisuke},
  journal={Submitted to IEEE Transactions on Transportation Electrification},
  year={2025}
}