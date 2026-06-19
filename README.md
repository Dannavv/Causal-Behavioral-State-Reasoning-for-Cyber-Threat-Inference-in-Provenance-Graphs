# ZeroCausal: Provable, Zero-Label Causal Anomaly Detection for APTs in Provenance Graphs

ZeroCausal is a novel intrusion detection framework designed to detect Advanced Persistent Threats (APTs) in provenance graphs with **zero labels, zero clean benign training data, and zero offline retraining**. It discovers online causal invariances (baseline Structural Causal Models) directly from unlabeled system logs and flags anomalies as violations of these learned causal equations.

---

## 🚀 1. Installation & Environment Setup

This project requires Python 3.9+ and uses a virtual environment to manage dependencies.

###AC Step 1: Create and Activate Virtual Environment
```bash
python3 -m venv .venv_zc
source .venv_zc/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📂 2. Directory Structure

```
.
├── zerocausal_core.py          # Core logic (change-point detection, OLS, SCM regression, conformal prediction)
├── 05_evaluate_zerocausal.py   # OpTC evaluation pipeline
├── 09_evaluate_additional_datasets.py # DARPA TC3 & NODLINK simulated logs generator and evaluator
├── 10_plot_comparisons.py      # Comparative multi-benchmark ROC curve generator (Figure 2)
├── 11_sensitivity_analysis.py  # Noise sensitivity & conformal threshold tracking generator (Figures 3 & 4)
├── generate_architecture_diagram.py # Matplotlib script for architecture diagram (Figure 1)
├── configs/
│   ├── default.yaml            # Default hyperparameters for TC3 and NODLINK
│   └── tuned.yaml              # Random-search-tuned parameters for OpTC
├── results/
│   └── final/                  # Final reproducible benchmarks, logs, and figures
├── plots/                      # Generated figures
├── requirements.txt            # Package list with exact version pins
└── README.md                   # This instruction file
```

---

## 📈 3. Step-by-Step Replication Commands

To reproduce the exact results and figures presented in the paper, execute the following commands in order:

### Step 3.1: Run the Final Benchmark Suite
Execute the three evaluation pipelines with fixed seeds (42). This will populate the logs directory.

1. **OpTC (Tuned)**:
   ```bash
   python3 05_evaluate_zerocausal.py --a-p 0.3632 --b-p 3.5284 --a-r 5.8843 --b-r 0.1495 --target-fpr 0.0822 --conformal-lr 0.0204 --run-name optc_tuned
   ```
2. **DARPA TC3 (Default)**:
   ```bash
   python3 09_evaluate_additional_datasets.py --dataset tc3 --baseline --run-name tc3_trace_default
   ```
3. **NODLINK (Default)**:
   ```bash
   python3 09_evaluate_additional_datasets.py --dataset nodlink --baseline --run-name nodlink_default
   ```

### Step 3.2: Generate the Paper Figures
Run the plotting scripts to create vector graphics and statistical evaluations. All figures are outputted directly to `results/final/` and copied to `plots/`.

1. **Figure 1 (Architecture Diagram)**:
   ```bash
   python3 generate_architecture_diagram.py
   ```
2. **Figure 2 (Comparative ROC Curve Overlay)**:
   ```bash
   python3 10_plot_comparisons.py
   ```
3. **Figures 3 & 4 (Conformal Threshold Adaptation & Noise Sensitivity)**:
   ```bash
   python3 11_sensitivity_analysis.py
   ```

### Step 3.3: Package Final Logs
Copy the newly generated logs into the `results/final/` folder for packaging:
```bash
cp logs/optc_tuned_* logs/tc3_trace_default_* logs/nodlink_default_* results/final/
```

---

## 🛠️ 4. Hyperparameter Summary

Configurations are stored in YAML format in the `configs/` folder. Key parameters include:
- `pcmci_alpha`: Significance level for the conditional independence tests.
- `a_p`, `b_p`, `a_r`, `b_r`: Parameters shaping the Beta distributions for the causal and residual anomaly score components.
- `target_fpr`: Target False Positive Rate for conformal calibration.
- `conformal_lr`: Adjusting step size for conformal threshold learning.
- `detector_short`/`detector_long`/`detector_threshold`: Windows and Z-score parameters for sliding-window change-point detection.
