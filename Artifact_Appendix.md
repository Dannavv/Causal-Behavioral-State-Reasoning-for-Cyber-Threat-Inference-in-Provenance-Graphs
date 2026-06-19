# ZeroCausal: Artifact Appendix

This appendix provides instructions for reviewers and researchers to set up the environment, run the evaluation pipelines, and reproduce all results and figures presented in the paper.

## 1. Abstract
The artifact consists of the core Python implementation of **ZeroCausal**, evaluation scripts for five benchmarks (DARPA OpTC, simulated DARPA TC3, simulated NODLINK, BETH, and StreamSpot), configurations, and plotting tools. The evaluations run online causal discovery (PCMCI with ParCorr), fit Structural Causal Models (SCMs), monitor anomalies using the Causal Anomaly Score (CAS), and perform online conformal calibration. Using a standard Linux host, all benchmarks and figures can be generated in under 5 minutes (excluding the one-time high-dimensional PCMCI discovery on BETH).

## 2. Artifact Meta-Information
- **Algorithm**: PCMCI causal discovery, OLS residual regression, conformal calibration, adaptive window change-point detection.
- **Program**: Python 3.9+.
- **Compilation/Run-time Environment**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows (via WSL2).
- **Data Sets**: DARPA OpTC (partial representative edge-list provided for artifact evaluation), simulated DARPA TC3 (generated programmatically), simulated NODLINK (generated programmatically), BETH (privilege-escalation log), and StreamSpot (drive-by download graph telemetry).
- **Run-time State**: All random seeds are fixed to `42` for exact reproducibility.
- **Execution Time**: ~20-30 seconds for OpTC, ~2 seconds each for TC3 and NODLINK. Total reproduction time is less than 5 minutes.
- **Output**: Summary metrics in JSON, step logs in CSV, and vector plots in PNG/SVG format.

## 3. Hardware and Software Requirements
### 3.1 Hardware Prerequisites
- **CPU**: Standard x86 or ARM64 processor (at least 2 cores recommended).
- **Memory**: Minimum 4 GB RAM.
- **Storage**: Minimum 500 MB of free space (to store the processed OpTC CSV edge lists and generated plots).

### 3.2 Software Prerequisites
- **OS**: Linux, macOS, or Windows (WSL2).
- **Python**: Version 3.9 to 3.11.
- **System Packages**: `graphviz` (optional, for visualization).

---

## 4. Installation & Environment Setup

### 4.1 Step 1: Clone or Access the Workspace
Ensure you are in the directory containing the source files:
```bash
cd /DATA/shourya_2211mc14/Arp/work2
```

### 4.2 Step 2: Establish the Virtual Environment
Create and activate a Python virtual environment:
```bash
python3 -m venv .venv_zc
source .venv_zc/bin/activate
```

### 4.3 Step 3: Install Required Dependencies
Install the required packages using the pinned versions in `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Note: Installing these dependencies takes approximately 1 to 2 minutes depending on network speed.*

---

## 5. Reproduction of Experimental Results

Follow these steps to run the benchmark suite and generate the comparative plots.

### 5.1 Run the Evaluation Pipeline
Execute the five benchmark evaluations to generate the underlying logs.

1. **Replicate DARPA OpTC Tuned Run**:
   ```bash
   python3 05_evaluate_zerocausal.py --a-p 0.3632 --b-p 3.5284 --a-r 5.8843 --b-r 0.1495 --target-fpr 0.0822 --conformal-lr 0.0204 --run-name optc_tuned
   ```
   *Expected Output*:
   - Streaming evaluation completes in ~18-20 seconds.
   - ZeroCausal AUC: **0.8825**.
   - Outputs: `logs/optc_tuned_summary.json`, `logs/optc_tuned_steps.csv`, `logs/optc_tuned_calib_steps.csv`.

2. **Replicate DARPA TC3 Simulation**:
   ```bash
   python3 09_evaluate_additional_datasets.py --dataset tc3 --baseline --run-name tc3_trace_default
   ```
   *Expected Output*:
   - Runs in ~1.5 seconds.
   - ZeroCausal AUC: **1.0000** | Isolation Forest AUC: **0.5513**.
   - Outputs: `logs/tc3_trace_default_summary.json`, `logs/tc3_trace_default_steps.csv`.

3. **Replicate NODLINK Simulation**:
   ```bash
   python3 09_evaluate_additional_datasets.py --dataset nodlink --baseline --run-name nodlink_default
   ```
   *Expected Output*:
   - Runs in ~1.5 seconds.
   - ZeroCausal AUC: **1.0000** | Isolation Forest AUC: **0.5011**.
   - Outputs: `logs/nodlink_default_summary.json`, `logs/nodlink_default_steps.csv`.

4. **Replicate BETH Dataset Run**:
   ```bash
   python3 09_evaluate_additional_datasets.py --dataset beth --baseline --run-name beth_default
   ```
   *Expected Output*:
   - Runs in ~2.0 seconds.
   - ZeroCausal AUC: **0.9656** | Isolation Forest AUC: **0.9981**.
   - Outputs: `logs/beth_default_summary.json`, `logs/beth_default_steps.csv`.

5. **Replicate StreamSpot Dataset Run**:
   ```bash
   python3 09_evaluate_additional_datasets.py --dataset streamspot --baseline --run-name streamspot_default
   ```
   *Expected Output*:
   - Runs in ~2.5 seconds.
   - ZeroCausal AUC: **0.4991** | Isolation Forest AUC: **0.6425**.
   - Outputs: `logs/streamspot_default_summary.json`, `logs/streamspot_default_steps.csv`.

### 5.2 Generate and Replicate the Figures
Run the plotting scripts to compile the vector figures. The scripts write directly to `results/final/` and copy to `plots/`.

1. **Replicate Figure 1 (Pipeline Architecture)**:
   ```bash
   python3 generate_architecture_diagram.py
   ```
   *Output*: Generates `results/final/zerocausal_architecture.png`.

2. **Replicate Figure 2 (Multi-Benchmark ROC Comparison)**:
   ```bash
   python3 10_plot_comparisons.py
   ```
   *Output*: Generates `results/final/benchmark_comparison_roc.png`.

3. **Replicate Figures 3 & 4 (Conformal Adaptation & Noise Sensitivity)**:
   ```bash
   python3 11_sensitivity_analysis.py
   ```
   *Output*: Generates `results/final/threshold_adaptation_learning.png` and `results/final/noise_sensitivity_analysis.png`.

### 5.3 Consolidated Outputs
Copy the newly generated logs to the packaging folder:
```bash
cp logs/optc_tuned_* logs/tc3_trace_default_* logs/nodlink_default_* logs/beth_default_* logs/streamspot_default_* results/final/
```
All outputs are now organized and ready for verification in `results/final/`.
