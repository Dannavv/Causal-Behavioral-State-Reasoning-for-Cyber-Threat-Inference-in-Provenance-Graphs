import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest

# Import generator functions and evaluation pipeline components
import sys
sys.path.append(os.getcwd())
from zerocausal_core import (
    AdaptiveWindowDetector,
    CausalRegressionModel,
    HybridAnomalyScorer,
    ConformalCalibrator
)
import importlib
evaluate_additional_datasets = importlib.import_module("09_evaluate_additional_datasets")
generate_tc3_data = evaluate_additional_datasets.generate_tc3_data
generate_nodlink_data = evaluate_additional_datasets.generate_nodlink_data
learn_baseline = evaluate_additional_datasets.learn_baseline

def run_single_eval(dataset_name, noise_level, seed=42):
    """
    Evaluates both ZeroCausal and Isolation Forest on the given dataset and noise level.
    Returns (zerocausal_auc, iforest_auc, thresholds_history)
    """
    if dataset_name == "tc3":
        ts_df, labels, attack_edges, attack_indices = generate_tc3_data(noise_level=noise_level, seed=seed)
    else:
        ts_df, labels, attack_edges, attack_indices = generate_nodlink_data(noise_level=noise_level, seed=seed)
        
    split_idx = int(len(ts_df) * 0.6)
    train_df = ts_df.iloc[:split_idx].copy()
    test_df = ts_df.iloc[split_idx:].copy()
    
    # Inject Synthetic APT only into test_df
    for edge in attack_edges:
        if edge not in test_df.columns:
            test_df[edge] = 0
            
    for idx in attack_indices:
        test_idx = idx - split_idx
        if 0 <= test_idx < len(test_df) - 2:
            burst = max(np.random.poisson(5), 3)
            test_df.iloc[test_idx, test_df.columns.get_loc(attack_edges[0])] += burst
            test_df.iloc[test_idx + 1, test_df.columns.get_loc(attack_edges[1])] += burst
            test_df.iloc[test_idx + 2, test_df.columns.get_loc(attack_edges[2])] += burst
            
    # Sub-split training data for conformal calibration
    calib_split_idx = int(len(train_df) * 0.7)
    train_proper = train_df.iloc[:calib_split_idx].copy()
    train_calib = train_df.iloc[calib_split_idx:].copy()
    
    # Fit causal baseline
    p_matrix, var_names = learn_baseline(train_proper, alpha=0.01, dataset=dataset_name, seed=seed)
    
    # Fit causal regression models
    causal_model = CausalRegressionModel(p_matrix, var_names, tau_max=1, alpha=0.01)
    causal_model.fit(train_proper, std_floor=1.0)
    
    # Score calibration set
    scorer = HybridAnomalyScorer(d=len(var_names), w=0.5, floor=1.0)
    calib_scores = []
    
    for i in range(1, len(train_calib)):
        idx = calib_split_idx + i
        residuals, p_vals = causal_model.predict_and_residual(train_df, idx)
        score = scorer.score(p_vals, residuals, causal_model.residual_stds)
        calib_scores.append(score)
        
    calibrator = ConformalCalibrator(target_fpr=0.05, lr=0.05, alpha_init=0.05)
    calibrator.calibrate(calib_scores)
    
    # Setup test logs and history
    test_labels = labels[split_idx:]
    scores = np.zeros(len(test_df))
    thresholds_history = np.zeros(len(test_df))
    
    detector = AdaptiveWindowDetector(num_features=len(var_names), short_window=10, long_window=50, threshold=4.0)
    
    test_history = pd.concat([train_df.iloc[-1:], test_df])
    history_cols = list(test_history.columns)
    history_arr = test_history.to_numpy().copy()
    col_to_idx = {col: idx for idx, col in enumerate(history_cols)}
    var_indices = [col_to_idx[v] for v in var_names if v in col_to_idx]
    
    for i in range(1, len(test_df) + 1):
        actual_row = history_arr[i]
        
        # Adaptive Windowing
        feat_subset = actual_row[var_indices]
        change_detected = detector.update(feat_subset)
        if change_detected:
            # Refit causal regression model on the recent window of size detector.long_window
            refit_len = max(50, detector.long_window)
            start_idx = max(0, i - refit_len + 1)
            refit_df = pd.DataFrame(history_arr[start_idx : i + 1], columns=history_cols)[var_names].copy()
            causal_model.fit(refit_df, std_floor=1.0)
            
            # Recalibrate the conformal calibrator on this same window to update the scores queue
            new_calib_scores = []
            hist_df = pd.DataFrame(history_arr, columns=history_cols)
            for k in range(1, len(refit_df)):
                hist_idx = start_idx + k
                res_k, pvals_k = causal_model.predict_and_residual(hist_df, hist_idx)
                score_k = scorer.score(pvals_k, res_k, causal_model.residual_stds)
                new_calib_scores.append(score_k)
            
            calibrator.calibrate(new_calib_scores)
            
        # Predict on baseline
        hist_df = pd.DataFrame(history_arr, columns=history_cols)
        known_residuals, known_pvals = causal_model.predict_and_residual(hist_df, i)
        residuals = known_residuals.copy()
        p_vals = known_pvals.copy()
        
        # Handle structural novelties
        active_indices = np.where(actual_row > 0)[0]
        active_novel_edges = []
        for idx in active_indices:
            edge = history_cols[idx]
            if edge not in var_names:
                active_novel_edges.append(edge)
                residuals[edge] = float(actual_row[idx])
                p_vals[edge] = 1e-15
            
        # Score anomaly
        res_stds = causal_model.residual_stds.copy()
        for edge in active_novel_edges:
            res_stds[edge] = 0.1
            
        score = scorer.score(p_vals, residuals, res_stds)
        scores[i - 1] = score
        
        # Conformal prediction
        conf_pval = calibrator.compute_conformal_pvalue(score)
        alarm = 1.0 if conf_pval < calibrator.alpha else 0.0
        thresholds_history[i - 1] = calibrator.alpha
        calibrator.update_threshold(alarm)
        
        # Maintain rolling calibration window with current score if normal (no alarm)
        if alarm == 0.0:
            calibrator.update_calibration(score, max_size=245)
        
    # Calculate AUC
    zc_auc = roc_auc_score(test_labels, scores)
    
    # Run Isolation Forest Baseline
    train_full = train_proper.copy()
    for col in attack_edges:
        if col not in train_full.columns:
            train_full[col] = 0.0
    cols = list(test_df.columns)
    train_full = train_full[cols]
    
    iforest = IsolationForest(contamination=0.05, random_state=seed)
    iforest.fit(train_full.values)
    iforest_scores = -iforest.score_samples(test_df.values)
    iforest_auc = roc_auc_score(test_labels, iforest_scores)
    
    return zc_auc, iforest_auc, thresholds_history

def main():
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    datasets = ["tc3", "nodlink"]
    
    results = {d: {"zc": [], "iforest": []} for d in datasets}
    threshold_data = {}
    
    print("Starting sensitivity analysis (noise level 0.0 to 0.3)...")
    
    for d in datasets:
        for nl in noise_levels:
            zc_auc, if_auc, th_hist = run_single_eval(d, nl, seed=42)
            results[d]["zc"].append(zc_auc)
            results[d]["iforest"].append(if_auc)
            print(f"Dataset: {d.upper()} | Noise: {nl:.2f} | ZeroCausal AUC: {zc_auc:.4f} | Isolation Forest AUC: {if_auc:.4f}")
            
            # Save the threshold history at noise level 0.1 for plotting
            if abs(nl - 0.1) < 0.01:
                threshold_data[d] = th_hist
                
    # Plot 1: Sensitivity Analysis
    plt.figure(figsize=(10, 6))
    
    plt.plot(noise_levels, results["tc3"]["zc"], color='#3182ce', marker='o', lw=2.5, label='ZeroCausal (DARPA TC3)')
    plt.plot(noise_levels, results["tc3"]["iforest"], color='#e53e3e', marker='x', lw=1.5, linestyle='--', label='Isolation Forest (DARPA TC3)')
    plt.plot(noise_levels, results["nodlink"]["zc"], color='#319795', marker='s', lw=2.5, label='ZeroCausal (NODLINK)')
    plt.plot(noise_levels, results["nodlink"]["iforest"], color='#d69e2e', marker='^', lw=1.5, linestyle='--', label='Isolation Forest (NODLINK)')
    
    plt.xlabel('Noise Level (Gaussian Scaling Factor)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Anomaly Detection AUC Score', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('ZeroCausal vs. Isolation Forest Noise Sensitivity Analysis', fontsize=14, fontweight='bold', pad=15)
    plt.ylim([0.45, 1.05])
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc="lower left", fontsize=11, frameon=True, facecolor='white', edgecolor='#e2e8f0')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_dir = os.path.join(os.getcwd(), "results", "final")
    os.makedirs(out_dir, exist_ok=True)
    sens_plot_path = os.path.join(out_dir, "noise_sensitivity_analysis.png")
    plt.tight_layout()
    plt.savefig(sens_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved sensitivity plot to {sens_plot_path}")
    
    # Copy to plots folder
    os.makedirs("plots", exist_ok=True)
    shutil_copy(sens_plot_path, "plots/noise_sensitivity_analysis.png")
    
    # Copy sensitivity plot to artifacts
    for conv_id in ["b7d7d095-4573-49cf-933c-6f4730b480d7", "873c6033-7d1f-4ef6-9c05-38e627d9e107"]:
        shutil_copy(sens_plot_path, f"/DATA/shourya_2211mc14/.gemini/antigravity-ide/brain/{conv_id}/plots/noise_sensitivity_analysis.png")

    # Plot 2: Online Threshold Learning (Adaptation over Time)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    steps_tc3 = np.arange(len(threshold_data["tc3"]))
    
    # Left subplot: Host-Level Datasets
    ax1.plot(steps_tc3, threshold_data["tc3"], color='#3182ce', lw=2, label='DARPA TC3 (Target FPR = 5%)')
    ax1.plot(steps_tc3, threshold_data["nodlink"], color='#319795', lw=2, label='NODLINK (Target FPR = 5%)')
    
    # Load OpTC Tuned (std_floor=1.0, target_fpr=8.2%) threshold history
    optc_csv = "logs/optc_tuned_steps.csv"
    if os.path.exists(optc_csv):
        df_optc = pd.read_csv(optc_csv)
        optc_steps = np.arange(len(df_optc))
        ax1.plot(optc_steps, df_optc['threshold'].values, color='#d69e2e', lw=2, label='OpTC Tuned (Target FPR = 8.2%)')
        
    # Load BETH default threshold history
    beth_csv = "logs/beth_default_steps.csv"
    if os.path.exists(beth_csv):
        df_beth = pd.read_csv(beth_csv)
        beth_steps = np.arange(len(df_beth))
        ax1.plot(beth_steps, df_beth['threshold'].values, color='#38a169', lw=2, label='BETH (Target FPR = 5%)')
        
    ax1.set_xlabel('Streaming Step (Seconds)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Adaptive Conformal Threshold (α)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title('Host-Level Telemetry Adaptation', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9, frameon=True, facecolor='white', edgecolor='#e2e8f0')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right subplot: Graph-Level Dataset (StreamSpot)
    streamspot_csv = "logs/streamspot_default_steps.csv"
    if os.path.exists(streamspot_csv):
        df_ss = pd.read_csv(streamspot_csv)
        ss_steps = np.arange(len(df_ss))
        ax2.plot(ss_steps, df_ss['threshold'].values, color='#805ad5', lw=2, label='StreamSpot (Target FPR = 5%)')
        
    ax2.set_xlabel('Streaming Step (Windows)', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Adaptive Conformal Threshold (α)', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title('Graph-Level Provenance Adaptation', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9, frameon=True, facecolor='white', edgecolor='#e2e8f0')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle('Conformal Threshold (α) Online Learning Adaptation over Time', fontsize=14, fontweight='bold', y=0.98)
    thresh_plot_path = os.path.join(out_dir, "threshold_adaptation_learning.png")
    plt.tight_layout()
    plt.savefig(thresh_plot_path, dpi=300)
    plt.close()
    print(f"Saved threshold adaptation plot to {thresh_plot_path}")
    
    # Copy to plots folder
    shutil_copy(thresh_plot_path, "plots/threshold_adaptation_learning.png")
    
    # Copy threshold plot to artifacts
    for conv_id in ["b7d7d095-4573-49cf-933c-6f4730b480d7", "873c6033-7d1f-4ef6-9c05-38e627d9e107"]:
        shutil_copy(thresh_plot_path, f"/DATA/shourya_2211mc14/.gemini/antigravity-ide/brain/{conv_id}/plots/threshold_adaptation_learning.png")

def shutil_copy(src, dst):
    import shutil
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        print(f"Copied {src} to {dst}")
    except Exception as e:
        print(f"Failed to copy {src} to {dst}: {e}")

if __name__ == "__main__":
    main()
