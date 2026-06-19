import pandas as pd
import numpy as np
import os
import scipy.stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import IsolationForest
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import warnings
import argparse
import time
import json

# Import our core components
from zerocausal_core import (
    AdaptiveWindowDetector,
    CausalRegressionModel,
    HybridAnomalyScorer,
    ConformalCalibrator
)

warnings.filterwarnings("ignore")

# 1. Load Data & Build Time Series
def load_and_prep_data(csv_path="optc_edges.csv"):
    print("Loading graph edges...")
    edges_df = pd.read_csv(csv_path)
    edges_df['timestamp'] = pd.to_datetime(edges_df['timestamp'], format='ISO8601')
    edges_df['edge_type'] = edges_df['src_type'] + ":" + edges_df['src_id'] + " -> " + \
                            edges_df['action'] + " -> " + edges_df['dst_type'] + ":" + edges_df['dst_id']
    
    # Group into 1-second bins
    ts_df = edges_df.groupby([pd.Grouper(key='timestamp', freq='1s'), 'edge_type']).size().unstack(fill_value=0)
    
    # Filter highly sparse edges for stable baseline
    ts_df = ts_df.loc[:, (ts_df.sum(axis=0) > 5)]
    
    # Inject background noise for synthetic APT edges so they aren't trivial novelties
    apt_edges = ["PROCESS:word.exe -> SPAWNS_PROCESS -> PROCESS:powershell.exe", 
                 "PROCESS:powershell.exe -> WRITES_FILE -> FILE:registry.dat"]
    for edge in apt_edges:
        if edge not in ts_df.columns:
            ts_df[edge] = np.random.poisson(lam=0.01, size=len(ts_df))
            
    return ts_df

# 2. Learn Causal Baseline
def learn_baseline(train_df, alpha=0.01, tau_max=2, seed=42):
    cache_path = f"logs/pcmci_cache_optc_{seed}_alpha{alpha}_tau{tau_max}.pkl"
    if os.path.exists(cache_path):
        print(f"Loading cached PCMCI baseline from {cache_path}...")
        import pickle
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            return cache_data['p_matrix'], cache_data['var_names']
            
    print(f"Learning Causal Baseline on {train_df.shape[0]} windows...")
    var_names = train_df.columns.tolist()
    data = train_df.astype(float).values
    dataframe = pp.DataFrame(data, datatime=np.arange(len(train_df)), var_names=var_names)
    
    cond_ind_test = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)
    
    # Save to cache
    os.makedirs("logs", exist_ok=True)
    import pickle
    with open(cache_path, 'wb') as f:
        pickle.dump({'p_matrix': results['p_matrix'], 'var_names': var_names}, f)
        
    return results['p_matrix'], var_names

# 3. Inject Attacks and Score
def evaluate(ts_df, p_matrix, var_names, args):
    # Split into Train (first 60%) and Test (last 40%)
    split_idx = int(len(ts_df) * 0.6)
    train_df = ts_df.iloc[:split_idx].copy()
    test_df = ts_df.iloc[split_idx:].copy()
    
    # Sub-split training data for conformal calibration:
    # 70% for proper regression training, 30% for calibration scores
    calib_split_idx = int(len(train_df) * 0.7)
    train_proper = train_df.iloc[:calib_split_idx].copy()
    train_calib = train_df.iloc[calib_split_idx:].copy()
    
    # Fit causal regression models on train_proper
    print(f"Fitting Causal Regression models (regressor={args.regressor})...")
    causal_model = CausalRegressionModel(p_matrix, var_names, tau_max=args.tau_max, alpha=args.pcmci_alpha, regressor_type=args.regressor)
    causal_model.fit(train_proper, std_floor=args.std_floor)
    
    # Score calibration set
    print("Calibrating Conformal Prediction...")
    scorer = HybridAnomalyScorer(d=len(var_names), w=0.5, floor=args.std_floor)
    calib_scores = []
    calib_log = []
    
    # We need lag history for predict_and_residual, so we start after the lag (tau_max=1)
    # Using the pre-aligned full train_df instead of slow concatenations inside the loop!
    for i in range(args.tau_max, len(train_calib)):
        idx = calib_split_idx + i
        residuals, p_vals = causal_model.predict_and_residual(train_df, idx)
        scorer.calibrate(residuals, causal_model.residual_stds)
        score = scorer.score(p_vals, residuals, causal_model.residual_stds)
        calib_scores.append(score)
        
        # Log calibration step residuals and pvalues
        calib_entry = {
            'step_idx': i - 1,
            'score': float(score)
        }
        for var in var_names:
            calib_entry[f'res_{var}'] = float(residuals.get(var, 0.0))
            calib_entry[f'pval_{var}'] = float(p_vals.get(var, 1.0))
        calib_log.append(calib_entry)
        
    calibrator = ConformalCalibrator(target_fpr=args.target_fpr, lr=args.conformal_lr, alpha_init=args.conformal_alpha_init)
    calibrator.calibrate(calib_scores)
    
    # Setup test labels and scores
    labels = np.zeros(len(test_df))
    scores = np.zeros(len(test_df))
    conformal_pvals = np.zeros(len(test_df))
    alarms = np.zeros(len(test_df))
    thresholds_history = np.zeros(len(test_df))
    change_points = np.zeros(len(test_df))
    novelties_detected = np.zeros(len(test_df))
    
    # Inject Synthetic APT (Word -> PowerShell -> Reg)
    apt_edges = ["PROCESS:word.exe -> SPAWNS_PROCESS -> PROCESS:powershell.exe", 
                 "PROCESS:powershell.exe -> WRITES_FILE -> FILE:registry.dat"]
            
    np.random.seed(args.seed)
    # Inject attacks into 50 random test windows
    attack_indices = np.random.choice(len(test_df) - 2, 50, replace=False) + 1  # avoid index 0 due to lag
    for idx in attack_indices:
        # Realistic APT burst: Poisson-distributed activity spike on attack edges
        burst = max(np.random.poisson(5), 3)  # At least 3 events per attack stage
        test_df.iloc[idx, test_df.columns.get_loc(apt_edges[0])] += burst
        test_df.iloc[idx + 1, test_df.columns.get_loc(apt_edges[1])] += burst
        labels[idx] = 1 # Mark as attack
        labels[idx + 1] = 1 # Mark stage 2 as attack
        
    print(f"Evaluating {len(test_df)} test windows (50 contain injected APT attacks)...")
    
    # Train Baselines
    iforest_auc, lof_auc, ocsvm_auc, ae_auc = None, None, None, None
    print("\n====================================== ")
    print("🤖 Running Anomaly Detection Baselines...")
    try:
        from baselines import run_classical_baselines, train_and_evaluate_ae
        iforest_scores, lof_scores, ocsvm_scores = run_classical_baselines(train_df.values, test_df.values, contamination=args.target_fpr)
        
        iforest_auc = roc_auc_score(labels, iforest_scores)
        lof_auc = roc_auc_score(labels, lof_scores)
        ocsvm_auc = roc_auc_score(labels, ocsvm_scores)
        
        print("🧠 Running PyTorch Autoencoder...")
        ae_scores = train_and_evaluate_ae(train_df.values, test_df.values, epochs=50)
        ae_auc = roc_auc_score(labels, ae_scores)
        
        print(f"📊 Isolation Forest AUC: {iforest_auc:.4f}")
        print(f"📊 Local Outlier Factor AUC: {lof_auc:.4f}")
        print(f"📊 One-Class SVM AUC: {ocsvm_auc:.4f}")
        print(f"📊 PyTorch Autoencoder AUC: {ae_auc:.4f}")
    except Exception as e:
        print(f"Error running baselines: {e}")
    print("======================================\n")
    
    # Initialize Adaptive Window (Change-point) Detector
    detector = AdaptiveWindowDetector(
        num_features=len(var_names), 
        short_window=args.detector_short, 
        long_window=args.detector_long, 
        threshold=args.detector_threshold
    )
    
    # We also keep history to maintain lag
    test_history = pd.concat([train_df.iloc[-1:], test_df])
    
    # Pre-extract test history properties to avoid pandas index lookup overhead inside the loop
    history_cols = list(test_history.columns)
    history_arr = test_history.values
    col_to_idx = {col: idx for idx, col in enumerate(history_cols)}
    var_indices = [col_to_idx[v] for v in var_names if v in col_to_idx]
    
    start_eval_time = time.time()
    steps_log = []
    
    for i in range(args.tau_max, len(test_df) + 1):
        actual_row = history_arr[i]
        
        # Adaptive Windowing: update change-point detector with training feature subset
        feat_subset = actual_row[var_indices]
        change_detected = detector.update(feat_subset)
        if change_detected:
            change_points[i - 1] = 1.0
            print(f"   [Change-point detected at test step {i-1}] Updating local baseline and recalibrating.")
            
            # Refit causal regression model on the recent window of size detector.long_window
            refit_len = max(50, detector.long_window)
            start_idx = max(0, i - refit_len + 1)
            refit_df = test_history.iloc[start_idx : i + 1][var_names].copy()
            causal_model.fit(refit_df, std_floor=args.std_floor)
            
            # Recalibrate the conformal calibrator on this same window to update the scores queue
            new_calib_scores = []
            for k in range(1, len(refit_df)):
                hist_idx = start_idx + k
                res_k, pvals_k = causal_model.predict_and_residual(test_history, hist_idx)
                score_k = scorer.score(pvals_k, res_k, causal_model.residual_stds)
                new_calib_scores.append(score_k)
            
            calibrator.calibrate(new_calib_scores)
            
        # Calculate residuals for known baseline features
        known_residuals, known_pvals = causal_model.predict_and_residual(test_history, i)
        residuals = known_residuals.copy()
        p_vals = known_pvals.copy()
        
        # Handle structural novelties (edges active in test window but not in baseline features)
        active_indices = np.where(actual_row > 0)[0]
        novelty_detected = False
        active_novel_edges = []
        for idx in active_indices:
            edge = history_cols[idx]
            if edge not in var_names:
                novelty_detected = True
                active_novel_edges.append(edge)
                residuals[edge] = float(actual_row[idx])
                p_vals[edge] = 1e-15 # Causal p-value set to minimum to signal novelty
                
        if novelty_detected:
            novelties_detected[i - 1] = 1.0
            
        # Score anomaly using the hybrid scorer
        # For the novelty features, standard deviation is set to a small value
        res_stds = causal_model.residual_stds.copy()
        for edge in active_novel_edges:
            res_stds[edge] = 0.1 # Small standard dev for novel features
                    
        score = scorer.score(p_vals, residuals, res_stds)
        scores[i - 1] = score
        
        # Conformal p-value and adaptive thresholding
        conf_pval = calibrator.compute_conformal_pvalue(score)
        conformal_pvals[i - 1] = conf_pval
        
        # Raise alarm if conformal p-value is below adaptive threshold
        alarm = 1.0 if conf_pval < calibrator.alpha else 0.0
        alarms[i - 1] = alarm
        thresholds_history[i - 1] = calibrator.alpha
        
        # Update conformal threshold online using the target FPR budget (5%)
        calibrator.update_threshold(alarm)
        
        # Maintain rolling calibration window with current score if normal (no alarm)
        if alarm == 0.0:
            calibrator.update_calibration(score, max_size=245)
        
        # Log step details
        step_timestamp = test_df.index[i - 1]
        step_log_entry = {
            'step_idx': i - 1,
            'timestamp': str(step_timestamp),
            'label': int(labels[i - 1]),
            'score': float(score),
            'conformal_pval': float(conf_pval),
            'alarm': int(alarm),
            'threshold': float(thresholds_history[i - 1]),
            'change_detected': int(change_detected),
            'novelty_detected': int(novelty_detected)
        }
        
        # Log feature-wise residuals and p-values
        for var in var_names:
            step_log_entry[f'res_{var}'] = float(residuals.get(var, 0.0))
            step_log_entry[f'pval_{var}'] = float(p_vals.get(var, 1.0))
            
        for edge in apt_edges:
            if edge not in var_names:
                step_log_entry[f'res_{edge}'] = float(residuals.get(edge, 0.0))
                step_log_entry[f'pval_{edge}'] = float(p_vals.get(edge, 1.0))
                
        steps_log.append(step_log_entry)

    eval_elapsed = time.time() - start_eval_time
    print(f"Streaming evaluation completed in {eval_elapsed:.2f} seconds.")

    # Calculate Metrics
    auc_score = roc_auc_score(labels, scores)
    print(f"\n======================================")
    print(f"🏆 ZeroCausal AUC Score (Post-Enhancements): {auc_score:.4f}")
    print(f"   (Competitor Causal-IDS: 0.8400)")
    print(f"======================================\n")
    
    # Calculate FPR at specific thresholds
    fp_rate, tp_rate, thresholds = roc_curve(labels, scores)
    fpr_at_95_recall = 1.0
    for k, fpr in enumerate(fp_rate):
        if tp_rate[k] >= 0.95: # At 95% Recall
            fpr_at_95_recall = float(fpr)
            print(f"At 95% Detection Rate, False Positive Rate (FPR) is: {fpr:.4f} ({(fpr*100):.2f}%)")
            break
            
    # Empirical False Positive Rate of conformal prediction layer (excluding injected attacks)
    normal_indices = (labels == 0)
    empirical_fpr = np.mean(alarms[normal_indices])
    print(f"Target Conformal FPR: {args.target_fpr * 100:.2f}%")
    print(f"Empirical Alarm Rate on Normal Data: {(empirical_fpr * 100):.2f}%")
    print(f"Average Adaptive Conformal Threshold (alpha): {np.mean(thresholds_history):.4f}")
    
    # Write Logs
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Summary JSON log
    summary = {
        'dataset': 'OpTC',
        'num_samples_test': len(test_df),
        'num_attacks_injected': int(np.sum(labels)),
        'num_baseline_features': len(var_names),
        'pcmci_alpha': args.pcmci_alpha,
        'std_floor': args.std_floor,
        'a_p': args.a_p,
        'b_p': args.b_p,
        'a_r': args.a_r,
        'b_r': args.b_r,
        'target_fpr': args.target_fpr,
        'conformal_lr': args.conformal_lr,
        'conformal_alpha_init': args.conformal_alpha_init,
        'detector_short': args.detector_short,
        'detector_long': args.detector_long,
        'detector_threshold': args.detector_threshold,
        'eval_elapsed_seconds': eval_elapsed,
        'metrics': {
            'auc': auc_score,
            'fpr_at_95_recall': fpr_at_95_recall,
            'empirical_conformal_fpr': float(empirical_fpr),
            'average_conformal_threshold': float(np.mean(thresholds_history)),
            'iforest_auc': float(iforest_auc) if iforest_auc is not None else None,
            'lof_auc': float(lof_auc) if lof_auc is not None else None,
            'ocsvm_auc': float(ocsvm_auc) if ocsvm_auc is not None else None,
            'pytorch_ae_auc': float(ae_auc) if ae_auc is not None else None
        },
        'causal_graph_edges': [],
        'residual_stds': {var: float(std) for var, std in causal_model.residual_stds.items()}
    }
    
    p_matrix_arr = np.array(p_matrix) if not isinstance(p_matrix, np.ndarray) else p_matrix
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            for tau in range(1, p_matrix_arr.shape[2]):
                if p_matrix_arr[i, j, tau] < args.pcmci_alpha:
                    summary['causal_graph_edges'].append({
                        'source': var_names[i],
                        'target': var_names[j],
                        'tau': int(tau),
                        'p_value': float(p_matrix_arr[i, j, tau])
                    })
                    
    summary_path = os.path.join(args.log_dir, f"{args.run_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary log written to {summary_path}")
    
    # Steps CSV log
    steps_df = pd.DataFrame(steps_log)
    steps_path = os.path.join(args.log_dir, f"{args.run_name}_steps.csv")
    steps_df.to_csv(steps_path, index=False)
    print(f"Steps log written to {steps_path}")
    
    # Calib steps CSV log
    calib_df = pd.DataFrame(calib_log)
    calib_path = os.path.join(args.log_dir, f"{args.run_name}_calib_steps.csv")
    calib_df.to_csv(calib_path, index=False)
    print(f"Calibration steps log written to {calib_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroCausal Anomaly Detection Evaluation")
    parser.add_argument("--tau_max", type=int, default=2, help="Maximum time lag for causal discovery")
    parser.add_argument("--pcmci-alpha", type=float, default=0.01, help="Significance level for PCMCI baseline learning")
    parser.add_argument("--std-floor", type=float, default=1.0, help="Std floor for causal regression")
    parser.add_argument("--a-p", type=float, default=0.1, help="Beta parameter a_p for p-values under H1")
    parser.add_argument("--b-p", type=float, default=5.0, help="Beta parameter b_p for p-values under H1")
    parser.add_argument("--a-r", type=float, default=5.0, help="Beta parameter a_r for residuals under H1")
    parser.add_argument("--b-r", type=float, default=0.1, help="Beta parameter b_r for residuals under H1")
    parser.add_argument("--target-fpr", type=float, default=0.05, help="Target False Positive Rate for Conformal prediction")
    parser.add_argument("--conformal-lr", type=float, default=0.05, help="Learning rate for conformal threshold tuning")
    parser.add_argument("--conformal-alpha-init", type=float, default=0.05, help="Initial conformal alpha threshold")
    parser.add_argument("--detector-short", type=int, default=10, help="Short window for change-point detector")
    parser.add_argument("--detector-long", type=int, default=50, help="Long window for change-point detector")
    parser.add_argument("--detector-threshold", type=float, default=4.0, help="Threshold for change-point detector")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save execution logs")
    parser.add_argument("--run-name", type=str, default="optc_run", help="Unique name for this evaluation run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    parser.add_argument("--regressor", type=str, default="linear", choices=["linear", "rf"], help="Regression model for SCM")
    
    args = parser.parse_args()
    
    if not os.path.exists("optc_edges.csv"):
        print("Missing optc_edges.csv")
    else:
        ts_df = load_and_prep_data()
        if ts_df.shape[0] > 10:
            train_df = ts_df.iloc[:int(len(ts_df)*0.6)]
            p_matrix, var_names = learn_baseline(train_df, alpha=args.pcmci_alpha, tau_max=args.tau_max, seed=args.seed)
            evaluate(ts_df, p_matrix, var_names, args)
        else:
            print("Not enough data to evaluate.")
