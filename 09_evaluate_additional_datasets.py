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
import sys

# Import our core components
from zerocausal_core import (
    AdaptiveWindowDetector,
    CausalRegressionModel,
    HybridAnomalyScorer,
    ConformalCalibrator
)

warnings.filterwarnings("ignore")

# 1. High-Fidelity Data Generators with Controlled Noise Injection

def generate_tc3_data(num_windows=1000, noise_level=0.1, seed=42):
    """
    Generates a realistic stream of Windows 10 host provenance logs representing
    the DARPA TC3 TRACE performer baseline with controlled background noise.
    """
    print(f"Generating simulated DARPA TC3 (TRACE Windows 10) logs (noise_level={noise_level}, seed={seed})...")
    np.random.seed(seed)
    time_index = pd.date_range('2026-06-14 00:00:00', periods=num_windows, freq='1s')
    
    # 15 normal baseline features
    baseline_edges = [
        "PROCESS:explorer.exe -> SPAWNS_PROCESS -> PROCESS:chrome.exe",
        "PROCESS:chrome.exe -> CONNECTS_TO -> FLOW:142.20.61.130:443",
        "PROCESS:chrome.exe -> READS_FILE -> FILE:History",
        "PROCESS:svchost.exe -> CONNECTS_TO -> FLOW:8.8.8.8:53",
        "PROCESS:svchost.exe -> WRITES_FILE -> FILE:System.evtx",
        "PROCESS:explorer.exe -> SPAWNS_PROCESS -> PROCESS:git.exe",
        "PROCESS:git.exe -> READS_FILE -> FILE:config",
        "PROCESS:git.exe -> CONNECTS_TO -> FLOW:140.82.121.3:443",
        "PROCESS:lsass.exe -> READS_FILE -> FILE:Security.log",
        "PROCESS:spoolsv.exe -> READS_FILE -> FILE:printers.db",
        "PROCESS:taskhostw.exe -> WRITES_FILE -> FILE:setupapi.log",
        "PROCESS:searchindexer.exe -> READS_FILE -> FILE:index.db",
        "PROCESS:OneDrive.exe -> WRITES_FILE -> FILE:SyncEngine.db",
        "PROCESS:GoogleUpdate.exe -> CONNECTS_TO -> FLOW:172.217.164.110:443",
        "PROCESS:unknown_process -> READS_FILE -> FILE:gpt.ini"
    ]
    
    data = {}
    # Simulate Poisson distributions for normal activity
    for edge in baseline_edges:
        if "svchost" in edge or "chrome" in edge:
            data[edge] = np.random.poisson(lam=3.0, size=num_windows)
        else:
            data[edge] = np.random.poisson(lam=0.5, size=num_windows)
            
    # Inject synthetic baseline dependencies (causality)
    data["PROCESS:spoolsv.exe -> READS_FILE -> FILE:printers.db"] += (
        data["PROCESS:svchost.exe -> CONNECTS_TO -> FLOW:8.8.8.8:53"] > 2
    ).astype(int)
    
    # Inject distractor background noise columns if noise_level > 0
    if noise_level > 0:
        distractor_edges = [
            "PROCESS:explorer.exe -> READS_FILE -> FILE:desktop.ini",
            "PROCESS:chrome.exe -> CONNECTS_TO -> FLOW:8.8.4.4:53",
            "PROCESS:svchost.exe -> WRITE -> REGISTRY:HKLM",
            "PROCESS:taskmgr.exe -> OPEN -> PROCESS:explorer.exe",
            "PROCESS:cmd.exe -> SPAWNS_PROCESS -> PROCESS:conhost.exe"
        ]
        for edge in distractor_edges:
            data[edge] = np.random.poisson(lam=noise_level * 3.0, size=num_windows)
            
    attack_edges = [
        "PROCESS:nginx.exe -> SPAWNS_PROCESS -> PROCESS:bash.exe",
        "PROCESS:bash.exe -> WRITES_FILE -> FILE:malicious.elf",
        "PROCESS:malicious.elf -> MODIFY -> FILE:passwd"
    ]
    
    # Inject background noise for attack edges so they aren't trivial novelties
    for edge in attack_edges:
        data[edge] = np.random.poisson(lam=0.01, size=num_windows)
        
    df = pd.DataFrame(data, index=time_index)
    
    # Add random Gaussian noise to count distributions to simulate log fluctuations
    if noise_level > 0:
        for col in df.columns:
            noise = np.random.normal(0, noise_level * 2.0, size=num_windows)
            df[col] = np.clip(df[col] + noise, 0, None)
            
    labels = np.zeros(num_windows)
    
    # 50 random test windows in the latter 40%
    test_start = int(num_windows * 0.6)
    attack_indices = np.random.choice(np.arange(test_start + 10, num_windows - 10), 50, replace=False)
    
    for idx in attack_indices:
        labels[idx:idx+3] = 1 # Mark step and next two as attack windows
        
    return df, labels, attack_edges, attack_indices

def generate_nodlink_data(num_windows=1000, noise_level=0.1, seed=42):
    """
    Generates a realistic stream of provenance logs representing the
    NODLINK multi-hop shell-to-reconnaissance scenario with controlled noise.
    """
    print(f"Generating simulated NODLINK (Multi-Hop APT) logs (noise_level={noise_level}, seed={seed})...")
    np.random.seed(seed)
    time_index = pd.date_range('2026-06-14 00:00:00', periods=num_windows, freq='1s')
    
    # 15 normal baseline features
    baseline_edges = [
        "PROCESS:explorer.exe -> SPAWNS_PROCESS -> PROCESS:outlook.exe",
        "PROCESS:outlook.exe -> READS_FILE -> FILE:inbox.db",
        "PROCESS:outlook.exe -> CONNECTS_TO -> FLOW:10.0.0.2:993",
        "PROCESS:winword.exe -> READS_FILE -> FILE:document.docx",
        "PROCESS:explorer.exe -> SPAWNS_PROCESS -> PROCESS:cmd.exe",
        "PROCESS:cmd.exe -> READS_FILE -> FILE:LocalSettings",
        "PROCESS:dns.exe -> CONNECTS_TO -> FLOW:10.0.0.1:53",
        "PROCESS:svchost.exe -> WRITES_FILE -> FILE:log.txt",
        "PROCESS:svchost.exe -> CONNECTS_TO -> FLOW:10.0.0.10:445",
        "PROCESS:lsass.exe -> READS_FILE -> FILE:sam.db",
        "PROCESS:taskhostw.exe -> WRITES_FILE -> FILE:setupapi.log",
        "PROCESS:searchindexer.exe -> READS_FILE -> FILE:index.db",
        "PROCESS:OneDrive.exe -> WRITES_FILE -> FILE:SyncEngine.db",
        "PROCESS:excel.exe -> READS_FILE -> FILE:finance.xlsx",
        "PROCESS:system -> WRITES_FILE -> FILE:$Mft"
    ]
    
    data = {}
    for edge in baseline_edges:
        if "svchost" in edge or "outlook" in edge:
            data[edge] = np.random.poisson(lam=2.5, size=num_windows)
        else:
            data[edge] = np.random.poisson(lam=0.4, size=num_windows)
            
    # Inject baseline dependencies
    data["PROCESS:cmd.exe -> READS_FILE -> FILE:LocalSettings"] += (
        data["PROCESS:explorer.exe -> SPAWNS_PROCESS -> PROCESS:cmd.exe"] > 0
    ).astype(int)
    
    # Inject distractor background noise columns if noise_level > 0
    if noise_level > 0:
        distractor_edges = [
            "PROCESS:chrome.exe -> READS_FILE -> FILE:History",
            "PROCESS:dns.exe -> CONNECTS_TO -> FLOW:8.8.8.8:53",
            "PROCESS:svchost.exe -> MODIFY -> FILE:gpt.ini",
            "PROCESS:wscript.exe -> SPAWNS_PROCESS -> PROCESS:conhost.exe",
            "PROCESS:winlogon.exe -> READS_FILE -> FILE:userenv.log"
        ]
        for edge in distractor_edges:
            data[edge] = np.random.poisson(lam=noise_level * 2.5, size=num_windows)
            
    attack_edges = [
        "PROCESS:outlook.exe -> SPAWNS_PROCESS -> PROCESS:cmd.exe",
        "PROCESS:cmd.exe -> WRITES_FILE -> FILE:recon_results.txt",
        "PROCESS:cmd.exe -> CONNECTS_TO -> FLOW:10.0.0.15:445"
    ]
    
    # Inject background noise for attack edges so they aren't trivial novelties
    for edge in attack_edges:
        data[edge] = np.random.poisson(lam=0.01, size=num_windows)
        
    df = pd.DataFrame(data, index=time_index)
    
    # Add random Gaussian noise to count distributions
    if noise_level > 0:
        for col in df.columns:
            noise = np.random.normal(0, noise_level * 1.5, size=num_windows)
            df[col] = np.clip(df[col] + noise, 0, None)
            
    labels = np.zeros(num_windows)
    
    # 50 random test windows in the latter 40%
    test_start = int(num_windows * 0.6)
    attack_indices = np.random.choice(np.arange(test_start + 10, num_windows - 10), 50, replace=False)
    
    for idx in attack_indices:
        labels[idx:idx+3] = 1
        
    return df, labels, attack_edges, attack_indices

# 2. Baseline Learning

def learn_baseline(train_df, alpha=0.01, tau_max=2, dataset="tc3", seed=42):
    cache_path = f"logs/pcmci_cache_{dataset}_{seed}_alpha{alpha}_tau{tau_max}_features{train_df.shape[1]}.pkl"
    if os.path.exists(cache_path):
        print(f"  ✅ Loading cached PCMCI baseline from {cache_path}")
        import pickle
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            return cache_data['p_matrix'], cache_data['var_names']
            
    n_vars = train_df.shape[1]
    n_windows = train_df.shape[0]
    print(f"\n{'='*60}")
    print(f"  🧠 PCMCI Causal Discovery")
    print(f"     Variables: {n_vars}, Windows: {n_windows}, tau_max: {tau_max}, alpha: {alpha}")
    print(f"     Estimated complexity: O({n_vars}² × {n_windows}) = ~{n_vars**2 * n_windows:,.0f} operations")
    print(f"{'='*60}")
    
    var_names = train_df.columns.tolist()
    data = train_df.astype(float).values
    dataframe = pp.DataFrame(data, datatime=np.arange(len(train_df)), var_names=var_names)
    
    cond_ind_test = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
    
    pcmci_start = time.time()
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)
    pcmci_elapsed = time.time() - pcmci_start
    
    n_edges = np.sum(results['p_matrix'] < alpha) - n_vars  # exclude self-links at tau=0
    print(f"  ✅ PCMCI complete in {pcmci_elapsed:.1f}s — discovered {n_edges} causal edges")
    
    # Save to cache
    os.makedirs("logs", exist_ok=True)
    import pickle
    with open(cache_path, 'wb') as f:
        pickle.dump({'p_matrix': results['p_matrix'], 'var_names': var_names}, f)
        
    return results['p_matrix'], var_names

# 3. Streaming Evaluation Loop

def evaluate(ts_df, labels, attack_edges, attack_indices, p_matrix, var_names, args, split_idx=None):
    if split_idx is None:
        split_idx = int(len(ts_df) * 0.6)
    train_df = ts_df.iloc[:split_idx].copy()
    test_df = ts_df.iloc[split_idx:].copy()
    
    # Inject Synthetic APT only into test_df (leaving train_df clean)
    for edge in attack_edges:
        if edge not in test_df.columns:
            test_df[edge] = 0
            
    for idx in attack_indices:
        test_idx = idx - split_idx
        if 0 <= test_idx < len(test_df) - 2 and len(attack_edges) > 2:
            # Realistic APT burst: Poisson-distributed activity spike on attack edges
            burst = max(np.random.poisson(5), 3)  # At least 3 events per attack stage
            test_df.iloc[test_idx, test_df.columns.get_loc(attack_edges[0])] += burst
            test_df.iloc[test_idx + 1, test_df.columns.get_loc(attack_edges[1])] += burst
            test_df.iloc[test_idx + 2, test_df.columns.get_loc(attack_edges[2])] += burst
            
    # Sub-split training data for conformal calibration
    calib_split_idx = int(len(train_df) * 0.7)
    train_proper = train_df.iloc[:calib_split_idx].copy()
    train_calib = train_df.iloc[calib_split_idx:].copy()
    
    # Fit causal regression models
    print(f"\n{'='*60}")
    print(f"  📐 Fitting Causal Regression (regressor={args.regressor})")
    print(f"     Training on {len(train_proper)} windows, {len(var_names)} features")
    print(f"{'='*60}")
    causal_model = CausalRegressionModel(p_matrix, var_names, tau_max=args.tau_max, alpha=args.pcmci_alpha, regressor_type=args.regressor)
    causal_model.fit(train_proper, std_floor=args.std_floor)
    print(f"  ✅ Causal regression fitted")
    
    # Score calibration set
    calib_total = len(train_calib) - args.tau_max
    print(f"\n{'='*60}")
    print(f"  📊 Calibrating Conformal Prediction ({calib_total} calibration windows)")
    print(f"{'='*60}")
    scorer = HybridAnomalyScorer(d=len(var_names), w=0.5, floor=args.std_floor)
    calib_scores = []
    calib_log = []
    calib_start = time.time()
    
    for i in range(args.tau_max, len(train_calib)):
        idx = calib_split_idx + i
        residuals, p_vals = causal_model.predict_and_residual(train_df, idx)
        scorer.calibrate(residuals, causal_model.residual_stds)
        score = scorer.score(p_vals, residuals, causal_model.residual_stds)
        calib_scores.append(score)
        
        calib_entry = {
            'step_idx': i - 1,
            'score': float(score)
        }
        for var in var_names:
            calib_entry[f'res_{var}'] = float(residuals.get(var, 0.0))
            calib_entry[f'pval_{var}'] = float(p_vals.get(var, 1.0))
        calib_log.append(calib_entry)
        
        done = len(calib_scores)
        if done % max(1, calib_total // 10) == 0 or done == calib_total:
            elapsed = time.time() - calib_start
            eta = (elapsed / done) * (calib_total - done) if done > 0 else 0
            print(f"  Calibration: {done}/{calib_total} ({done/calib_total*100:.0f}%) — {elapsed:.1f}s elapsed, ETA {eta:.0f}s")
        
    calibrator = ConformalCalibrator(target_fpr=args.target_fpr, lr=args.conformal_lr, alpha_init=args.conformal_alpha_init)
    calibrator.calibrate(calib_scores)
    print(f"  ✅ Calibration complete ({len(calib_scores)} scores, threshold={calibrator.alpha:.4f})")
    
    # Setup test logs and history
    test_labels = labels[split_idx:]
    scores = np.zeros(len(test_df))
    conformal_pvals = np.zeros(len(test_df))
    alarms = np.zeros(len(test_df))
    thresholds_history = np.zeros(len(test_df))
    change_points = np.zeros(len(test_df))
    novelties_detected = np.zeros(len(test_df))
    
    detector = AdaptiveWindowDetector(
        num_features=len(var_names), 
        short_window=args.detector_short, 
        long_window=args.detector_long, 
        threshold=args.detector_threshold
    )
    
    test_history = pd.concat([train_df.iloc[-1:], test_df])
    history_cols = list(test_history.columns)
    history_arr = test_history.to_numpy().copy()
    col_to_idx = {col: idx for idx, col in enumerate(history_cols)}
    var_indices = [col_to_idx[v] for v in var_names if v in col_to_idx]
    
    start_eval_time = time.time()
    steps_log = []
    total_steps = len(test_df) - args.tau_max + 1
    attacks_found = 0
    
    print(f"\n{'='*60}")
    print(f"  🔍 Streaming Evaluation: {total_steps} steps")
    print(f"     Test windows: {len(test_df)}, Attack labels in test: {int(np.sum(test_labels))}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    for i in range(args.tau_max, len(test_df) + 1):
        # If simulating drift, double the rate of several normal edges starting from test step 200
        if args.simulate_drift and i >= 200:
            for idx in var_indices[:5]:
                history_arr[i, idx] = history_arr[i, idx] * 2.0
                
        actual_row = history_arr[i]
        
        # Adaptive Windowing
        feat_subset = actual_row[var_indices]
        change_detected = detector.update(feat_subset)
        if change_detected:
            change_points[i - 1] = 1.0
            print(f"   [Change-point detected at step {i-1}] Refitting local causal baseline and recalibrating.")
            
            # Refit causal regression model on the recent window of size detector.long_window
            refit_len = max(50, detector.long_window)
            start_idx = max(0, i - refit_len + 1)
            refit_df = pd.DataFrame(history_arr[start_idx : i + 1], columns=history_cols)[var_names].copy()
            causal_model.fit(refit_df, std_floor=args.std_floor)
            
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
        novelty_detected = False
        active_novel_edges = []
        for idx in active_indices:
            edge = history_cols[idx]
            if edge not in var_names:
                novelty_detected = True
                active_novel_edges.append(edge)
                residuals[edge] = float(actual_row[idx])
                p_vals[edge] = 1e-15
                
        if novelty_detected:
            novelties_detected[i - 1] = 1.0
            
        # Score anomaly
        res_stds = causal_model.residual_stds.copy()
        for edge in active_novel_edges:
            res_stds[edge] = 0.1
            
        score = scorer.score(p_vals, residuals, res_stds)
        scores[i - 1] = score
        
        # Conformal prediction
        conf_pval = calibrator.compute_conformal_pvalue(score)
        conformal_pvals[i - 1] = conf_pval
        
        alarm = 1.0 if conf_pval < calibrator.alpha else 0.0
        alarms[i - 1] = alarm
        thresholds_history[i - 1] = calibrator.alpha
        
        # Update threshold
        calibrator.update_threshold(alarm)
        
        # Maintain rolling calibration window with current score if normal (no alarm)
        if alarm == 0.0:
            calibrator.update_calibration(score, max_size=245)
        
        # Log entry
        step_timestamp = test_df.index[i - 1]
        step_log_entry = {
            'step_idx': i - 1,
            'timestamp': str(step_timestamp),
            'label': int(test_labels[i - 1]),
            'score': float(score),
            'conformal_pval': float(conf_pval),
            'alarm': int(alarm),
            'threshold': float(thresholds_history[i - 1]),
            'change_detected': int(change_detected),
            'novelty_detected': int(novelty_detected)
        }
        
        for var in var_names:
            step_log_entry[f'res_{var}'] = float(residuals.get(var, 0.0))
            step_log_entry[f'pval_{var}'] = float(p_vals.get(var, 1.0))
            
        for edge in attack_edges:
            if edge not in var_names:
                step_log_entry[f'res_{edge}'] = float(residuals.get(edge, 0.0))
                step_log_entry[f'pval_{edge}'] = float(p_vals.get(edge, 1.0))
                
        steps_log.append(step_log_entry)
        
        # Track attacks found
        if test_labels[i - 1] == 1 and alarm == 1.0:
            attacks_found += 1
            
        # Progress + periodic checkpoint save
        step_num = i - args.tau_max + 1
        if step_num % max(1, total_steps // 20) == 0 or i % args.checkpoint_interval == 0 or step_num == total_steps:
            elapsed = time.time() - start_eval_time
            rate = step_num / elapsed if elapsed > 0 else 0
            eta = (total_steps - step_num) / rate if rate > 0 else 0
            pct = step_num / total_steps * 100
            bar_len = 30
            filled = int(bar_len * step_num / total_steps)
            bar = '█' * filled + '░' * (bar_len - filled)
            alarms_so_far = int(np.sum(alarms[:i]))
            print(f"  [{bar}] {pct:5.1f}% | Step {step_num}/{total_steps} | "
                  f"Alarms: {alarms_so_far} | Attacks detected: {attacks_found} | "
                  f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s")
            sys.stdout.flush()
            
        if i % args.checkpoint_interval == 0:
            os.makedirs(args.log_dir, exist_ok=True)
            run_name = args.run_name if args.run_name else f"{args.dataset}_default"
            steps_path = os.path.join(args.log_dir, f"{run_name}_steps.csv")
            pd.DataFrame(steps_log).to_csv(steps_path, index=False)
        
    eval_elapsed = time.time() - start_eval_time
    print(f"\n  ✅ Streaming evaluation completed in {eval_elapsed:.1f}s ({total_steps/eval_elapsed:.0f} steps/sec)")
    
    # Compute final metrics
    auc_score = roc_auc_score(test_labels, scores)
    print(f"\n{'='*60}")
    print(f"  🏆 RESULTS: {args.dataset.upper()}")
    print(f"     AUC Score: {auc_score:.4f}")
    print(f"{'='*60}")
    
    fp_rate, tp_rate, _ = roc_curve(test_labels, scores)
    fpr_at_95_recall = 1.0
    for k, fpr in enumerate(fp_rate):
        if tp_rate[k] >= 0.95:
            fpr_at_95_recall = float(fpr)
            print(f"At 95% Recall, False Positive Rate (FPR) is: {fpr:.4f} ({(fpr*100):.2f}%)")
            break
            
    normal_indices = (test_labels == 0)
    empirical_fpr = np.mean(alarms[normal_indices])
    print(f"Target Conformal FPR budget: {args.target_fpr * 100:.2f}%")
    print(f"Empirical Alarm Rate on Normal Data: {(empirical_fpr * 100):.2f}%")
    print(f"Average Adaptive Conformal Threshold: {np.mean(thresholds_history):.4f}")
    
    # Run Baselines
    iforest_auc, lof_auc, ocsvm_auc, ae_auc = None, None, None, None
    if args.baseline:
        print("\n====================================== ")
        print("🤖 Running Anomaly Detection Baselines...")
        train_full = train_proper.copy()
        for col in attack_edges:
            if col not in train_full.columns:
                train_full[col] = 0.0
                
        # Align column orders
        cols = list(test_df.columns)
        train_full = train_full[cols]
        
        try:
            from baselines import run_classical_baselines, train_and_evaluate_ae
            iforest_scores, lof_scores, ocsvm_scores = run_classical_baselines(train_full.values, test_df.values, contamination=args.target_fpr)
            
            iforest_auc = roc_auc_score(test_labels, iforest_scores)
            lof_auc = roc_auc_score(test_labels, lof_scores)
            ocsvm_auc = roc_auc_score(test_labels, ocsvm_scores)
            
            print("🧠 Running PyTorch Autoencoder...")
            ae_scores = train_and_evaluate_ae(train_full.values, test_df.values, epochs=50)
            ae_auc = roc_auc_score(test_labels, ae_scores)
            
            print(f"📊 Isolation Forest AUC: {iforest_auc:.4f}")
            print(f"📊 Local Outlier Factor AUC: {lof_auc:.4f}")
            print(f"📊 One-Class SVM AUC: {ocsvm_auc:.4f}")
            print(f"📊 PyTorch Autoencoder AUC: {ae_auc:.4f}")
        except Exception as e:
            print(f"Error running baselines: {e}")
        print("======================================\n")
        
    # Save logs
    os.makedirs(args.log_dir, exist_ok=True)
    run_name = args.run_name if args.run_name else f"{args.dataset}_default"
    
    summary = {
        'dataset': args.dataset.upper(),
        'num_samples_test': len(test_df),
        'num_attacks_injected': int(np.sum(test_labels)),
        'num_baseline_features': len(var_names),
        'noise_level': args.noise_level,
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
            'isolation_forest_auc': iforest_auc,
            'lof_auc': lof_auc,
            'ocsvm_auc': ocsvm_auc,
            'pytorch_ae_auc': ae_auc
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
                    
    summary_path = os.path.join(args.log_dir, f"{run_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary log written to {summary_path}")
    
    steps_df = pd.DataFrame(steps_log)
    steps_path = os.path.join(args.log_dir, f"{run_name}_steps.csv")
    steps_df.to_csv(steps_path, index=False)
    print(f"Steps log written to {steps_path}")
    
    calib_df = pd.DataFrame(calib_log)
    calib_path = os.path.join(args.log_dir, f"{run_name}_calib_steps.csv")
    calib_df.to_csv(calib_path, index=False)
    print(f"Calibration steps log written to {calib_path}")

def main():
    parser = argparse.ArgumentParser(description="ZeroCausal Full-Dataset Evaluation")
    parser.add_argument("--tau_max", type=int, default=2, help="Maximum time lag for causal discovery")
    parser.add_argument("--dataset", type=str, required=True, choices=["tc3", "nodlink", "streamspot", "beth"], help="Dataset to evaluate")
    parser.add_argument("--noise-level", type=float, default=0.1, help="Controlled noise scaling factor (0.0 to 1.0)")
    parser.add_argument("--baseline", action="store_true", help="Run classical ML baselines (IF, LOF, OCSVM, AE)")
    parser.add_argument("--simulate-drift", action="store_true", help="Simulate a benign concept drift at step 500")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--pcmci-alpha", type=float, default=0.01, help="Significance level for PCMCI")
    parser.add_argument("--std-floor", type=float, default=1.0, help="Std floor for causal regression")
    parser.add_argument("--a-p", type=float, default=0.1, help="Beta parameter a_p")
    parser.add_argument("--b-p", type=float, default=5.0, help="Beta parameter b_p")
    parser.add_argument("--a-r", type=float, default=5.0, help="Beta parameter a_r")
    parser.add_argument("--b-r", type=float, default=0.1, help="Beta parameter b_r")
    parser.add_argument("--target-fpr", type=float, default=0.05, help="Target FPR for conformal prediction")
    parser.add_argument("--conformal-lr", type=float, default=0.05, help="Learning rate for conformal threshold")
    parser.add_argument("--conformal-alpha-init", type=float, default=0.05, help="Initial conformal alpha")
    parser.add_argument("--detector-short", type=int, default=10, help="Short window for change-point detector")
    parser.add_argument("--detector-long", type=int, default=50, help="Long window for change-point detector")
    parser.add_argument("--detector-threshold", type=float, default=4.0, help="Change-point detector threshold")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--run-name", type=str, default="", help="Unique run name")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit PCMCI training windows")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Checkpoint every N steps")
    parser.add_argument("--regressor", type=str, default="linear", choices=["linear", "rf"], help="SCM regressor")
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"  ZeroCausal Full-Dataset Evaluation")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")
    
    split_idx = None  # will be determined per-dataset
    
    if args.dataset == "tc3":
        ts_df, labels, attack_edges, attack_indices = generate_tc3_data(noise_level=args.noise_level, seed=args.seed)
    elif args.dataset == "nodlink":
        ts_df, labels, attack_edges, attack_indices = generate_nodlink_data(noise_level=args.noise_level, seed=args.seed)
    elif args.dataset == "streamspot":
        print("[StreamSpot] Loading FULL preprocessed dataset...")
        df = pd.read_csv("data/processed/streamspot_edges.csv", index_col='timestamp')
        labels = df['is_attack'].values
        ts_df = df.drop(columns=['is_attack'])
        attack_edges = []
        attack_indices = np.where(labels == 1)[0]
        
        # Graph-aware split: train on benign-only, test on attack + held-out benign
        # Attack windows are at indices 26356-29199 (graph_ids 300-399)
        benign_mask = labels == 0
        attack_mask = labels == 1
        benign_indices = np.where(benign_mask)[0]
        attack_indices_arr = np.where(attack_mask)[0]
        
        # Use first 60% of benign windows for training (all clean)
        n_benign_train = int(0.6 * len(benign_indices))
        train_benign_idx = benign_indices[:n_benign_train]
        test_benign_idx = benign_indices[n_benign_train:]
        
        # Test set = remaining 40% benign + ALL attack windows
        test_idx = np.sort(np.concatenate([test_benign_idx, attack_indices_arr]))
        
        # Rebuild ts_df and labels: training block first, then test block
        train_df_ss = ts_df.iloc[train_benign_idx]
        test_df_ss = ts_df.iloc[test_idx]
        ts_df = pd.concat([train_df_ss, test_df_ss])
        
        train_labels = labels[train_benign_idx]
        test_labels_ss = labels[test_idx]
        labels = np.concatenate([train_labels, test_labels_ss])
        
        split_idx = len(train_df_ss)
        attack_indices = np.where(labels == 1)[0]
        
        print(f"  Total windows: {len(ts_df)}")
        print(f"  Graph-aware split:")
        print(f"    Training (benign only): {split_idx} windows (0 attacks)")
        print(f"    Testing (benign + attack): {len(test_df_ss)} windows ({int(test_labels_ss.sum())} attacks)")
        print(f"  Attack indices in combined: {attack_indices.min()}-{attack_indices.max()}")
        
        # Inject noise if requested
        if args.noise_level > 0:
            for col in ts_df.columns:
                noise = np.random.normal(0, args.noise_level * 2.0, size=len(ts_df))
                ts_df[col] = np.clip(ts_df[col] + noise, 0, None)
                
    elif args.dataset == "beth":
        print("[BETH] Loading preprocessed per-host data...")
        df = pd.read_csv("data/processed/beth_edges.csv", index_col='window')
        labels = df['is_attack'].values
        ts_df = df.drop(columns=['is_attack'])
        attack_edges = []
        attack_indices = np.where(labels == 1)[0]
        
        # Verify attacks exist in test split
        test_split = int(len(ts_df) * 0.6)
        attacks_in_test = np.sum(labels[test_split:] == 1)
        print(f"  Total windows: {len(ts_df)}")
        print(f"  Attack windows: {int(labels.sum())} ({labels.sum()/len(labels)*100:.1f}%)")
        print(f"  60/40 split: train={test_split}, test={len(ts_df)-test_split} ({int(attacks_in_test)} attacks in test)")
        
        if attacks_in_test == 0:
            print("  ⚠️  No attacks in test split! Using stratified shuffle...")
            # Stratified shuffle: ensure proportional attacks in both splits
            np.random.seed(args.seed)
            normal_idx = np.where(labels == 0)[0]
            attack_idx = np.where(labels == 1)[0]
            np.random.shuffle(normal_idx)
            np.random.shuffle(attack_idx)
            n_train_normal = int(0.6 * len(normal_idx))
            n_train_attack = int(0.6 * len(attack_idx))
            train_idx = np.sort(np.concatenate([normal_idx[:n_train_normal], attack_idx[:n_train_attack]]))
            test_idx_b = np.sort(np.concatenate([normal_idx[n_train_normal:], attack_idx[n_train_attack:]]))
            
            ts_df = pd.concat([ts_df.iloc[train_idx], ts_df.iloc[test_idx_b]])
            labels = np.concatenate([labels[train_idx], labels[test_idx_b]])
            split_idx = len(train_idx)
            attack_indices = np.where(labels == 1)[0]
            attacks_in_test = np.sum(labels[split_idx:] == 1)
            print(f"  ✅ After stratified split: {int(attacks_in_test)} attacks in test")
        
        if args.noise_level > 0:
            for col in ts_df.columns:
                noise = np.random.normal(0, args.noise_level * 2.0, size=len(ts_df))
                ts_df[col] = np.clip(ts_df[col] + noise, 0, None)
    
    # Determine training data for PCMCI
    if split_idx is None:
        split_idx = int(len(ts_df) * 0.6)
    train_df = ts_df.iloc[:split_idx]
    
    # Verify clean training for real-label datasets
    if args.dataset in ('streamspot', 'beth'):
        train_attacks = int(np.sum(labels[:split_idx]))
        print(f"\n  🔒 Training set attack verification: {train_attacks} attacks in training")
        if train_attacks > 0:
            print(f"  ⚠️  WARNING: {train_attacks} attack windows in training set!")
    
    pcmci_train_df = train_df
    if args.train_limit and len(train_df) > args.train_limit:
        print(f"\n  ⚡ Limiting PCMCI training to {args.train_limit} / {len(train_df)} windows")
        pcmci_train_df = train_df.iloc[-args.train_limit:]
        
    p_matrix, var_names = learn_baseline(pcmci_train_df, alpha=args.pcmci_alpha, tau_max=args.tau_max, dataset=args.dataset, seed=args.seed)
    evaluate(ts_df, labels, attack_edges, attack_indices, p_matrix, var_names, args, split_idx=split_idx)
    
    print(f"\n{'#'*60}")
    print(f"  Evaluation complete: {args.dataset.upper()}")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()
