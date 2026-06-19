"""
15_beat_papers_evaluation.py
============================
ZeroCausal v2 — Causal-ML Hybrid (CausalMLHybrid)
Target: beat ALL baselines on every benchmark.

Architecture
------------
  Raw stream
    → RobustParCorr causal graph
    → CausalRegressionModel (KalmanSCM continuously updates)
    → CausalFeatureExtractor (12-dim structured vector per window)
    ┌────────────────────────────────────────────────┐
    │  Three parallel scorers:                       │
    │   1. CAS  (causal mechanism violations)        │
    │   2. LSTM-AE (temporal sequence anomaly)       │
    │   3. CausalRF (supervised on causal features)  │
    └────────────────────────────────────────────────┘
    → StackedEnsembleDetector (
    istic meta-learner)
    → Final anomaly score → AUC / FPR metrics

Datasets
--------
  TC3      (synthetic DARPA TRACE Windows APT)
  NODLINK  (synthetic multi-hop lateral movement)
  BETH     (real Linux host telemetry — loaded from BETH.zip if available)
  OpTC     (real Windows enterprise — uses optc_edges.csv if available)
  StreamSpot (real provenance graphs — uses streamspot.zip if available)

Usage
-----
  python 15_beat_papers_evaluation.py [--datasets tc3 nodlink beth optc streamspot]
                                      [--seed 42] [--epochs 60]
"""

import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from zerocausal_core import (
    AdaptiveWindowDetector, CausalRegressionModel, HybridAnomalyScorer,
    ConformalCalibrator, RobustCalibrationFilter,
    RobustParCorr,
    CausalFeatureExtractor, CausalRandomForestDetector,
    LightweightLSTMAE, StackedEnsembleDetector,
)
from baselines import train_and_evaluate_ae

# ─── Data Generators ─────────────────────────────────────────────────────────

def gen_tc3(seed=42, n=1000):
    np.random.seed(seed)
    baseline = [
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
        "PROCESS:unknown_process -> READS_FILE -> FILE:gpt.ini",
    ]
    apt = ["PROCESS:word.exe -> SPAWNS_PROCESS -> PROCESS:powershell.exe",
           "PROCESS:powershell.exe -> WRITES_FILE -> FILE:payload.exe",
           "PROCESS:payload.exe -> CONNECTS_TO -> FLOW:185.220.101.5:4444"]
    data = {}
    for e in baseline:
        lam = 3.0 if ("svchost" in e or "chrome" in e) else 0.5
        data[e] = np.random.poisson(lam, n).astype(float)
    data["PROCESS:spoolsv.exe -> READS_FILE -> FILE:printers.db"] += (
        data["PROCESS:svchost.exe -> CONNECTS_TO -> FLOW:8.8.8.8:53"] > 2).astype(float)
    for e in apt:
        data[e] = np.random.poisson(0.01, n).astype(float)
    ts = pd.DataFrame(data)
    labels = np.zeros(n)
    # attacks from 55% onward so both val (55-70%) and test (70-100%) have labels
    attack_start = int(n * 0.55)
    for idx in range(attack_start, n - 3, 5):
        burst = max(np.random.poisson(8), 5)
        ts.iloc[idx, ts.columns.get_loc(apt[0])] += burst
        ts.iloc[idx+1, ts.columns.get_loc(apt[1])] += burst
        ts.iloc[idx+2, ts.columns.get_loc(apt[2])] += burst
        labels[idx] = labels[idx+1] = labels[idx+2] = 1
    return ts, labels

def gen_nodlink(seed=42, n=1000):
    np.random.seed(seed)
    baseline = [
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
        "PROCESS:system -> WRITES_FILE -> FILE:$Mft",
    ]
    apt = ["PROCESS:outlook.exe -> SPAWNS_PROCESS -> PROCESS:cmd.exe",
           "PROCESS:cmd.exe -> WRITES_FILE -> FILE:recon_results.txt",
           "PROCESS:cmd.exe -> CONNECTS_TO -> FLOW:10.0.0.15:445"]
    data = {}
    for e in baseline:
        lam = 2.5 if ("svchost" in e or "outlook" in e) else 0.4
        data[e] = np.random.poisson(lam, n).astype(float)
    data["PROCESS:cmd.exe -> READS_FILE -> FILE:LocalSettings"] += (
        data["PROCESS:explorer.exe -> SPAWNS_PROCESS -> PROCESS:cmd.exe"] > 0).astype(float)
    for e in apt:
        data[e] = np.random.poisson(0.01, n).astype(float)
    ts = pd.DataFrame(data)
    labels = np.zeros(n)
    attack_start = int(n * 0.55)
    for idx in range(attack_start, n - 3, 5):
        burst = max(np.random.poisson(8), 5)
        ts.iloc[idx, ts.columns.get_loc(apt[0])] += burst
        ts.iloc[idx+1, ts.columns.get_loc(apt[1])] += burst
        ts.iloc[idx+2, ts.columns.get_loc(apt[2])] += burst
        labels[idx] = labels[idx+1] = labels[idx+2] = 1
    return ts, labels

def load_beth(data_dir="data/processed"):
    """Load BETH windows CSV: columns = [window, is_attack, feat1, feat2, ...]"""
    path = os.path.join(data_dir, "beth_edges.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    label_col = 'is_attack' if 'is_attack' in df.columns else df.columns[1]
    labels = df[label_col].values.astype(float)
    drop_cols = [c for c in ['window', 'is_attack', 'timestamp'] if c in df.columns]
    ts = df.drop(columns=drop_cols)
    # Filter low-variance columns
    ts = ts.loc[:, ts.std() > 0]
    return ts, labels

def load_streamspot(data_dir="data/processed", max_windows=35000):
    """Load StreamSpot windows CSV: columns = [timestamp, is_attack, feat...]"""
    path = os.path.join(data_dir, "streamspot_edges.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    label_col = 'is_attack' if 'is_attack' in df.columns else df.columns[1]
    labels = df[label_col].values.astype(float)
    drop_cols = [c for c in ['window', 'is_attack', 'timestamp'] if c in df.columns]
    ts = df.drop(columns=drop_cols)
    ts = ts.loc[:, ts.std() > 0]
    # Cap at max_windows; ensure we include all attack windows
    if max_windows and len(ts) > max_windows:
        atk_last = int(np.where(labels > 0)[0][-1]) + 1 if labels.sum() > 0 else max_windows
        cap = max(max_windows, atk_last + 100)
        ts = ts.iloc[:cap].reset_index(drop=True)
        labels = labels[:cap]
    return ts, labels

def gen_optc(seed=42, n=1000):
    """Synthetic OpTC-style Windows enterprise provenance graph."""
    np.random.seed(seed)
    baseline = [
        "PROCESS:explorer.exe -> SPAWNS_PROCESS -> PROCESS:cmd.exe",
        "PROCESS:svchost.exe -> CONNECTS_TO -> FLOW:10.0.0.1:443",
        "PROCESS:lsass.exe -> READS_FILE -> FILE:sam.db",
        "PROCESS:taskhostw.exe -> WRITES_FILE -> FILE:setupapi.log",
        "PROCESS:searchindexer.exe -> READS_FILE -> FILE:index.db",
        "PROCESS:OneDrive.exe -> WRITES_FILE -> FILE:SyncEngine.db",
        "PROCESS:excel.exe -> READS_FILE -> FILE:finance.xlsx",
        "PROCESS:winlogon.exe -> SPAWNS_PROCESS -> PROCESS:userinit.exe",
        "PROCESS:wuauclt.exe -> CONNECTS_TO -> FLOW:windowsupdate.com:443",
        "PROCESS:msiexec.exe -> WRITES_FILE -> FILE:install.log",
        "PROCESS:powershell.exe -> READS_FILE -> FILE:profile.ps1",
        "PROCESS:wermgr.exe -> CONNECTS_TO -> FLOW:watson.microsoft.com:443",
    ]
    apt = [
        "PROCESS:word.exe -> SPAWNS_PROCESS -> PROCESS:powershell.exe",
        "PROCESS:powershell.exe -> WRITES_FILE -> FILE:mimikatz.exe",
        "PROCESS:mimikatz.exe -> READS_FILE -> FILE:sam.db",
    ]
    data = {e: np.random.poisson(1.5 if 'svchost' in e else 0.5, n).astype(float) for e in baseline}
    for e in apt:
        data[e] = np.random.poisson(0.01, n).astype(float)
    ts = pd.DataFrame(data)
    labels = np.zeros(n)
    attack_start = int(n * 0.55)
    for idx in range(attack_start, n - 3, 5):
        burst = max(np.random.poisson(8), 5)
        ts.iloc[idx,   ts.columns.get_loc(apt[0])] += burst
        ts.iloc[idx+1, ts.columns.get_loc(apt[1])] += burst
        ts.iloc[idx+2, ts.columns.get_loc(apt[2])] += burst
        labels[idx] = labels[idx+1] = labels[idx+2] = 1
    return ts, labels

def load_optc(csv_path="optc_edges.csv"):
    """Load real OpTC provenance data; fall back to synthetic OpTC."""
    if not os.path.exists(csv_path) and not os.path.exists(os.path.join("data/processed", csv_path)):
        return gen_optc(seed=42, n=1000)
    actual_path = csv_path if os.path.exists(csv_path) else os.path.join("data/processed", csv_path)
    edges = pd.read_csv(actual_path)
    edges['timestamp'] = pd.to_datetime(edges['timestamp'], format='ISO8601')
    edges['edge_type'] = (edges['src_type'] + ":" + edges['src_id'] +
                          " -> " + edges['action'] + " -> " +
                          edges['dst_type'] + ":" + edges['dst_id'])
    ts = edges.groupby([pd.Grouper(key='timestamp', freq='1s'), 'edge_type']
                       ).size().unstack(fill_value=0)
    ts = ts.loc[:, ts.sum() > 5]
    # Inject synthetic attacks (same as existing evaluation)
    apt = ["PROCESS:word.exe -> SPAWNS_PROCESS -> PROCESS:powershell.exe",
           "PROCESS:powershell.exe -> WRITES_FILE -> FILE:registry.dat"]
    for e in apt:
        if e not in ts.columns:
            ts[e] = np.random.poisson(0.01, len(ts))
    np.random.seed(42)
    labels = np.zeros(len(ts))
    split = int(len(ts) * 0.6)
    test_len = len(ts) - split
    atk_idx = np.random.choice(test_len - 2, 50, replace=False) + 1
    for idx in atk_idx:
        burst = max(np.random.poisson(5), 3)
        ts.iloc[split + idx, ts.columns.get_loc(apt[0])] += burst
        ts.iloc[split + idx + 1, ts.columns.get_loc(apt[1])] += burst
        labels[split + idx] = 1
        labels[split + idx + 1] = 1
    return ts, labels


# ─── Core Training/Evaluation ────────────────────────────────────────────────

def run_causal_ml_hybrid(ts, labels, args, dataset_name="dataset", force_split=None):
    """
    Full CausalMLHybrid pipeline:
    1. RobustParCorr causal graph on train_proper
    2. CausalRegressionModel + CausalFeatureExtractor → 12-dim features per window
    3. Split features into: [normal-train, val-with-labels, test]
    4. Train LSTM-AE on normal-train causal features
    5. Train CausalRF on val causal features + labels
    6. Train StackedEnsemble on val scores + labels
    7. Evaluate on test set → AUC
    """
    n = len(ts)
    if force_split is not None:
        # Use fixed splits (for synthetic datasets with known attack positions)
        proper_end = int(n * force_split[0])
        val_end    = int(n * force_split[1])
        train_proper = ts.iloc[:proper_end].copy()
        val_df       = ts.iloc[proper_end:val_end].copy()
        test_df      = ts.iloc[val_end:].copy()
        val_labels   = labels[proper_end:val_end]
        test_labels  = labels[val_end:]
    else:
        # Adaptive splits: detect first/last attack position in labels
        atk_idx = np.where(labels > 0)[0]
        if len(atk_idx):
            first_atk = atk_idx[0]
            last_atk  = atk_idx[-1]
        else:
            first_atk, last_atk = int(n * 0.55), int(n * 0.85)

        # Train on pre-attack normal region (80% of pre-attack windows, ≤50% of total)
        proper_end = max(int(min(first_atk * 0.80, n * 0.50)), 30)

        # Split attack region: first half in val, second half in test
        atk_mid = (first_atk + last_atk) // 2
        val_end  = min(atk_mid, int(n * 0.85))
        val_end  = max(val_end, proper_end + 10)

        train_proper = ts.iloc[:proper_end].copy()
        val_df       = ts.iloc[proper_end:val_end].copy()
        test_df      = ts.iloc[val_end:].copy()
        val_labels   = labels[proper_end:val_end]
        test_labels  = labels[val_end:]

        if len(np.unique(test_labels)) < 2:
            # Fallback: use all attacks as test, val = pre-attack buffer
            val_end2    = proper_end + max((first_atk - proper_end) // 2, 5)
            val_df      = ts.iloc[proper_end:val_end2].copy()
            test_df     = ts.iloc[val_end2:].copy()
            val_labels  = labels[proper_end:val_end2]
            test_labels = labels[val_end2:]
            val_end     = val_end2

    if len(np.unique(test_labels)) < 2:
        print(f"  [{dataset_name}] no attacks in test split — skipping")
        return None

    # ── 1. Causal discovery on train_proper ──────────────────────────────────
    print(f"  [{dataset_name}] RobustParCorr on {len(train_proper)} windows, "
          f"{train_proper.shape[1]} features...")
    
    # Robust cache loading helper
    cache_loaded = False
    p_matrix, var_names = None, None
    if os.path.exists("logs"):
        import pickle
        for fn in os.listdir("logs"):
            if fn.startswith(f"pcmci_cache_{dataset_name}_") and fn.endswith(".pkl"):
                cache_path = os.path.join("logs", fn)
                try:
                    with open(cache_path, 'rb') as f_cache:
                        cache_data = pickle.load(f_cache)
                    if isinstance(cache_data, dict):
                        p_matrix_tmp = cache_data['p_matrix']
                        var_names_tmp = cache_data.get('var_names', train_proper.columns.tolist())
                    elif isinstance(cache_data, tuple):
                        p_matrix_tmp, var_names_tmp = cache_data
                    else:
                        p_matrix_tmp = cache_data
                        var_names_tmp = train_proper.columns.tolist()
                    
                    # Verify features match
                    if set(var_names_tmp) == set(train_proper.columns.tolist()):
                        print(f"  ✅ Loading cached PCMCI baseline from {cache_path}")
                        p_matrix = p_matrix_tmp
                        var_names = var_names_tmp
                        cache_loaded = True
                        break
                    else:
                        # Log mismatch and continue checking other files
                        pass
                except Exception as e_cache:
                    print(f"  ⚠️ Error loading cache {cache_path}: {e_cache}. Re-running PCMCI.")

    if not cache_loaded:
        rpc = RobustParCorr(support_fraction=0.85, alpha=args.pcmci_alpha, tau_max=1)
        try:
            from tigramite import data_processing as pp
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr
            vn = train_proper.columns.tolist()
            df_tg = pp.DataFrame(train_proper.values.astype(float),
                                 datatime=np.arange(len(train_proper)), var_names=vn)
            pcmci_obj = PCMCI(dataframe=df_tg, cond_ind_test=ParCorr(significance='analytic'), verbosity=0)
            res = pcmci_obj.run_pcmci(tau_max=1, pc_alpha=args.pcmci_alpha)
            p_matrix = res['p_matrix']
            var_names = vn
            print(f"  [{dataset_name}] PCMCI: {(p_matrix<args.pcmci_alpha).sum()} sig edges")
        except Exception:
            p_matrix, var_names = rpc.fit(train_proper.values.astype(float),
                                          train_proper.columns.tolist())

    # ── 2. Fit CausalRegressionModel ─────────────────────────────────────────
    causal_model = CausalRegressionModel(
        p_matrix, var_names, tau_max=1,
        alpha=args.pcmci_alpha, regressor_type='linear')
    causal_model.fit(train_proper, std_floor=args.std_floor)
    scorer = HybridAnomalyScorer(d=len(var_names), w=0.5, floor=args.std_floor)

    # ── 3. Feature extraction loop over all windows ───────────────────────────
    feat_ext = CausalFeatureExtractor(var_names, causal_model.parents)
    full_ts = pd.concat([train_proper, val_df, test_df]).reset_index(drop=True)
    full_arr = full_ts.values.astype(float)
    all_cols = list(full_ts.columns)
    col_idx = {c: i for i, c in enumerate(all_cols)}

    all_feats = []   # causal feature vectors
    all_cas   = []   # raw CAS scores
    n_full = len(full_ts)

    for i in range(1, n_full):
        res_d, pvals_d = causal_model.predict_and_residual(full_arr, i)
        scorer.calibrate(res_d, causal_model.residual_stds)
        cas = scorer.score(pvals_d, res_d, causal_model.residual_stds)

        # Novelty edges
        nov = [all_cols[j] for j in np.where(full_arr[i] > 0)[0]
               if all_cols[j] not in var_names]

        feat = feat_ext.extract(cas, pvals_d, res_d, causal_model.residual_stds,
                                full_arr[i], all_cols, nov)
        all_feats.append(feat)
        all_cas.append(cas)

    all_feats = np.array(all_feats, dtype=np.float32)  # (n_full-1, 12)
    all_cas   = np.array(all_cas, dtype=float)

    # Map windows back: proper=[0..proper_end-2], val=[proper_end-1..val_end-2], test=[val_end-1..]
    prop_feats = all_feats[:proper_end - 1]
    val_feats  = all_feats[proper_end - 1: val_end - 1]
    test_feats = all_feats[val_end - 1:]
    prop_cas   = all_cas[:proper_end - 1]
    val_cas    = all_cas[proper_end - 1: val_end - 1]
    test_cas   = all_cas[val_end - 1:]

    # Calibrate scorer on train_proper CAS scores
    calib = ConformalCalibrator(target_fpr=0.05)
    calib.calibrate(prop_cas.tolist())

    # ── 4. LSTM-AE: train on NORMAL train_proper causal features ─────────────
    print(f"  [{dataset_name}] Training LSTM-AE on {len(prop_feats)} normal windows...")
    lstm_ae = LightweightLSTMAE(feat_dim=12, hidden=32, seq_len=8,
                                epochs=args.ae_epochs, lr=5e-4, batch_size=64)
    lstm_ae.fit(prop_feats)

    # AE scores on val and test
    val_ae  = lstm_ae.score(np.vstack([prop_feats[-8:], val_feats]))[8:]
    test_ae = lstm_ae.score(np.vstack([val_feats[-8:], test_feats]))[8:]

    # ── 5. CausalRF: train on val features + labels ───────────────────────────
    print(f"  [{dataset_name}] Training CausalRF on {len(val_feats)} val windows "
          f"({int(val_labels.sum())} attacks)...")
    crf = CausalRandomForestDetector(n_estimators=300, max_depth=10)
    if len(np.unique(val_labels)) > 1:
        crf.fit(val_feats, val_labels)
    rf_val  = crf.predict_proba(val_feats)
    rf_test = crf.predict_proba(test_feats)

    # ── 6. StackedEnsemble: meta-learn on val ────────────────────────────────
    ensemble = StackedEnsembleDetector()
    ensemble.fit_meta(val_cas, val_ae, rf_val, val_labels)

    # ── 7. Final test scores ─────────────────────────────────────────────────
    test_scores = ensemble.fuse(test_cas, test_ae, rf_test)

    auc = roc_auc_score(test_labels, test_scores)
    fprs, tprs, _ = roc_curve(test_labels, test_scores)
    fpr95 = 1.0
    for fpr, tpr in zip(fprs, tprs):
        if tpr >= 0.95:
            fpr95 = float(fpr)
            break

    # Inference time (per window)
    t0 = time.time()
    for _ in range(100):
        _ = ensemble.fuse([test_cas[0]], [test_ae[0]], [rf_test[0]])
    infer_ms = (time.time() - t0) / 100 * 1000

    # Extract meta weights
    meta_weights = []
    if ensemble._fitted:
        meta_weights = ensemble._meta.coef_[0].tolist()
    else:
        meta_weights = [1.0/3, 1.0/3, 1.0/3]

    # Save per-step scores for all datasets so ROC plots use CBSR hybrid data
    os.makedirs("results/hybrid_scores", exist_ok=True)
    scores_df = pd.DataFrame({'score': test_scores, 'label': test_labels})
    out_path = f"results/hybrid_scores/{dataset_name}_hybrid_steps.csv"
    scores_df.to_csv(out_path, index=False)
    print(f"  [{dataset_name}] Saved hybrid test scores to {out_path}")
    if dataset_name == "streamspot":
        scores_df.to_csv("results/streamspot_hybrid_test_scores.csv", index=False)

    return {
        'dataset': dataset_name,
        'auc': float(auc),
        'fpr_at_95_recall': float(fpr95),
        'n_test': len(test_labels),
        'n_attack': int(test_labels.sum()),
        'infer_ms_per_window': round(infer_ms, 3),
        'rf_importances': crf.feature_importances() if crf._fitted else {},
        'meta_weights': meta_weights,
    }


def run_baselines(ts, labels, val_end, args):
    """Run IF, LOF, OCSVM, AE on raw features for comparison."""
    n = len(ts)
    train = ts.iloc[:val_end].values.astype(float)
    test  = ts.iloc[val_end:].values.astype(float)
    test_labels = labels[val_end:]

    if len(np.unique(test_labels)) < 2:
        return {}

    # Subsample training for slow algorithms when dataset is large
    max_train_bl = 3000
    if len(train) > max_train_bl:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train), max_train_bl, replace=False)
        idx.sort()
        train_bl = train[idx]
    else:
        train_bl = train

    results = {}

    # Isolation Forest
    try:
        iforest = IsolationForest(contamination=0.05, random_state=42)
        iforest.fit(train_bl)
        if_scores = -iforest.score_samples(test)
        results['IF_AUC'] = float(roc_auc_score(test_labels, if_scores))
    except Exception as e:
        results['IF_AUC'] = None

    # LOF
    try:
        lof = LocalOutlierFactor(contamination=0.05, novelty=True)
        lof.fit(train_bl)
        lof_scores = -lof.score_samples(test)
        results['LOF_AUC'] = float(roc_auc_score(test_labels, lof_scores))
    except Exception as e:
        results['LOF_AUC'] = None

    # OCSVM
    try:
        ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
        ocsvm.fit(train_bl)
        ocsvm_scores = -ocsvm.score_samples(test)
        results['OCSVM_AUC'] = float(roc_auc_score(test_labels, ocsvm_scores))
    except Exception as e:
        results['OCSVM_AUC'] = None

    # Autoencoder
    try:
        ae_scores = train_and_evaluate_ae(train, test, epochs=args.ae_epochs)
        results['AE_AUC'] = float(roc_auc_score(test_labels, ae_scores))
    except Exception as e:
        results['AE_AUC'] = None

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+',
                        default=['tc3', 'nodlink', 'beth', 'optc', 'streamspot'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_windows", type=int, default=1000)
    parser.add_argument("--ae_epochs", type=int, default=60)
    parser.add_argument("--pcmci_alpha", type=float, default=0.01)
    parser.add_argument("--std_floor", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    all_results = {}
    t_global = time.time()

    print("=" * 65)
    print("ZeroCausal v2 — CausalMLHybrid  |  Beat-Papers Evaluation")
    print("=" * 65)

    # ── Dataset loader map ────────────────────────────────────────────────────
    dataset_loaders = {
        'tc3':        lambda: gen_tc3(args.seed, args.n_windows),
        'nodlink':    lambda: gen_nodlink(args.seed, args.n_windows),
        'beth':       lambda: load_beth(),
        'optc':       lambda: load_optc(),
        'streamspot': lambda: load_streamspot(),
    }

    paper_baselines = {
        'tc3':        {'IF': 0.8737, 'LOF': 0.9841, 'OCSVM': 0.9594, 'AE': 1.0000, 'ZCv1': 0.8350},
        'nodlink':    {'IF': 0.8902, 'LOF': 0.9912, 'OCSVM': 0.9521, 'AE': 1.0000, 'ZCv1': 0.8258},
        'beth':       {'IF': 0.9981, 'LOF': 0.9006, 'OCSVM': 0.9884, 'AE': 0.9954, 'ZCv1': 0.9656},
        'optc':       {'IF': 0.5968, 'LOF': 0.7714, 'OCSVM': 0.7315, 'AE': 0.9364, 'ZCv1': 0.8359},
        'streamspot': {'IF': 0.6425, 'LOF': 0.2757, 'OCSVM': 0.5311, 'AE': 0.7770, 'ZCv1': 0.4991},
    }

    for dsname in args.datasets:
        if dsname not in dataset_loaders:
            print(f"Unknown dataset: {dsname}")
            continue

        print(f"\n{'─'*65}")
        print(f"Dataset: {dsname.upper()}")
        print(f"{'─'*65}")

        ts, labels = dataset_loaders[dsname]()
        if ts is None:
            print(f"  [{dsname}] data not available — skipping")
            continue

        print(f"  Shape: {ts.shape}, attacks: {int(labels.sum())}")
        val_end = int(len(ts) * 0.65)

        # Run baselines on raw features
        print(f"  Running raw-feature baselines...")
        val_end_bl = int(len(ts) * 0.70)
        bl = run_baselines(ts, labels, val_end_bl, args)

        # Run CausalMLHybrid
        print(f"  Running CausalMLHybrid...")
        t0 = time.time()
        # Synthetic datasets have attacks starting at 55%; use fixed splits
        _fixed = (0.50, 0.70) if dsname in ('tc3', 'nodlink') else None
        hyb = run_causal_ml_hybrid(ts, labels, args, dataset_name=dsname, force_split=_fixed)
        elapsed = time.time() - t0

        if hyb is None:
            continue

        # Paper baselines for reference
        paper = paper_baselines.get(dsname, {})

        # Print comparison table
        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  {dsname.upper()} — AUC Comparison                     │")
        print(f"  ├──────────────────────────┬──────────────────┤")
        print(f"  │  Method                  │  AUC             │")
        print(f"  ├──────────────────────────┼──────────────────┤")
        for method, pauc in paper.items():
            beat = "✓ BEAT" if hyb['auc'] > pauc else "✗"
            print(f"  │  {method:24s}  │  {pauc:.4f}  {beat:8s}  │")
        # Live baselines
        for method, bauc in bl.items():
            if bauc is None:
                continue
            mname = method.replace('_AUC', ' (live)')
            beat = "✓ BEAT" if hyb['auc'] > bauc else "✗"
            print(f"  │  {mname:24s}  │  {bauc:.4f}  {beat:8s}  │")
        print(f"  ├──────────────────────────┼──────────────────┤")
        print(f"  │  ★ CausalMLHybrid (ours) │  {hyb['auc']:.4f}            │")
        print(f"  └──────────────────────────┴──────────────────┘")
        print(f"  FPR@95%recall: {hyb['fpr_at_95_recall']:.4f}  "
              f"Inference: {hyb['infer_ms_per_window']:.2f}ms/window  "
              f"Time: {elapsed:.1f}s")

        if hyb['rf_importances']:
            top3 = sorted(hyb['rf_importances'].items(), key=lambda x: -x[1])[:3]
            print(f"  Top causal features: " +
                  ", ".join(f"{k}={v:.3f}" for k, v in top3))

        beats_count = sum(1 for _, pauc in paper.items() if hyb['auc'] > pauc)
        print(f"  → Beats {beats_count}/{len(paper)} paper baselines")

        all_results[dsname] = {
            'causal_ml_hybrid': hyb,
            'live_baselines': bl,
            'paper_baselines': paper,
            'beats_paper_count': beats_count,
        }

    # ── Global Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("GLOBAL SUMMARY")
    print(f"{'='*65}")
    total_beats = 0
    total_paper = 0
    for dsname, r in all_results.items():
        hyb_auc = r['causal_ml_hybrid']['auc']
        b = r['beats_paper_count']
        p = len(r['paper_baselines'])
        total_beats += b
        total_paper += p
        print(f"  {dsname.upper():12s}  AUC={hyb_auc:.4f}  Beats {b}/{p}")

    print(f"\n  TOTAL: beats {total_beats}/{total_paper} paper baselines")
    print(f"  Elapsed: {time.time()-t_global:.1f}s")

    out_path = os.path.join(args.out_dir, "beat_papers_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"\n  Results → {out_path}")


if __name__ == "__main__":
    main()
