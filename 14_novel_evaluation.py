"""
14_novel_evaluation.py
======================
End-to-end evaluation of all 8 novel ZeroCausal components on the TC3
simulation dataset.  Each component is tested independently and then
combined to show cumulative gains.

Novel Components Tested
-----------------------
N1  RobustParCorr            — contamination-aware causal discovery
N2  WeightedConformalCalibrator — non-exchangeable conformal prediction
N3  CausalInterventionScorer — attacker-effort estimation
N4  CausalGraphEvolutionDetector — structural causal-change detection
N5  KalmanSCM               — continuously evolving SCM
N6  SelfHealingCalibration  — retroactive calibration poisoning removal
N7  CausalRobustnessMetric  — pre-deployment detectability certificate
N8  MultiScaleCausalFusion  — multi-horizon causal fusion

Usage
-----
python 14_novel_evaluation.py [--contamination 0.05] [--noise_level 0.1]
                              [--num_windows 800] [--seed 42]
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from zerocausal_core import (
    AdaptiveWindowDetector,
    CausalRegressionModel,
    HybridAnomalyScorer,
    ConformalCalibrator,
    RobustCalibrationFilter,
    # Novel components
    RobustParCorr,
    WeightedConformalCalibrator,
    CausalInterventionScorer,
    CausalGraphEvolutionDetector,
    KalmanSCM,
    SelfHealingCalibration,
    CausalRobustnessMetric,
    MultiScaleCausalFusion,
)

# ─── Data Generation (TC3-style) ────────────────────────────────────────────

def generate_tc3(num_windows=800, noise_level=0.1, contamination=0.0, seed=42):
    """
    Generates TC3-style Windows provenance stream with:
    - 15 normal baseline edges (Poisson-distributed)
    - Synthetic causal dependency between two pairs
    - 3-stage APT dropper attack in last 20% of stream
    - Optional contamination: replaces `contamination` fraction of training
      windows with mild attack-like bursts to test robustness
    """
    np.random.seed(seed)
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
        "PROCESS:unknown_process -> READS_FILE -> FILE:gpt.ini",
    ]
    apt_edges = [
        "PROCESS:word.exe -> SPAWNS_PROCESS -> PROCESS:powershell.exe",
        "PROCESS:powershell.exe -> WRITES_FILE -> FILE:payload.exe",
        "PROCESS:payload.exe -> CONNECTS_TO -> FLOW:185.220.101.5:4444",
    ]
    all_cols = baseline_edges + apt_edges
    data = {}

    # Normal activity (Poisson)
    for edge in baseline_edges:
        lam = 3.0 if ("svchost" in edge or "chrome" in edge) else 0.5
        data[edge] = np.random.poisson(lam=lam + noise_level, size=num_windows).astype(float)

    # Synthetic causal link: spoolsv depends on svchost
    data["PROCESS:spoolsv.exe -> READS_FILE -> FILE:printers.db"] += (
        data["PROCESS:svchost.exe -> CONNECTS_TO -> FLOW:8.8.8.8:53"] > 2
    ).astype(float)

    # APT edges: zero during baseline, burst during attack
    for edge in apt_edges:
        data[edge] = np.zeros(num_windows)

    ts = pd.DataFrame(data)
    labels = np.zeros(num_windows)

    # Inject attack in last 20%
    attack_start = int(num_windows * 0.80)
    attack_windows = list(range(attack_start, num_windows - 2))
    for idx in attack_windows:
        burst = max(np.random.poisson(6), 4)
        ts.iloc[idx, ts.columns.get_loc(apt_edges[0])] += burst
        ts.iloc[idx + 1, ts.columns.get_loc(apt_edges[1])] += burst
        ts.iloc[idx + 2, ts.columns.get_loc(apt_edges[2])] += burst
        labels[idx] = 1
        labels[idx + 1] = 1
        labels[idx + 2] = 1

    # Contaminate training partition
    if contamination > 0.0:
        train_end = int(num_windows * 0.6)
        n_contaminate = max(1, int(train_end * contamination))
        c_indices = np.random.choice(train_end, n_contaminate, replace=False)
        for idx in c_indices:
            for edge in apt_edges:
                ts.iloc[idx, ts.columns.get_loc(edge)] += np.random.poisson(3)

    return ts, labels, apt_edges, baseline_edges


# ─── Shared Setup ────────────────────────────────────────────────────────────

def split_data(ts, labels):
    n = len(ts)
    train_end = int(n * 0.60)
    calib_split = int(train_end * 0.70)
    train_proper = ts.iloc[:calib_split].copy()
    train_calib  = ts.iloc[calib_split:train_end].copy()
    test         = ts.iloc[train_end:].copy()
    test_labels  = labels[train_end:]
    return train_proper, train_calib, test, test_labels, calib_split, train_end


def run_pcmci(train_proper, alpha=0.01, tau_max=1):
    """Run standard PCMCI; return p_matrix and var_names."""
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        var_names = train_proper.columns.tolist()
        data = train_proper.astype(float).values
        df_tg = pp.DataFrame(data, datatime=np.arange(len(data)), var_names=var_names)
        cit = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=df_tg, cond_ind_test=cit, verbosity=0)
        res = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)
        return res['p_matrix'], var_names
    except Exception as e:
        print(f"  [PCMCI fallback — {e}] using RobustParCorr for p_matrix")
        return None, train_proper.columns.tolist()


def run_robust_parcorr(train_proper, alpha=0.01, tau_max=1):
    """Run RobustParCorr (N1); return p_matrix and var_names."""
    var_names = train_proper.columns.tolist()
    data = train_proper.astype(float).values
    rpc = RobustParCorr(support_fraction=0.85, alpha=alpha, tau_max=tau_max)
    p_matrix, _ = rpc.fit(data, var_names)
    return p_matrix, var_names


def evaluate_stream(ts, labels, p_matrix, var_names, calibrator_cls, args,
                    rcf=None, kalman=False, self_heal=None,
                    graph_evo=None, multi_scale=None, cis_scorer=None):
    """
    Generic streaming evaluation loop.
    Returns (auc, fpr_95recall, empirical_fpr, scores, step_logs)
    """
    n = len(ts)
    train_end = int(n * 0.60)
    calib_split = int(train_end * 0.70)
    train_proper = ts.iloc[:calib_split].copy()
    train_calib  = ts.iloc[calib_split:train_end].copy()
    test         = ts.iloc[train_end:].copy()
    test_labels  = labels[train_end:]

    # Fit causal regression
    causal_model = CausalRegressionModel(
        p_matrix, var_names, tau_max=args.tau_max,
        alpha=args.pcmci_alpha, regressor_type='linear')
    causal_model.fit(train_proper, std_floor=args.std_floor)

    # Optionally init Kalman SCM
    if kalman:
        kal = KalmanSCM(process_noise_var=1e-4, observation_noise_var=1.0)
        kal.init_from_causal_model(causal_model)
    else:
        kal = None

    # Scorer (H-CAS)
    scorer = HybridAnomalyScorer(d=len(var_names), w=0.5, floor=args.std_floor)

    # Self-heal setup
    if self_heal is not None:
        shc = SelfHealingCalibration(heal_threshold=0.85, max_heal_fraction=0.30)
    else:
        shc = None

    # Build calibration scores
    full_train = ts.iloc[:train_end].copy()
    full_arr = full_train.values
    calib_scores = []
    calib_z_vecs = []
    for i in range(args.tau_max, len(train_calib)):
        idx = calib_split + i
        res, pvals = causal_model.predict_and_residual(full_arr, idx)
        scorer.calibrate(res, causal_model.residual_stds)
        s = scorer.score(pvals, res, causal_model.residual_stds)
        z_vec = np.array([res.get(v, 0.0) / max(causal_model.residual_stds.get(v, 1.0), 1e-9)
                          for v in var_names])
        p_min = min(pvals.values()) if pvals else 1.0
        # Gate via RCF
        if rcf is not None:
            if rcf.admit(p_min, s, z_vec):
                calib_scores.append(s)
                if shc is not None:
                    shc.add(s, z_vec)
                    calib_z_vecs.append(z_vec)
        else:
            calib_scores.append(s)
            if shc is not None:
                shc.add(s, z_vec)

    # Setup calibrator
    calibrator = calibrator_cls()
    calibrator.calibrate(calib_scores)

    if rcf is not None:
        rcf.seed_energy(calib_scores)

    # Graph evolution baseline
    if graph_evo is not None:
        graph_evo.set_baseline(p_matrix)

    # Detector (change-point)
    detector = AdaptiveWindowDetector(
        num_features=len(var_names),
        short_window=args.detector_short,
        long_window=args.detector_long,
        threshold=args.detector_threshold)

    history = pd.concat([ts.iloc[train_end - 1:train_end], test])
    hist_arr = history.values
    col_to_idx = {c: i for i, c in enumerate(history.columns)}
    var_indices = [col_to_idx[v] for v in var_names if v in col_to_idx]

    scores_out = np.zeros(len(test))
    alarms_out = np.zeros(len(test))

    for i in range(args.tau_max, len(test) + 1):
        row = hist_arr[i]
        feat_subset = row[var_indices]
        change = detector.update(feat_subset)
        if change and rcf is not None:
            rcf.set_changepoint()

        # Compute residuals
        if kal is not None:
            x_dict = {v: float(hist_arr[i, col_to_idx[v]])
                      for v in var_names if v in col_to_idx}
            res, res_stds = kal.predict_and_update(x_dict, var_names)
            pvals = {}
            for v in var_names:
                z = res.get(v, 0.0) / max(res_stds.get(v, 1.0), 1e-9)
                pvals[v] = float(np.clip(2.0 * (1.0 - scipy.stats.norm.cdf(abs(z))), 1e-15, 1.0))
        else:
            res, pvals = causal_model.predict_and_residual(hist_arr, i)
            res_stds = causal_model.residual_stds

        z_vec = np.array([res.get(v, 0.0) / max(res_stds.get(v, 1.0), 1e-9)
                          for v in var_names])
        p_min = min(pvals.values()) if pvals else 1.0

        # CAS score
        cas = scorer.score(pvals, res, res_stds)

        # Optional graph evolution blend
        if graph_evo is not None:
            ged_score = graph_evo.update_and_score(p_matrix)
            cas = 0.7 * cas + 0.3 * float(np.clip(ged_score, 0, 1))

        scores_out[i - 1] = cas

        # Multi-scale override
        if multi_scale is not None:
            ms_score = multi_scale.score(hist_arr, i, col_to_idx)
            if ms_score > 0.0:
                scores_out[i - 1] = 0.6 * cas + 0.4 * ms_score

        # Conformal p-value
        if hasattr(calibrator, 'compute_conformal_pvalue'):
            cp = calibrator.compute_conformal_pvalue(cas)
        else:
            cp = calibrator.compute_conformal_pvalue(cas)

        alarm = 1.0 if cp < calibrator.alpha else 0.0
        alarms_out[i - 1] = alarm
        calibrator.update_threshold(alarm)

        # Calibration update
        if alarm == 0.0:
            if rcf is not None:
                if rcf.admit(p_min, cas, z_vec):
                    calibrator.update_calibration(cas)
                    if shc is not None:
                        shc.add(cas, z_vec)
            else:
                calibrator.update_calibration(cas)
                if shc is not None:
                    shc.add(cas, z_vec)
        elif alarm == 1.0 and shc is not None:
            # Self-heal: remove poisoned calibration entries
            removed = shc.heal(z_vec)
            if removed:
                clean_scores = shc.get_scores()
                calibrator.calibrate(clean_scores)

    # Metrics
    auc = roc_auc_score(test_labels, scores_out) if len(np.unique(test_labels)) > 1 else 0.5
    normal_mask = (test_labels == 0)
    emp_fpr = float(np.mean(alarms_out[normal_mask])) if normal_mask.sum() > 0 else 0.0

    fprs, tprs, _ = roc_curve(test_labels, scores_out)
    fpr95 = 1.0
    for fpr, tpr in zip(fprs, tprs):
        if tpr >= 0.95:
            fpr95 = float(fpr)
            break

    return auc, fpr95, emp_fpr, scores_out


# ─── Conformal calibrator adapters ─────────────────────────────────────────

class _StdCalibAdapter:
    """Thin wrapper so ConformalCalibrator matches the generic interface."""
    def __init__(self):
        self._c = ConformalCalibrator(target_fpr=0.05, lr=0.05, alpha_init=0.05)
        self.alpha = self._c.alpha

    def calibrate(self, scores):
        self._c.calibrate(scores)
        self.alpha = self._c.alpha

    def compute_conformal_pvalue(self, score):
        return self._c.compute_conformal_pvalue(score)

    def update_calibration(self, score, max_size=245):
        self._c.update_calibration(score, max_size)

    def update_threshold(self, alarm):
        self.alpha = self._c.update_threshold(alarm)


class _WeightedCalibAdapter:
    """Thin wrapper so WeightedConformalCalibrator matches the generic interface."""
    def __init__(self):
        self._c = WeightedConformalCalibrator(lambda_decay=0.01, target_fpr=0.05)
        self.alpha = self._c.alpha

    def calibrate(self, scores):
        self._c.calibrate(scores)
        self.alpha = self._c.alpha

    def compute_conformal_pvalue(self, score):
        return self._c.compute_conformal_pvalue(score)

    def update_calibration(self, score, max_size=500):
        self._c.update_calibration(score, max_size)

    def update_threshold(self, alarm):
        self.alpha = self._c.update_threshold(alarm)


# ─── Experiments ─────────────────────────────────────────────────────────────

def experiment_baseline(ts, labels, args, tag="Baseline"):
    p_matrix, var_names = run_pcmci(ts.iloc[:int(len(ts)*0.6*0.7)], args.pcmci_alpha, args.tau_max)
    if p_matrix is None:
        p_matrix, var_names = run_robust_parcorr(ts.iloc[:int(len(ts)*0.6*0.7)], args.pcmci_alpha, args.tau_max)
    auc, fpr95, emp_fpr, _ = evaluate_stream(ts, labels, p_matrix, var_names,
                                              _StdCalibAdapter, args)
    print(f"  [{tag}]  AUC={auc:.4f}  FPR@95%recall={fpr95:.4f}  EmpFAR={emp_fpr:.4f}")
    return {'tag': tag, 'auc': auc, 'fpr95': fpr95, 'emp_fpr': emp_fpr}


def experiment_n1_robust_parcorr(ts, labels, args, contamination):
    """N1: RobustParCorr vs. standard ParCorr under contamination."""
    train_proper = ts.iloc[:int(len(ts)*0.6*0.7)]

    # Standard
    p_std, vn_std = run_pcmci(train_proper, args.pcmci_alpha, args.tau_max)
    if p_std is None:
        p_std, vn_std = run_robust_parcorr(train_proper, args.pcmci_alpha, args.tau_max)
    auc_std, *_ = evaluate_stream(ts, labels, p_std, vn_std, _StdCalibAdapter, args)

    # Robust
    p_rob, vn_rob = run_robust_parcorr(train_proper, args.pcmci_alpha, args.tau_max)
    auc_rob, *_ = evaluate_stream(ts, labels, p_rob, vn_rob, _StdCalibAdapter, args)

    print(f"  [N1-RobustParCorr @ε={contamination:.0%}]  "
          f"Standard AUC={auc_std:.4f}  Robust AUC={auc_rob:.4f}  "
          f"Delta={auc_rob-auc_std:+.4f}")
    return {'standard_auc': auc_std, 'robust_auc': auc_rob,
            'delta': auc_rob - auc_std, 'contamination': contamination}


def experiment_n2_weighted_conformal(ts, labels, args):
    """N2: WeightedConformal vs. standard conformal."""
    train_proper = ts.iloc[:int(len(ts)*0.6*0.7)]
    p_matrix, var_names = run_robust_parcorr(train_proper, args.pcmci_alpha, args.tau_max)

    auc_std, fpr95_std, fpr_std, _ = evaluate_stream(
        ts, labels, p_matrix, var_names, _StdCalibAdapter, args)
    auc_w, fpr95_w, fpr_w, _ = evaluate_stream(
        ts, labels, p_matrix, var_names, _WeightedCalibAdapter, args)

    # Theoretical FAR bound (bounded drift TV=0.002 per step, λ=0.05)
    bound = CausalRobustnessMetric.theorem1_far_bound(
        alpha=0.05, epsilon=args.contamination, drift_tv=0.002, lambda_decay=0.05)

    print(f"  [N2-WeightedConformal]  "
          f"Std AUC={auc_std:.4f}/FAR={fpr_std:.4f}  "
          f"Weighted AUC={auc_w:.4f}/FAR={fpr_w:.4f}  "
          f"TheoreticalBound={bound:.4f}")
    return {'std_auc': auc_std, 'weighted_auc': auc_w,
            'std_far': fpr_std, 'weighted_far': fpr_w,
            'theoretical_far_bound': bound}


def experiment_n3_intervention_score(ts, labels, args):
    """N3: CausalInterventionScore distribution on normal vs. attack windows."""
    train_proper = ts.iloc[:int(len(ts)*0.6*0.7)]
    p_matrix, var_names = run_robust_parcorr(train_proper, args.pcmci_alpha, args.tau_max)
    causal_model = CausalRegressionModel(
        p_matrix, var_names, tau_max=args.tau_max,
        alpha=args.pcmci_alpha, regressor_type='linear')
    causal_model.fit(train_proper, std_floor=args.std_floor)

    cis_scorer = CausalInterventionScorer(z_thresh=2.5)
    train_end = int(len(ts) * 0.60)
    test = ts.iloc[train_end:].copy()
    test_labels = labels[train_end:]
    hist_arr = ts.values

    cis_normal, cis_attack = [], []
    for i in range(args.tau_max, len(test)):
        idx = train_end + i
        if idx >= len(ts):
            break
        res, pvals = causal_model.predict_and_residual(hist_arr, idx)
        z_dict = {v: res.get(v, 0.0) / max(causal_model.residual_stds.get(v, 1.0), 1e-9)
                  for v in var_names}
        cis, _, _ = cis_scorer.score(z_dict, causal_model.parents)
        if test_labels[i] == 1:
            cis_attack.append(cis)
        else:
            cis_normal.append(cis)

    mn = float(np.mean(cis_normal)) if cis_normal else 0.0
    ma = float(np.mean(cis_attack)) if cis_attack else 0.0
    print(f"  [N3-CIS]  Normal mean CIS={mn:.4f}  Attack mean CIS={ma:.4f}  "
          f"Separation={ma-mn:+.4f}")
    return {'cis_normal_mean': mn, 'cis_attack_mean': ma, 'separation': ma - mn}


def experiment_n4_graph_evolution(ts, labels, args):
    """N4: CausalGraphEvolutionDetector as standalone scorer."""
    train_proper = ts.iloc[:int(len(ts)*0.6*0.7)]
    p_matrix, var_names = run_robust_parcorr(train_proper, args.pcmci_alpha, args.tau_max)
    ged = CausalGraphEvolutionDetector(ewma_alpha=0.15)
    auc, fpr95, emp_fpr, _ = evaluate_stream(
        ts, labels, p_matrix, var_names,
        _StdCalibAdapter, args, graph_evo=ged)
    births, deaths = ged.structural_births_deaths()
    print(f"  [N4-GraphEvolution]  AUC={auc:.4f}  "
          f"EdgeBirths={len(births)}  EdgeDeaths={len(deaths)}")
    return {'auc': auc, 'fpr95': fpr95, 'emp_fpr': emp_fpr,
            'n_edge_births': len(births), 'n_edge_deaths': len(deaths)}


def experiment_n5_kalman_scm(ts, labels, args):
    """N5: KalmanSCM (continuously evolving mechanisms)."""
    train_proper = ts.iloc[:int(len(ts)*0.6*0.7)]
    p_matrix, var_names = run_robust_parcorr(train_proper, args.pcmci_alpha, args.tau_max)
    auc, fpr95, emp_fpr, _ = evaluate_stream(
        ts, labels, p_matrix, var_names,
        _StdCalibAdapter, args, kalman=True)
    print(f"  [N5-KalmanSCM]  AUC={auc:.4f}  FPR@95%recall={fpr95:.4f}  EmpFAR={emp_fpr:.4f}")
    return {'auc': auc, 'fpr95': fpr95, 'emp_fpr': emp_fpr}


def experiment_n6_self_healing(ts, labels, args):
    """N6: SelfHealingCalibration."""
    train_proper = ts.iloc[:int(len(ts)*0.6*0.7)]
    p_matrix, var_names = run_robust_parcorr(train_proper, args.pcmci_alpha, args.tau_max)
    rcf = RobustCalibrationFilter()
    auc, fpr95, emp_fpr, _ = evaluate_stream(
        ts, labels, p_matrix, var_names,
        _StdCalibAdapter, args, rcf=rcf, self_heal=True)
    print(f"  [N6-SelfHeal]  AUC={auc:.4f}  FPR@95%recall={fpr95:.4f}  EmpFAR={emp_fpr:.4f}")
    return {'auc': auc, 'fpr95': fpr95, 'emp_fpr': emp_fpr}


def experiment_n7_robustness_metric(ts, labels, args):
    """N7: CausalRobustnessMetric pre-deployment certificate."""
    train_proper = ts.iloc[:int(len(ts)*0.6*0.7)]
    p_matrix, var_names = run_robust_parcorr(train_proper, args.pcmci_alpha, args.tau_max)
    causal_model = CausalRegressionModel(
        p_matrix, var_names, tau_max=args.tau_max,
        alpha=args.pcmci_alpha, regressor_type='linear')
    causal_model.fit(train_proper, std_floor=args.std_floor)

    crm = CausalRobustnessMetric()
    results = []
    for eps in [0.0, 0.01, 0.05, 0.10, 0.20]:
        r = crm.compute(p_matrix, causal_model.residual_stds, eps)
        bound = CausalRobustnessMetric.theorem1_far_bound(0.05, eps, drift_tv=0.002, lambda_decay=0.05)
        r['far_bound'] = round(bound, 4)
        results.append(r)
        print(f"  [N7-CRM @ε={eps:.2f}]  ρ={r['rho']:.3f}  "
              f"Detectable={r['detectable']}  FAR≤{bound:.4f}")
    return results


def experiment_n8_multiscale(ts, labels, args):
    """N8: MultiScaleCausalFusion."""
    train_proper = ts.iloc[:int(len(ts)*0.6*0.7)]
    p_matrix, var_names = run_robust_parcorr(train_proper, args.pcmci_alpha, args.tau_max)

    msf = MultiScaleCausalFusion(tau_max=args.tau_max, alpha=args.pcmci_alpha)
    msf.fit(train_proper, p_matrix, var_names)

    # Seed scale calibration variances with uniform placeholder
    for scale in ['process', 'file', 'network']:
        msf.calibrate_scale(scale, [0.5] * 50)

    auc, fpr95, emp_fpr, _ = evaluate_stream(
        ts, labels, p_matrix, var_names,
        _StdCalibAdapter, args, multi_scale=msf)
    print(f"  [N8-MultiScale]  AUC={auc:.4f}  FPR@95%recall={fpr95:.4f}  EmpFAR={emp_fpr:.4f}")
    return {'auc': auc, 'fpr95': fpr95, 'emp_fpr': emp_fpr}


def experiment_full_combination(ts, labels, args):
    """All novelties combined: RobustParCorr + WeightedConformal + RCF + SelfHeal + Kalman."""
    train_proper = ts.iloc[:int(len(ts)*0.6*0.7)]
    p_matrix, var_names = run_robust_parcorr(train_proper, args.pcmci_alpha, args.tau_max)
    rcf = RobustCalibrationFilter()
    auc, fpr95, emp_fpr, _ = evaluate_stream(
        ts, labels, p_matrix, var_names,
        _WeightedCalibAdapter, args, rcf=rcf, kalman=True, self_heal=True)
    print(f"  [FULL]  AUC={auc:.4f}  FPR@95%recall={fpr95:.4f}  EmpFAR={emp_fpr:.4f}")
    return {'auc': auc, 'fpr95': fpr95, 'emp_fpr': emp_fpr}


# ─── Contamination Sweep ────────────────────────────────────────────────────

def contamination_sweep(args):
    print("\n=== Contamination Sweep (N1 RobustParCorr) ===")
    sweep_results = []
    for eps in [0.0, 0.01, 0.05, 0.10, 0.20]:
        ts, labels, *_ = generate_tc3(
            num_windows=args.num_windows, noise_level=args.noise_level,
            contamination=eps, seed=args.seed)
        r = experiment_n1_robust_parcorr(ts, labels, args, eps)
        sweep_results.append(r)
    return sweep_results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZeroCausal Novel Components Evaluation")
    parser.add_argument("--num_windows", type=int, default=800)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau_max", type=int, default=1)
    parser.add_argument("--pcmci_alpha", type=float, default=0.01)
    parser.add_argument("--std_floor", type=float, default=1.0)
    parser.add_argument("--detector_short", type=int, default=15)
    parser.add_argument("--detector_long", type=int, default=60)
    parser.add_argument("--detector_threshold", type=float, default=3.5)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("ZeroCausal Novel Components Evaluation")
    print("=" * 60)

    # Generate data with configured contamination
    ts, labels, apt_edges, baseline_edges = generate_tc3(
        num_windows=args.num_windows, noise_level=args.noise_level,
        contamination=args.contamination, seed=args.seed)
    print(f"Dataset: {len(ts)} windows, {ts.shape[1]} edge types, "
          f"attack windows: {int(labels.sum())}")

    results = {}

    print("\n--- Baseline ---")
    results['baseline'] = experiment_baseline(ts, labels, args)

    print("\n--- N1: Contamination-Aware Causal Discovery (RobustParCorr) ---")
    results['n1_robust_parcorr'] = experiment_n1_robust_parcorr(
        ts, labels, args, args.contamination)

    print("\n--- N2: Non-Exchangeable Conformal Prediction ---")
    results['n2_weighted_conformal'] = experiment_n2_weighted_conformal(ts, labels, args)

    print("\n--- N3: Causal Intervention Score ---")
    results['n3_cis'] = experiment_n3_intervention_score(ts, labels, args)

    print("\n--- N4: Causal Graph Evolution Detector ---")
    results['n4_graph_evo'] = experiment_n4_graph_evolution(ts, labels, args)

    print("\n--- N5: Kalman-SCM (Continuously Evolving Mechanisms) ---")
    results['n5_kalman'] = experiment_n5_kalman_scm(ts, labels, args)

    print("\n--- N6: Self-Healing Calibration ---")
    results['n6_self_heal'] = experiment_n6_self_healing(ts, labels, args)

    print("\n--- N7: Causal Robustness Metric (Pre-deployment) ---")
    results['n7_robustness'] = experiment_n7_robustness_metric(ts, labels, args)

    print("\n--- N8: Multi-Scale Causal Fusion ---")
    results['n8_multiscale'] = experiment_n8_multiscale(ts, labels, args)

    print("\n--- FULL COMBINATION ---")
    results['full_combination'] = experiment_full_combination(ts, labels, args)

    print("\n--- Contamination Sweep (N1) ---")
    results['contamination_sweep'] = contamination_sweep(args)

    # Save
    out_path = os.path.join(args.out_dir, "novel_evaluation.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print(f"Results saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    for tag in ['baseline', 'n1_robust_parcorr', 'n2_weighted_conformal',
                'n4_graph_evo', 'n5_kalman', 'n6_self_heal', 'n8_multiscale',
                'full_combination']:
        r = results.get(tag, {})
        if isinstance(r, dict):
            auc = r.get('auc', r.get('robust_auc', r.get('weighted_auc', '-')))
            fpr = r.get('fpr95', r.get('std_far', '-'))
            efar = r.get('emp_fpr', r.get('weighted_far', '-'))
            auc_s = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
            fpr_s = f"{fpr:.4f}" if isinstance(fpr, float) else str(fpr)
            ef_s = f"{efar:.4f}" if isinstance(efar, float) else str(efar)
            print(f"  {tag:35s}  AUC={auc_s}  FPR95={fpr_s}  EmpFAR={ef_s}")


if __name__ == "__main__":
    main()
