#!/usr/bin/env python3
"""
Drift FPR Head-to-Head Comparison
===================================
Runs a TC3 concept-drift scenario where the activity rate of five normal edge
types doubles at test step DRIFT_STEP, simulating a benign regime shift
(e.g., a patch deployment or workload change).

Shows:
  - ZeroCausal: FPR spikes briefly, then recovers to the 5% budget via
    online change-point detection + SCM refit.
  - Isolation Forest / Autoencoder: static models trained on the pre-drift
    distribution; their FPR stays elevated because they have no adaptation.

Output:
  results/drift_fpr_comparison.json
  plots/drift_fpr_comparison.png
"""

import sys
import os
import warnings
import json
import pickle
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

warnings.filterwarnings("ignore")

from zerocausal_core import (
    AdaptiveWindowDetector,
    CausalRegressionModel,
    HybridAnomalyScorer,
    ConformalCalibrator,
)
from baselines import train_ae_and_get_scores


# ---------------------------------------------------------------------------
# TC3 data generator (self-contained)
# ---------------------------------------------------------------------------

def generate_tc3_data(num_windows=800, noise_level=0.1, seed=42):
    np.random.seed(seed)
    t = pd.date_range("2026-01-01", periods=num_windows, freq="1s")

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

    data = {}
    for edge in baseline_edges:
        lam = 3.0 if ("svchost" in edge or "chrome" in edge) else 0.5
        data[edge] = np.random.poisson(lam=lam, size=num_windows).astype(float)

    data["PROCESS:spoolsv.exe -> READS_FILE -> FILE:printers.db"] += (
        data["PROCESS:svchost.exe -> CONNECTS_TO -> FLOW:8.8.8.8:53"] > 2
    ).astype(float)

    distractor_edges = [
        "PROCESS:explorer.exe -> READS_FILE -> FILE:desktop.ini",
        "PROCESS:chrome.exe -> CONNECTS_TO -> FLOW:8.8.4.4:53",
        "PROCESS:svchost.exe -> WRITE -> REGISTRY:HKLM",
        "PROCESS:taskmgr.exe -> OPEN -> PROCESS:explorer.exe",
        "PROCESS:cmd.exe -> SPAWNS_PROCESS -> PROCESS:conhost.exe",
    ]
    if noise_level > 0:
        for edge in distractor_edges:
            data[edge] = np.random.poisson(lam=noise_level * 3.0, size=num_windows).astype(float)

    attack_edges = [
        "PROCESS:nginx.exe -> SPAWNS_PROCESS -> PROCESS:bash.exe",
        "PROCESS:bash.exe -> WRITES_FILE -> FILE:malicious.elf",
        "PROCESS:malicious.elf -> MODIFY -> FILE:passwd",
    ]
    for edge in attack_edges:
        data[edge] = np.random.poisson(lam=0.01, size=num_windows).astype(float)

    df = pd.DataFrame(data, index=t)
    if noise_level > 0:
        for col in df.columns:
            df[col] = np.clip(
                df[col] + np.random.normal(0, noise_level * 2.0, num_windows), 0, None
            )

    labels = np.zeros(num_windows)
    test_start = int(num_windows * 0.6)
    attack_indices = np.random.choice(
        np.arange(test_start + 10, num_windows - 10), 50, replace=False
    )
    for idx in attack_indices:
        labels[idx : idx + 3] = 1

    return df, labels, attack_edges, attack_indices


# ---------------------------------------------------------------------------
# Rolling FPR on normal-only windows
# ---------------------------------------------------------------------------

def rolling_fpr(alarms, labels, window=50):
    """Returns per-step estimate of FPR using a sliding window over normal windows."""
    n = len(alarms)
    result = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        w_alarms = alarms[start : i + 1]
        w_labels = labels[start : i + 1]
        normal_mask = w_labels == 0
        if normal_mask.sum() > 0:
            result[i] = w_alarms[normal_mask].mean()
    return result


# ---------------------------------------------------------------------------
# ZeroCausal with drift adaptation
# ---------------------------------------------------------------------------

def run_zerocausal_drift(
    train_df, test_df, test_labels, attack_edges, attack_indices,
    var_names, p_matrix, split_idx,
    drift_step, tau_max=2, target_fpr=0.05, seed=42,
):
    np.random.seed(seed)

    # Inject attack bursts into test at pre-selected indices
    test_df = test_df.copy()
    for edge in attack_edges:
        if edge not in test_df.columns:
            test_df[edge] = 0.0
    for idx in attack_indices:
        test_idx = idx - split_idx
        if 0 <= test_idx < len(test_df) - 2:
            burst = max(np.random.poisson(5), 3)
            for k, edge in enumerate(attack_edges):
                t_idx = test_idx + k
                if t_idx < len(test_df) and edge in test_df.columns:
                    test_df.iloc[t_idx, test_df.columns.get_loc(edge)] += burst

    calib_split = int(len(train_df) * 0.7)
    train_proper = train_df.iloc[:calib_split].copy()
    train_calib = train_df.iloc[calib_split:].copy()

    causal_model = CausalRegressionModel(
        p_matrix, var_names, tau_max=tau_max, alpha=0.01, regressor_type="linear"
    )
    causal_model.fit(train_proper, std_floor=1.0)

    scorer = HybridAnomalyScorer(d=len(var_names), w=0.5, floor=1.0)
    calib_scores = []
    for i in range(tau_max, len(train_calib)):
        idx = calib_split + i
        res, pv = causal_model.predict_and_residual(train_df, idx)
        scorer.calibrate(res, causal_model.residual_stds)
        calib_scores.append(scorer.score(pv, res, causal_model.residual_stds))

    calibrator = ConformalCalibrator(target_fpr=target_fpr, lr=0.05, alpha_init=0.05)
    calibrator.calibrate(calib_scores)

    detector = AdaptiveWindowDetector(
        num_features=len(var_names), short_window=10, long_window=50, threshold=4.0
    )

    test_history = pd.concat([train_df.iloc[-1:], test_df])
    history_cols = list(test_history.columns)
    history_arr = test_history.to_numpy().copy()
    col_to_idx = {c: ii for ii, c in enumerate(history_cols)}
    var_indices = [col_to_idx[v] for v in var_names if v in col_to_idx]

    scores = np.zeros(len(test_df))
    alarms = np.zeros(len(test_df))
    change_steps = []

    for i in range(tau_max, len(test_df) + 1):
        # Apply drift: double rate of first 5 baseline features at drift_step
        if i >= drift_step:
            for vidx in var_indices[:5]:
                history_arr[i, vidx] = history_arr[i, vidx] * 2.0

        actual_row = history_arr[i]
        feat_subset = actual_row[var_indices]
        change_detected = detector.update(feat_subset)

        if change_detected:
            change_steps.append(i - 1)
            refit_len = max(50, 50)
            start_idx = max(0, i - refit_len + 1)
            refit_arr = history_arr[start_idx : i + 1]
            refit_df = pd.DataFrame(refit_arr, columns=history_cols)[var_names].copy()
            causal_model.fit(refit_df, std_floor=1.0)

            new_calib = []
            hist_df_tmp = pd.DataFrame(history_arr, columns=history_cols)
            for k in range(1, len(refit_df)):
                hist_idx = start_idx + k
                res_k, pv_k = causal_model.predict_and_residual(hist_df_tmp, hist_idx)
                new_calib.append(scorer.score(pv_k, res_k, causal_model.residual_stds))
            calibrator.calibrate(new_calib)

        hist_df = pd.DataFrame(history_arr, columns=history_cols)
        known_res, known_pv = causal_model.predict_and_residual(hist_df, i)
        res, pv = known_res.copy(), known_pv.copy()

        active = np.where(actual_row > 0)[0]
        novel_edges = []
        for eidx in active:
            edge = history_cols[eidx]
            if edge not in var_names:
                res[edge] = float(actual_row[eidx])
                pv[edge] = 1e-15
                novel_edges.append(edge)

        res_stds = causal_model.residual_stds.copy()
        for edge in novel_edges:
            res_stds[edge] = 0.1

        score = scorer.score(pv, res, res_stds)
        scores[i - 1] = score

        conf_pval = calibrator.compute_conformal_pvalue(score)
        alarm = 1.0 if conf_pval < calibrator.alpha else 0.0
        alarms[i - 1] = alarm

        calibrator.update_threshold(alarm)
        if alarm == 0.0:
            calibrator.update_calibration(score, max_size=245)

    return alarms, scores, change_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DRIFT_STEP = 200     # test-step index where drift begins (0-indexed, in test array)
NUM_WINDOWS = 800
NOISE_LEVEL = 0.1
SEED = 42
TAU_MAX = 2
FPR_BUDGET = 0.05
ROLLING_WINDOW = 50


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("  Drift FPR Head-to-Head Comparison")
    print(f"  Drift onset: test step {DRIFT_STEP}")
    print(f"  Rolling FPR window: {ROLLING_WINDOW} steps")
    print("=" * 60)

    print(f"\nGenerating TC3 data ({NUM_WINDOWS} windows, seed={SEED})...")
    ts_df, labels, attack_edges, attack_indices = generate_tc3_data(
        num_windows=NUM_WINDOWS, noise_level=NOISE_LEVEL, seed=SEED
    )
    split_idx = int(NUM_WINDOWS * 0.6)
    train_df = ts_df.iloc[:split_idx].copy()
    test_df_clean = ts_df.iloc[split_idx:].copy()
    test_labels = labels[split_idx:]
    n_test = len(test_df_clean)

    print(f"  Train: {split_idx}  Test: {n_test}  Attacks in test: {int(test_labels.sum())}")

    # --- PCMCI on clean training data ---
    cache_path = f"logs/pcmci_cache_drift_fpr_seed{SEED}.pkl"
    print(f"\nRunning PCMCI (tau_max={TAU_MAX})...")
    t0 = time.time()
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        p_matrix, var_names = d["p_matrix"], d["var_names"]
        print(f"  Loaded from cache ({len(var_names)} variables)")
    else:
        var_names = train_df.columns.tolist()
        data_arr = train_df.astype(float).values
        df_pp = pp.DataFrame(data_arr, datatime=np.arange(len(train_df)), var_names=var_names)
        cit = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=df_pp, cond_ind_test=cit, verbosity=0)
        results = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=0.01)
        p_matrix = results["p_matrix"]
        with open(cache_path, "wb") as f:
            pickle.dump({"p_matrix": p_matrix, "var_names": var_names}, f)
        print(f"  PCMCI done in {time.time()-t0:.1f}s")

    # =========================================================
    # ZeroCausal with drift adaptation
    # =========================================================
    print(f"\nRunning ZeroCausal (with change-point detection + online refit)...")
    t0 = time.time()
    zc_alarms, zc_scores, zc_change_steps = run_zerocausal_drift(
        train_df.copy(),
        test_df_clean.copy(),
        test_labels,
        attack_edges,
        attack_indices,
        var_names,
        p_matrix,
        split_idx,
        drift_step=DRIFT_STEP,
        tau_max=TAU_MAX,
        target_fpr=FPR_BUDGET,
        seed=SEED,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Change-points detected at test steps: {zc_change_steps}")

    # =========================================================
    # Build drifted test array for static baselines
    # The drift transformation mirrors what ZeroCausal's loop applies
    # =========================================================
    all_cols = sorted(set(train_df.columns) | set(test_df_clean.columns))
    train_aligned = train_df.reindex(columns=all_cols, fill_value=0.0).values.astype(float)
    test_aligned = test_df_clean.reindex(columns=all_cols, fill_value=0.0).values.astype(float).copy()

    # Apply same drift: double first 5 var_names in test_aligned at drift_step
    col_to_all_idx = {c: ii for ii, c in enumerate(all_cols)}
    drift_col_indices = [col_to_all_idx[v] for v in var_names[:5] if v in col_to_all_idx]
    test_aligned[DRIFT_STEP:, drift_col_indices] *= 2.0

    # =========================================================
    # Isolation Forest — static, trained on pre-drift distribution
    # =========================================================
    print(f"\nRunning Isolation Forest (static baseline)...")
    iforest = IsolationForest(contamination=FPR_BUDGET, random_state=SEED)
    iforest.fit(train_aligned)

    # Set threshold from pre-drift test scores so pre-drift FPR = 5%
    test_if_scores = -iforest.score_samples(test_aligned)
    pre_drift_test_if_scores = test_if_scores[:DRIFT_STEP][test_labels[:DRIFT_STEP] == 0]
    if_threshold = np.percentile(pre_drift_test_if_scores, 95)
    if_alarms = (test_if_scores > if_threshold).astype(float)

    pre_if_fpr = np.mean(if_alarms[:DRIFT_STEP][test_labels[:DRIFT_STEP] == 0])
    post_if_fpr = np.mean(if_alarms[DRIFT_STEP:][test_labels[DRIFT_STEP:] == 0])
    print(f"  Pre-drift FPR: {pre_if_fpr*100:.1f}%  Post-drift FPR: {post_if_fpr*100:.1f}%")

    # =========================================================
    # Autoencoder — static, trained on pre-drift distribution
    # =========================================================
    ae_alarms = None
    ae_rolling = None
    print(f"\nRunning Autoencoder (static baseline)...")
    try:
        train_ae_scores, test_ae_scores = train_ae_and_get_scores(
            train_aligned, test_aligned, epochs=50, seed=SEED
        )
        pre_drift_test_ae_scores = test_ae_scores[:DRIFT_STEP][test_labels[:DRIFT_STEP] == 0]
        ae_threshold = np.percentile(pre_drift_test_ae_scores, 95)
        ae_alarms = (test_ae_scores > ae_threshold).astype(float)

        pre_ae_fpr = np.mean(ae_alarms[:DRIFT_STEP][test_labels[:DRIFT_STEP] == 0])
        post_ae_fpr = np.mean(ae_alarms[DRIFT_STEP:][test_labels[DRIFT_STEP:] == 0])
        print(f"  Pre-drift FPR: {pre_ae_fpr*100:.1f}%  Post-drift FPR: {post_ae_fpr*100:.1f}%")
    except Exception as exc:
        print(f"  Autoencoder skipped: {exc}")

    # =========================================================
    # Rolling FPR curves
    # =========================================================
    zc_rolling = rolling_fpr(zc_alarms, test_labels, window=ROLLING_WINDOW)
    if_rolling = rolling_fpr(if_alarms, test_labels, window=ROLLING_WINDOW)
    if ae_alarms is not None:
        ae_rolling = rolling_fpr(ae_alarms, test_labels, window=ROLLING_WINDOW)

    # Summary statistics
    pre_zc = float(np.nanmean(zc_rolling[:DRIFT_STEP])) * 100
    post_zc = float(np.nanmean(zc_rolling[DRIFT_STEP:])) * 100
    pre_if = float(np.nanmean(if_rolling[:DRIFT_STEP])) * 100
    post_if = float(np.nanmean(if_rolling[DRIFT_STEP:])) * 100

    print(f"\n{'='*60}")
    print("  DRIFT FPR SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':>20} | Pre-drift FPR | Post-drift FPR")
    print(f"  {'-'*50}")
    print(f"  {'ZeroCausal':>20} | {pre_zc:>12.1f}% | {post_zc:>12.1f}%")
    print(f"  {'Isolation Forest':>20} | {pre_if:>12.1f}% | {post_if:>12.1f}%")
    if ae_alarms is not None:
        pre_ae2 = float(np.nanmean(ae_rolling[:DRIFT_STEP])) * 100
        post_ae2 = float(np.nanmean(ae_rolling[DRIFT_STEP:])) * 100
        print(f"  {'Autoencoder':>20} | {pre_ae2:>12.1f}% | {post_ae2:>12.1f}%")
    print(f"  {'FPR budget':>20} | {FPR_BUDGET*100:>12.1f}% | {FPR_BUDGET*100:>12.1f}%")

    # =========================================================
    # Plot
    # =========================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = np.arange(n_test)

    ax.plot(
        steps, zc_rolling * 100,
        "-", color="#2166ac", linewidth=2.5,
        label="ZeroCausal (Ours, online refit)",
        zorder=4,
    )
    ax.plot(
        steps, if_rolling * 100,
        "--", color="#d73027", linewidth=2.0,
        label="Isolation Forest (static)",
        zorder=3,
    )
    if ae_rolling is not None:
        ax.plot(
            steps, ae_rolling * 100,
            ":", color="#f46d43", linewidth=2.0,
            label="Autoencoder (static)",
            zorder=3,
        )

    ax.axvline(
        x=DRIFT_STEP, color="#4dac26", linestyle="--", linewidth=1.8,
        label=f"Drift onset (step {DRIFT_STEP})", zorder=5,
    )
    ax.axhline(
        y=FPR_BUDGET * 100, color="gray", linestyle=":", linewidth=1.3,
        label=f"{FPR_BUDGET*100:.0f}% FPR budget", zorder=2,
    )
    # Mark detected change-points
    for cps in zc_change_steps:
        ax.axvline(x=cps, color="#2166ac", linestyle=":", linewidth=1.0, alpha=0.5, zorder=4)
    if zc_change_steps:
        ax.axvline(
            x=zc_change_steps[0], color="#2166ac", linestyle=":", linewidth=1.0, alpha=0.5,
            label="ZeroCausal refit triggers",
        )

    ax.set_xlabel("Test Step", fontsize=13)
    ax.set_ylabel(f"Rolling FPR (%) — window = {ROLLING_WINDOW} steps", fontsize=13)
    ax.set_title(
        "False-Alarm Rate Under Concept Drift: ZeroCausal vs. Static Baselines\n"
        f"(TC3 Simulation — Benign Activity Doubles at Step {DRIFT_STEP})",
        fontsize=12,
    )
    ax.set_xlim(0, n_test - 1)
    ymax = max(
        np.nanmax(if_rolling) * 100 * 1.15 if len(if_rolling) else 30,
        30,
    )
    ax.set_ylim(0, ymax)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "plots/drift_fpr_comparison.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {plot_path}")

    # Copy to results/final
    os.makedirs("results/final", exist_ok=True)
    import shutil
    shutil.copy(plot_path, "results/final/drift_fpr_comparison.png")

    # Save results
    result_data = {
        "drift_step": DRIFT_STEP,
        "rolling_window": ROLLING_WINDOW,
        "n_test": n_test,
        "fpr_budget": FPR_BUDGET,
        "pre_drift_zc_fpr_pct": round(pre_zc, 2),
        "post_drift_zc_fpr_pct": round(post_zc, 2),
        "pre_drift_if_fpr_pct": round(pre_if, 2),
        "post_drift_if_fpr_pct": round(post_if, 2),
        "zc_change_steps": zc_change_steps,
        "zc_rolling_fpr": [None if np.isnan(v) else round(float(v) * 100, 3) for v in zc_rolling],
        "if_rolling_fpr": [None if np.isnan(v) else round(float(v) * 100, 3) for v in if_rolling],
        "ae_rolling_fpr": (
            [None if np.isnan(v) else round(float(v) * 100, 3) for v in ae_rolling]
            if ae_rolling is not None else None
        ),
    }
    out_path = "results/drift_fpr_comparison.json"
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
