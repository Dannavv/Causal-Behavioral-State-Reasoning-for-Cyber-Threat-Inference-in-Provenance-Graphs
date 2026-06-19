#!/usr/bin/env python3
"""
Contamination Sweep Experiment (Multi-Seed)
===========================================
Tests ZeroCausal and static baselines as the TRAINING split is contaminated
with increasing rates of injected attack events (0%, 1%, 5%, 10%, 20%).

Central claim being tested: causal-structure detection degrades much more
slowly under poisoned training than distribution-based methods, because
conditional-independence relationships between normal features are more robust
to sparse contamination than reconstruction error or density estimates.

Output:
  results/contamination_sweep.json
  plots/contamination_sweep.png
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
# TC3 data generator (self-contained so this script has no numbered imports)
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

    # One causal dependency: spoolsv traffic caused by svchost DNS bursts
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
# PCMCI wrapper with per-contamination-rate caching
# ---------------------------------------------------------------------------

def run_pcmci(train_df, cache_path, tau_max=2, alpha=0.01):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        return d["p_matrix"], d["var_names"]

    var_names = train_df.columns.tolist()
    data = train_df.astype(float).values
    df_pp = pp.DataFrame(data, datatime=np.arange(len(train_df)), var_names=var_names)
    cit = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=df_pp, cond_ind_test=cit, verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)

    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"p_matrix": results["p_matrix"], "var_names": var_names}, f)

    return results["p_matrix"], var_names


# ---------------------------------------------------------------------------
# Shared burst injector — used by both ZeroCausal and baselines
# ---------------------------------------------------------------------------

def inject_attack_bursts(test_df, attack_edges, attack_indices, split_idx, seed=42):
    """Return a copy of test_df with realistic APT bursts on attack_edges."""
    np.random.seed(seed)
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
    return test_df


# ---------------------------------------------------------------------------
# ZeroCausal single-run — expects pre-injected test_df
# ---------------------------------------------------------------------------

def run_zerocausal(
    train_df, test_df, test_labels,
    var_names, p_matrix, tau_max=2, target_fpr=0.05,
):

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

    test_history = pd.concat([train_df.iloc[-1:], test_df])
    history_cols = list(test_history.columns)
    history_arr = test_history.to_numpy().copy()
    col_to_idx = {c: ii for ii, c in enumerate(history_cols)}
    var_indices = [col_to_idx[v] for v in var_names if v in col_to_idx]

    scores = np.zeros(len(test_df))

    for i in range(tau_max, len(test_df) + 1):
        actual_row = history_arr[i]

        # Build a temporary DataFrame slice to satisfy predict_and_residual's interface
        # We pass the full test_history DataFrame
        known_res, known_pv = causal_model.predict_and_residual(test_history, i)

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
        calibrator.update_threshold(alarm)
        if alarm == 0.0:
            calibrator.update_calibration(score, max_size=245)

    try:
        return roc_auc_score(test_labels, scores)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Main contamination sweep
# ---------------------------------------------------------------------------

CONTAMINATION_RATES = [0.00, 0.01, 0.05, 0.10, 0.20]
NUM_WINDOWS = 800
NOISE_LEVEL = 0.1
TAU_MAX = 2
TARGET_FPR = 0.05


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    SEEDS = [42, 43, 44, 45, 46]

    print("=" * 60)
    print("  Contamination Sweep Experiment (Multi-Seed)")
    print(f"  Rates: {[f'{r*100:.0f}%' for r in CONTAMINATION_RATES]}")
    print(f"  Dataset: TC3-simulated ({NUM_WINDOWS} windows, seeds={SEEDS})")
    print("=" * 60)

    sweep_results = []

    for rate in CONTAMINATION_RATES:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  Contamination rate: {rate*100:.0f}%")
        print(f"{'='*60}")

        zc_aucs_for_rate = []
        if_aucs_for_rate = []
        ae_aucs_for_rate = []

        split_idx = int(NUM_WINDOWS * 0.6)
        n_contaminate = int(rate * split_idx)

        for seed in SEEDS:
            print(f"  --- Running Seed {seed} ---")
            # Generate TC3 data for this seed
            ts_df, labels, attack_edges, attack_indices = generate_tc3_data(
                num_windows=NUM_WINDOWS, noise_level=NOISE_LEVEL, seed=seed
            )
            test_labels = labels[split_idx:]
            test_df_base = ts_df.iloc[split_idx:].copy()
            train_df_base = ts_df.iloc[:split_idx].copy()

            # --- Build possibly-contaminated training data ---
            train_df = train_df_base.copy()

            if n_contaminate > 0:
                rng = np.random.default_rng(seed + 999)
                contam_idxs = rng.choice(split_idx - 3, n_contaminate, replace=False)
                for cidx in contam_idxs:
                    burst = int(max(rng.poisson(5), 3))
                    for k, edge in enumerate(attack_edges):
                        t_idx = int(cidx) + k
                        if t_idx < split_idx:
                            if edge in train_df.columns:
                                train_df.iloc[t_idx, train_df.columns.get_loc(edge)] += burst
                            else:
                                train_df[edge] = 0.0
                                train_df.iloc[t_idx, train_df.columns.get_loc(edge)] = float(burst)

            # --- PCMCI on (possibly contaminated) training data ---
            rate_str = f"{rate:.2f}".replace(".", "p")
            cache_path = f"logs/pcmci_cache_contamination_{rate_str}_seed{seed}.pkl"
            p_matrix, var_names = run_pcmci(train_df, cache_path, tau_max=TAU_MAX)

            # --- Inject attack bursts into test ---
            test_df_injected = inject_attack_bursts(
                test_df_base, attack_edges, attack_indices, split_idx, seed=seed
            )

            # --- ZeroCausal ---
            zc_auc = run_zerocausal(
                train_df.copy(),
                test_df_injected.copy(),
                test_labels,
                var_names,
                p_matrix,
                tau_max=TAU_MAX,
                target_fpr=TARGET_FPR,
            )
            zc_aucs_for_rate.append(zc_auc)

            # --- Baselines ---
            all_cols = sorted(set(train_df.columns) | set(test_df_injected.columns))
            train_aligned = train_df.reindex(columns=all_cols, fill_value=0.0).values.astype(float)
            test_aligned = test_df_injected.reindex(columns=all_cols, fill_value=0.0).values.astype(float)

            # IF
            iforest = IsolationForest(contamination=TARGET_FPR, random_state=seed)
            iforest.fit(train_aligned)
            if_scores = -iforest.score_samples(test_aligned)
            if_auc = roc_auc_score(test_labels, if_scores)
            if_aucs_for_rate.append(if_auc)

            # AE
            try:
                _, ae_test_scores = train_ae_and_get_scores(
                    train_aligned, test_aligned, epochs=50, seed=seed
                )
                ae_auc = roc_auc_score(test_labels, ae_test_scores)
                ae_aucs_for_rate.append(ae_auc)
            except Exception as exc:
                print(f"  Autoencoder skipped: {exc}")

        # Compute stats for this rate
        zc_mean, zc_std = np.mean(zc_aucs_for_rate), np.std(zc_aucs_for_rate)
        if_mean, if_std = np.mean(if_aucs_for_rate), np.std(if_aucs_for_rate)
        ae_mean, ae_std = np.mean(ae_aucs_for_rate), np.std(ae_aucs_for_rate) if ae_aucs_for_rate else (0.5, 0.0)

        elapsed = time.time() - t0
        entry = {
            "contamination_rate": rate,
            "contamination_pct": rate * 100,
            "n_contaminated_windows": n_contaminate,
            "zc_auc": float(zc_mean),
            "zc_auc_std": float(zc_std),
            "if_auc": float(if_mean),
            "if_auc_std": float(if_std),
            "ae_auc": float(ae_mean),
            "ae_auc_std": float(ae_std),
            "elapsed_s": round(elapsed, 2),
        }
        sweep_results.append(entry)
        print(f"  ZeroCausal AUC: {zc_mean:.4f} ± {zc_std:.4f}")
        print(f"  Isolation Forest AUC: {if_mean:.4f} ± {if_std:.4f}")
        print(f"  Autoencoder AUC: {ae_mean:.4f} ± {ae_std:.4f}")
        print(f"  Total time for contamination rate: {elapsed:.1f}s")

    # --- Save results ---
    out_path = "results/contamination_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Plot ---
    rates_pct = [r["contamination_pct"] for r in sweep_results]
    zc_aucs = [r["zc_auc"] for r in sweep_results]
    zc_stds = [r["zc_auc_std"] for r in sweep_results]
    if_aucs = [r["if_auc"] for r in sweep_results]
    if_stds = [r["if_auc_std"] for r in sweep_results]
    ae_aucs = [r["ae_auc"] for r in sweep_results]
    ae_stds = [r["ae_auc_std"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        rates_pct, zc_aucs, yerr=zc_stds,
        fmt="o-", color="#2166ac", linewidth=2.5, markersize=8, capsize=5,
        label="ZeroCausal (Ours)",
        zorder=3,
    )
    ax.errorbar(
        rates_pct, if_aucs, yerr=if_stds,
        fmt="s--", color="#d73027", linewidth=2.0, markersize=7, capsize=5,
        label="Isolation Forest",
        zorder=3,
    )
    ax.errorbar(
        rates_pct, ae_aucs, yerr=ae_stds,
        fmt="^--", color="#f46d43", linewidth=2.0, markersize=7, capsize=5,
        label="Autoencoder",
        zorder=3,
    )

    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1.5, label="Random baseline")
    ax.set_xlabel("Training-Set Contamination Rate (%)", fontsize=13)
    ax.set_ylabel("AUROC", fontsize=13)
    ax.set_title(
        "Detection Performance Under Poisoned Training\n(TC3 Simulation — 5-Seed Average with Std Dev)",
        fontsize=12,
    )
    ax.set_xticks(rates_pct)
    ax.set_xticklabels([f"{r:.0f}%" for r in rates_pct], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = "plots/contamination_sweep.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")

    # Copy to results/final
    os.makedirs("results/final", exist_ok=True)
    import shutil
    shutil.copy(plot_path, "results/final/contamination_sweep.png")

    # --- Console summary ---
    print(f"\n{'='*60}")
    print("  CONTAMINATION SWEEP SUMMARY (5-Seed Average)")
    print(f"{'='*60}")
    hdr = f"{'Rate':>8} | {'ZeroCausal':>18} | {'Iso Forest':>18} | {'Autoencoder':>18}"
    print(hdr)
    print("-" * len(hdr))
    for r in sweep_results:
        zc_str = f"{r['zc_auc']:.4f}±{r['zc_auc_std']:.4f}"
        if_str = f"{r['if_auc']:.4f}±{r['if_auc_std']:.4f}"
        ae_str = f"{r['ae_auc']:.4f}±{r['ae_auc_std']:.4f}"
        print(
            f"{r['contamination_pct']:>7.0f}% | "
            f"{zc_str:>18} | "
            f"{if_str:>18} | "
            f"{ae_str:>18}"
        )

    # Compute degradation from 0% to 20% for each method
    def deg(aucs):
        return aucs[0] - aucs[-1]

    print(f"\n  AUC drop (0% → 20% contamination):")
    print(f"    ZeroCausal:     {deg(zc_aucs):+.4f}")
    print(f"    Iso Forest:     {deg(if_aucs):+.4f}")
    print(f"    Autoencoder:    {deg(ae_aucs):+.4f}")


if __name__ == "__main__":
    main()
