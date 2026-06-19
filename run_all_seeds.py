"""
run_all_seeds.py
================
Runs ZeroCausal evaluation on OpTC over 10 random seeds (42–51) to compute
mean ± std of AUC, Empirical FPR, and Isolation Forest baseline AUC.

The PCMCI causal graph is loaded from a pre-computed summary JSON to avoid
re-running the expensive causal discovery step 10 times.
"""
import os
import json
import numpy as np
import importlib
from argparse import Namespace

# ── Import evaluation functions from the digit-prefixed module ────────────────
import sys
sys.path.append(os.getcwd())
_eval_mod = importlib.import_module("05_evaluate_zerocausal")
load_and_prep_data = _eval_mod.load_and_prep_data
evaluate          = _eval_mod.evaluate

# ── Helper: reconstruct p_matrix from saved causal graph edges ────────────────
def load_pmatrix_from_summary(summary_path, var_names, tau_max=1):
    """
    Reconstruct a p_matrix (shape: n_vars × n_vars × (tau_max+1)) from
    the 'causal_graph_edges' list stored in an existing summary JSON.

    Edges that are *not* listed are assigned p=1.0 (no significant edge).
    Listed edges are assigned p=0.0 (significant at any alpha ≤ their original p-value).
    We store the exact p-value so callers can use any pcmci_alpha threshold.
    """
    with open(summary_path) as f:
        summary = json.load(f)

    n = len(var_names)
    p_matrix = np.ones((n, n, tau_max + 1))     # default: not significant

    var_to_idx = {v: i for i, v in enumerate(var_names)}

    for edge in summary.get("causal_graph_edges", []):
        src  = edge["source"]
        tgt  = edge["target"]
        tau  = edge["tau"]
        pval = edge["p_value"]
        if src in var_to_idx and tgt in var_to_idx and 0 < tau <= tau_max:
            p_matrix[var_to_idx[src], var_to_idx[tgt], tau] = pval

    return p_matrix


def main():
    SEEDS      = list(range(42, 52))   # 10 seeds
    LOG_DIR    = "logs"
    SUMMARY    = os.path.join(LOG_DIR, "optc_run_summary.json")   # pre-computed graph
    DATA_CSV   = "optc_edges.csv"

    print("=" * 60)
    print("ZeroCausal — 10-Seed Statistical Evaluation on OpTC")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading OpTC data …")
    ts_df = load_and_prep_data(DATA_CSV)

    # ── Recover var_names from the saved summary ───────────────────────────────
    with open(SUMMARY) as f:
        ref_summary = json.load(f)
    var_names = list(ref_summary["residual_stds"].keys())
    print(f"  Loaded {len(var_names)} causal variables from cached summary.")

    # ── Reconstruct p_matrix (skip expensive PCMCI re-run) ────────────────────
    print("  Reconstructing p_matrix from saved causal graph edges …")
    p_matrix = load_pmatrix_from_summary(SUMMARY, var_names, tau_max=1)

    # ── Base args (shared across all seeds) ───────────────────────────────────
    base_args = Namespace(
        pcmci_alpha          = 0.01,
        std_floor            = 1.0,
        a_p                  = 0.1,
        b_p                  = 5.0,
        a_r                  = 5.0,
        b_r                  = 0.1,
        target_fpr           = 0.05,
        conformal_lr         = 0.05,
        conformal_alpha_init = 0.05,
        detector_short       = 10,
        detector_long        = 50,
        detector_threshold   = 4.0,
        log_dir              = LOG_DIR,
        run_name             = "optc_run",
        seed                 = 42,
    )

    # ── Loop over seeds ────────────────────────────────────────────────────────
    zc_aucs      = []
    zc_fprs      = []
    iforest_aucs = []

    for seed in SEEDS:
        print(f"\n{'─'*50}")
        print(f"  Seed {seed} …")
        base_args.seed     = seed
        base_args.run_name = f"optc_seed_{seed}"

        evaluate(ts_df, p_matrix, var_names, base_args)

        result_path = os.path.join(LOG_DIR, f"optc_seed_{seed}_summary.json")
        with open(result_path) as f:
            res = json.load(f)

        auc      = res["metrics"]["auc"]
        efpr     = res["metrics"]["empirical_conformal_fpr"]
        if_auc   = res["metrics"].get("iforest_auc", float("nan"))

        zc_aucs.append(auc)
        zc_fprs.append(efpr)
        iforest_aucs.append(if_auc)
        print(f"  → AUC={auc:.4f}  eFPR={efpr*100:.2f}%  IF-AUC={if_auc:.4f}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("🏆  10-SEED AGGREGATED METRICS — OpTC")
    print(f"{'='*60}")
    print(f"ZeroCausal   AUC : {np.mean(zc_aucs):.4f} ± {np.std(zc_aucs):.4f}")
    print(f"ZeroCausal   eFPR: {np.mean(zc_fprs)*100:.2f}% ± {np.std(zc_fprs)*100:.2f}%  (target 5.00%)")
    print(f"Isol. Forest AUC : {np.mean(iforest_aucs):.4f} ± {np.std(iforest_aucs):.4f}")
    print(f"{'='*60}")

    # ── LaTeX row ─────────────────────────────────────────────────────────────
    print("\nLaTeX Table Row:")
    print(
        f"ZeroCausal & "
        f"{np.mean(zc_aucs):.4f} $\\pm$ {np.std(zc_aucs):.4f} & "
        f"{np.mean(zc_fprs)*100:.2f}\\% $\\pm$ {np.std(zc_fprs)*100:.2f}\\% & "
        f"{np.mean(iforest_aucs):.4f} $\\pm$ {np.std(iforest_aucs):.4f} \\\\"
    )

    # ── Save aggregate JSON ────────────────────────────────────────────────────
    aggregate = {
        "seeds": SEEDS,
        "zc_aucs":       zc_aucs,
        "zc_fprs":       zc_fprs,
        "iforest_aucs":  iforest_aucs,
        "mean_zc_auc":   float(np.mean(zc_aucs)),
        "std_zc_auc":    float(np.std(zc_aucs)),
        "mean_zc_fpr":   float(np.mean(zc_fprs)),
        "std_zc_fpr":    float(np.std(zc_fprs)),
        "mean_if_auc":   float(np.mean(iforest_aucs)),
        "std_if_auc":    float(np.std(iforest_aucs)),
    }
    agg_path = os.path.join(LOG_DIR, "optc_10seed_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate results saved to {agg_path}")


if __name__ == "__main__":
    main()
