#!/usr/bin/env python3
"""Compute operational metrics using conformal p-values at various alpha budgets."""
import numpy as np
import csv
from sklearn.metrics import roc_auc_score

def load_steps(path):
    labels, scores, conf_pvals = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row['label']))
            scores.append(float(row['score']))
            conf_pvals.append(float(row['conformal_pval']))
    return np.array(labels), np.array(scores), np.array(conf_pvals)

def metrics_at_alpha(labels, conf_pvals, alpha):
    """Raise alarm when conf_pval < alpha."""
    alarms = (conf_pvals < alpha).astype(int)
    n_normal = np.sum(labels == 0)
    n_attack = np.sum(labels == 1)
    
    tp = np.sum((alarms == 1) & (labels == 1))
    fp = np.sum((alarms == 1) & (labels == 0))
    fn = np.sum((alarms == 0) & (labels == 1))
    
    empirical_fpr = fp / n_normal if n_normal > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / n_attack if n_attack > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return empirical_fpr, precision, recall, f1, tp, fp, fn

# OpTC Tuned
print("=== OpTC (Tuned, seed 42) ===")
labels, scores, cpvals = load_steps('results/final/optc_tuned_steps.csv')
print(f"Total: {len(labels)} windows, {np.sum(labels==1)} attacks, {np.sum(labels==0)} normal")
print(f"Score AUC: {roc_auc_score(labels, scores):.4f}")
print(f"Conformal p-value AUC: {roc_auc_score(labels, -cpvals):.4f}")
print(f"Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
print(f"Normal score range: [{np.min(scores[labels==0]):.4f}, {np.max(scores[labels==0]):.4f}]")
print(f"Attack score range: [{np.min(scores[labels==1]):.4f}, {np.max(scores[labels==1]):.4f}]")
print(f"Conformal p-value range (normal): [{np.min(cpvals[labels==0]):.4f}, {np.max(cpvals[labels==0]):.4f}]")
print(f"Conformal p-value range (attack): [{np.min(cpvals[labels==1]):.4f}, {np.max(cpvals[labels==1]):.4f}]")
print()

# The conformal p-value IS the operational mechanism
for alpha in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    fpr, prec, rec, f1, tp, fp, fn = metrics_at_alpha(labels, cpvals, alpha)
    print(f"α={alpha:.2f}: FPR={fpr*100:.2f}%, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f} (TP={tp}, FP={fp}, FN={fn})")

# 10-seed aggregate
print("\n=== OpTC 10-seed aggregate FPR stability ===")
import json
with open('logs/optc_10seed_aggregate.json') as f:
    agg = json.load(f)
fprs = agg['zc_fprs']
aucs = agg['zc_aucs']
print(f"AUC per seed: {[round(a,4) for a in aucs]}")
print(f"FPR per seed: {[round(f*100,2) for f in fprs]}")
print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"FPR: {np.mean(fprs)*100:.2f}% ± {np.std(fprs)*100:.2f}%")

# OpTC Fixed
print("\n=== OpTC (Default, seed 42) ===")
labels2, scores2, cpvals2 = load_steps('results/final/optc_run_fixed_steps.csv')
print(f"Score AUC: {roc_auc_score(labels2, scores2):.4f}")
for alpha in [0.01, 0.03, 0.05, 0.08, 0.10]:
    fpr, prec, rec, f1, tp, fp, fn = metrics_at_alpha(labels2, cpvals2, alpha)
    print(f"α={alpha:.2f}: FPR={fpr*100:.2f}%, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f} (TP={tp}, FP={fp}, FN={fn})")
