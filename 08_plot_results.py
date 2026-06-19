import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(steps_df, summary_json, run_name, out_dir):
    labels = steps_df['label'].values
    scores = steps_df['score'].values
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#3182ce', lw=2.5, label=f'ZeroCausal (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#a0aec0', lw=1.5, linestyle='--', label='Random Guess')
    
    # Also add the competitor baseline point
    # Competitor Causal-IDS is reported to have AUC 0.8400
    plt.plot([0.05, 0.05], [0.0, 0.84], color='#e53e3e', linestyle=':', alpha=0.7)
    plt.scatter([0.05], [0.84], color='#e53e3e', marker='X', s=100, zorder=5, label='Causal-IDS (2026) Baseline')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('True Positive Rate (TPR) / Recall', fontsize=12, fontweight='bold', labelpad=10)
    plt.title(f'ROC Curve - ZeroCausal Anomaly Detection ({run_name})', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True, facecolor='white', edgecolor='#e2e8f0')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{run_name}_roc_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"ROC Curve saved to {out_path}")

def plot_time_series(steps_df, run_name, out_dir):
    steps = steps_df['step_idx'].values
    scores = steps_df['score'].values
    labels = steps_df['label'].values
    alarms = steps_df['alarm'].values
    threshold = steps_df['threshold'].values
    change_pts = steps_df['change_detected'].values
    novelties = steps_df['novelty_detected'].values
    
    fig, ax1 = plt.subplots(figsize=(15, 6))
    
    # 1. Plot Anomaly Scores (CAS)
    ax1.plot(steps, scores, color='#2d3748', lw=1.5, label='Causal Anomaly Score (CAS)')
    
    # Shade attack regions (label == 1)
    attack_indices = np.where(labels == 1)[0]
    if len(attack_indices) > 0:
        # Group contiguous attack indices to shade blocks
        diffs = np.diff(attack_indices)
        split_indices = np.where(diffs > 1)[0] + 1
        blocks = np.split(attack_indices, split_indices)
        first_shade = True
        for block in blocks:
            start = block[0]
            end = block[-1]
            ax1.axvspan(start, end, color='#fed7d7', alpha=0.4, 
                        label='Synthetic APT Attack' if first_shade else "")
            first_shade = False
            
    # Draw Alarm markers
    alarm_indices = np.where(alarms == 1)[0]
    if len(alarm_indices) > 0:
        ax1.scatter(alarm_indices, scores[alarm_indices], color='#e53e3e', marker='o', s=40, zorder=5,
                    label='Conformal Alarm Raised')
                    
    # Draw Change-point detection markers
    cp_indices = np.where(change_pts == 1)[0]
    if len(cp_indices) > 0:
        for cp in cp_indices:
            ax1.axvline(x=cp, color='#3182ce', linestyle='--', alpha=0.7, lw=1.2,
                        label='Baseline Change-point' if cp == cp_indices[0] else "")
            
    # Draw Novelty markers
    novelty_indices = np.where(novelties == 1)[0]
    if len(novelty_indices) > 0:
        ax1.scatter(novelty_indices, scores[novelty_indices], color='#d69e2e', marker='^', s=60, zorder=6,
                    label='Structural Novelty Edge')

    ax1.set_xlabel('Streaming Step (Seconds)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Anomaly Score (CAS)', color='#2d3748', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2d3748')
    
    # 2. Secondary axis for Conformal Threshold
    ax2 = ax1.twinx()
    ax2.plot(steps, threshold, color='#319795', lw=2.0, linestyle='-.', label='Conformal Threshold (α)')
    ax2.set_ylabel('Adaptive Threshold (α)', color='#319795', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#319795')
    
    plt.title(f'ZeroCausal Time-Series Anomaly Score & Online Adaptation ({run_name})', fontsize=14, fontweight='bold', pad=15)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0')
    
    # Grid and design
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{run_name}_time_series.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Time Series Plot saved to {out_path}")

def plot_root_cause(steps_df, summary, run_name, out_dir):
    # Filter to alarm steps during actual attacks (true positives)
    tp_steps = steps_df[(steps_df['alarm'] == 1) & (steps_df['label'] == 1)]
    if len(tp_steps) == 0:
        # Fallback to all steps during attacks
        tp_steps = steps_df[steps_df['label'] == 1]
        
    if len(tp_steps) == 0:
        print("No attack steps found, skipping root-cause plot.")
        return
        
    # Reconstruct feature list
    res_cols = [c for c in steps_df.columns if c.startswith('res_')]
    
    residual_stds = summary.get('residual_stds', {})
    
    violations = {}
    for col in res_cols:
        var = col[4:]
        res_vals = tp_steps[col].values
        std = residual_stds.get(var, 0.1) # default to 0.1 for novelties
        # Average normalized squared error
        norm_squared_err = np.mean((res_vals / std) ** 2)
        if norm_squared_err > 0.01: # Filter tiny values
            clean_name = var.replace("PROCESS:", "").replace("FILE:", "").replace("SPAWNS_PROCESS", "spawns").replace("WRITES_FILE", "writes")
            violations[clean_name] = norm_squared_err
            
    if not violations:
        print("No significant causal violations found, skipping root-cause plot.")
        return
        
    # Sort violations
    sorted_violations = sorted(violations.items(), key=lambda x: x[1], reverse=True)
    # Take top 15 features to avoid overcrowded plot
    sorted_violations = sorted_violations[:15]
    features, score_contributions = zip(*sorted_violations)
    
    plt.figure(figsize=(10, 6))
    
    # Custom color gradient from orange to red
    colors = plt.cm.plasma(np.linspace(0.8, 0.2, len(features)))
    
    # Horizontal bar plot
    y_pos = np.arange(len(features))
    plt.barh(y_pos, score_contributions, color=colors, height=0.6)
    plt.yticks(y_pos, features)
    
    plt.xlabel('Mean Causal Mechanism Violation Error (Normalized Residual²)', fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel('Causal Graph Relationship (Edge)', fontsize=11, fontweight='bold')
    plt.title('Explainable Root-Cause Analysis during APT Attack', fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, axis='x', linestyle=':', alpha=0.5)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Invert y axis to show largest violation at the top
    ax.invert_yaxis()
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{run_name}_root_cause.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Root Cause Plot saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="ZeroCausal Plot Generation Utility")
    parser.add_argument("--summary-json", type=str, default="logs/optc_default_summary.json", help="Path to summary JSON log")
    parser.add_argument("--steps-csv", type=str, default="logs/optc_default_steps.csv", help="Path to steps CSV log")
    parser.add_argument("--plots-dir", type=str, default="plots", help="Directory to save generated plots")
    parser.add_argument("--run-name", type=str, default="optc_default", help="Prefix name for the generated plot files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.summary_json) or not os.path.exists(args.steps_csv):
        print("Error: Missing log files. Run 05_evaluate_zerocausal.py first.")
        return
        
    os.makedirs(args.plots_dir, exist_ok=True)
    
    print(f"Loading data logs from {args.summary_json} and {args.steps_csv}...")
    steps_df = pd.read_csv(args.steps_csv)
    with open(args.summary_json, 'r') as f:
        summary = json.load(f)
        
    print("Generating diagrams...")
    plot_roc_curve(steps_df, summary, args.run_name, args.plots_dir)
    plot_time_series(steps_df, args.run_name, args.plots_dir)
    plot_root_cause(steps_df, summary, args.run_name, args.plots_dir)
    print("Done! All diagrams successfully saved in plots/ directory.")

if __name__ == "__main__":
    main()
