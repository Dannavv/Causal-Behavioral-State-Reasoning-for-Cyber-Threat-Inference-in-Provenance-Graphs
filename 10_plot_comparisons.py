import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def main():
    runs = {
        "OpTC (Tuned, seed 42)": "results/final/optc_tuned_steps.csv",
        "DARPA TC3 (Default)": "logs/tc3_default_steps.csv",
        "NODLINK (Default)": "logs/nodlink_default_steps.csv",
        "StreamSpot (Real Provenance)": "logs/streamspot_default_steps.csv",
        "BETH (Real Sysdig)": "logs/beth_default_steps.csv"
    }
    
    colors = {
        "OpTC (Tuned, seed 42)": "#3182ce",          # Blue
        "DARPA TC3 (Default)": "#319795",   # Teal
        "NODLINK (Default)": "#d69e2e",       # Gold/Orange
        "StreamSpot (Real Provenance)": "#805ad5",  # Purple
        "BETH (Real Sysdig)": "#38a169"             # Green
    }
    
    plt.figure(figsize=(9, 7))
    
    for label, csv_path in runs.items():
        if not os.path.exists(csv_path):
            print(f"Warning: Missing log file {csv_path}, skipping.")
            continue
            
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        labels = df['label'].values
        scores = df['score'].values
        
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[label], lw=3, label=f'{label} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='#a0aec0', lw=1.5, linestyle='--', label='Random Guess')
    
    # Highlight competitor Causal-IDS baseline
    plt.scatter([0.05], [0.84], color='#e53e3e', marker='X', s=120, zorder=5, label='Causal-IDS (2026) Baseline')
    plt.plot([0.05, 0.05], [0.0, 0.84], color='#e53e3e', linestyle=':', alpha=0.7)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('True Positive Rate (TPR) / Recall', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('ZeroCausal Multi-Benchmark Generalization Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True, facecolor='white', edgecolor='#e2e8f0')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_dir = os.path.join(os.getcwd(), "results", "final")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_comparison_roc.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Saved comparative ROC plot to {out_path}")
    
    # Copy to plots folder
    os.makedirs("plots", exist_ok=True)
    import shutil
    shutil.copy(out_path, "plots/benchmark_comparison_roc.png")
    
    # Copy to artifacts directory
    for conv_id in ["ad6045ea-9073-4041-b8f0-6ca206a46340", "873c6033-7d1f-4ef6-9c05-38e627d9e107"]:
        artifact_path = f"/DATA/shourya_2211mc14/.gemini/antigravity-ide/brain/{conv_id}/plots/benchmark_comparison_roc.png"
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        shutil.copy(out_path, artifact_path)
        print(f"Copied comparative ROC plot to {artifact_path}")

if __name__ == "__main__":
    main()
