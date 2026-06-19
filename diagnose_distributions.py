import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_dataset(name, csv_path):
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} does not exist. Skipping {name}.")
        return None
        
    print(f"\n========================================")
    print(f"Analyzing {name} ({csv_path})...")
    df = pd.read_csv(csv_path)
    labels = df['label'].values
    scores = df['score'].values
    
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    
    print(f"Normal samples count: {len(neg_scores)}")
    print(f"Anomaly samples count: {len(pos_scores)}")
    
    print("Normal score percentiles:")
    for p in [0, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  {p}%: {np.percentile(neg_scores, p):.6f}")
        
    print("Anomaly score percentiles:")
    for p in [0, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  {p}%: {np.percentile(pos_scores, p):.6f}")
        
    # Plotting score distributions
    plt.figure(figsize=(8, 4))
    
    # We use log scale if values range is huge
    use_log = scores.max() / (scores.min() + 1e-12) > 100
    
    if use_log:
        bins = np.logspace(np.log10(max(1e-12, scores.min())), np.log10(scores.max()), 50)
        plt.xscale('log')
    else:
        bins = np.linspace(scores.min(), scores.max(), 50)
        
    plt.hist(neg_scores, bins=bins, alpha=0.6, label='Normal (Benign)', color='#3182ce', density=True)
    plt.hist(pos_scores, bins=bins, alpha=0.6, label='Attack (Anomaly)', color='#e53e3e', density=True)
    
    plt.title(f'{name} Anomaly Score Distribution')
    plt.xlabel('Causal Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    out_dir = "/DATA/shourya_2211mc14/.gemini/antigravity-ide/brain/873c6033-7d1f-4ef6-9c05-38e627d9e107/plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name.lower().replace(' ', '_')}_scores.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved distribution plot to {out_path}")
    
    return neg_scores, pos_scores

def main():
    runs = {
        "OpTC": "logs/optc_tuned_steps.csv",
        "TC3": "logs/tc3_default_steps.csv",
        "NODLINK": "logs/nodlink_default_steps.csv",
        "BETH": "logs/beth_default_steps.csv",
        "StreamSpot": "logs/streamspot_default_steps.csv"
    }
    
    for name, path in runs.items():
        analyze_dataset(name, path)

if __name__ == "__main__":
    main()
