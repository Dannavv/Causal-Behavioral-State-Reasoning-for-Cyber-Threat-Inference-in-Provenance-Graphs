import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    datasets = ['TC3', 'NODLINK', 'BETH', 'OpTC', 'StreamSpot']
    
    # AUROC values
    v1_auc = [0.8350, 0.8258, 0.9656, 0.8359, 0.4991]
    v2_auc = [1.0000, 1.0000, 1.0000, 0.9628, 0.6909]
    if_auc = [0.8738, 0.8902, 0.9981, 0.5968, 0.6425]
    ae_auc = [1.0000, 1.0000, 0.9954, 0.9364, 0.7770]
    
    x = np.arange(len(datasets))
    width = 0.2
    
    plt.figure(figsize=(10, 6.5))
    
    # Modern publication colors
    rects1 = plt.bar(x - 1.5*width, if_auc, width, label='Isolation Forest (Static)', color='#cbd5e0')
    rects2 = plt.bar(x - 0.5*width, ae_auc, width, label='PyTorch AE (Static/Offline)', color='#a3b18a')
    rects3 = plt.bar(x + 0.5*width, v1_auc, width, label='ZeroCausal v1 (Pure Causal)', color='#e07a5f')
    rects4 = plt.bar(x + 1.5*width, v2_auc, width, label='ZeroCausal v2 (Causal-ML Hybrid)', color='#3d5a80')
    
    plt.ylabel('AUC-ROC Score', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('Performance Comparison: How Causal-ML Hybrid Enhances Baseline AUC', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, datasets, fontsize=11, fontweight='semibold')
    plt.ylim(0.4, 1.05)
    
    plt.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', edgecolor='#e2e8f0')
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = 'plots/v1_vs_v2_comparison.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved to {out_path}")

if __name__ == '__main__':
    main()
