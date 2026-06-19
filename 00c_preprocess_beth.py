import pandas as pd
import numpy as np
import os
import sys

def preprocess_beth(window_seconds=10):
    """
    Preprocesses BETH dataset using per-host files for full coverage.
    
    Uses the two attacked hosts (ip-10-100-1-105 and ip-10-100-1-4) which
    together provide 10,005 evil events across ~5 hours of activity each.
    
    Args:
        window_seconds: Size of time windows in seconds (default 10)
    """
    host_files = [
        ('data/raw/beth/labelled_2021may-ip-10-100-1-105.csv', 'host-105'),
        ('data/raw/beth/labelled_2021may-ip-10-100-1-4.csv', 'host-4'),
    ]
    
    all_dfs = []
    for path, host_label in host_files:
        print(f"[BETH] Reading {os.path.basename(path)}...")
        df = pd.read_csv(path, usecols=['timestamp', 'processName', 'eventName', 'evil', 'sus'])
        df['host'] = host_label
        total = len(df)
        evil = (df['evil'] == 1).sum()
        print(f"  → {total:,} rows, {evil:,} evil events, "
              f"timeline: {df['timestamp'].min():.1f}s – {df['timestamp'].max():.1f}s")
        all_dfs.append(df)
    
    print(f"\n[BETH] Combining {len(host_files)} hosts...")
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    total_rows = len(df)
    total_evil = (df['evil'] == 1).sum()
    print(f"  → Combined: {total_rows:,} rows, {total_evil:,} evil events")
    
    print(f"\n[BETH] Constructing edge representations (processName → eventName)...")
    df['edge'] = df['processName'].astype(str) + '->' + df['eventName'].astype(str)
    unique_edges = df['edge'].nunique()
    print(f"  → {unique_edges} unique edge types")
    
    print(f"\n[BETH] Binning into {window_seconds}-second time windows...")
    t_min = df['timestamp'].min()
    df['window'] = ((df['timestamp'] - t_min) / window_seconds).astype(int)
    num_windows = df['window'].nunique()
    avg_events = total_rows / num_windows
    print(f"  → {num_windows} windows, avg {avg_events:.0f} events/window")
    
    print(f"\n[BETH] Pivoting to edge-count matrix...")
    counts = df.groupby(['window', 'edge']).size().reset_index(name='count')
    pivot = counts.pivot(index='window', columns='edge', values='count').fillna(0)
    
    # Target label: max evil in each window
    labels = df.groupby('window')['evil'].max()
    
    final = pd.concat([labels, pivot], axis=1).fillna(0)
    final.rename(columns={'evil': 'is_attack'}, inplace=True)
    
    # Verify attack distribution
    attack_windows = (final['is_attack'] == 1).sum()
    normal_windows = (final['is_attack'] == 0).sum()
    
    # Check where attacks fall for split verification
    attack_indices = final.index[final['is_attack'] == 1].tolist()
    split_60 = int(len(final) * 0.6)
    attacks_in_train = sum(1 for i in attack_indices if i < split_60)
    attacks_in_test = sum(1 for i in attack_indices if i >= split_60)
    
    print(f"\n[BETH] Final dataset summary:")
    print(f"  → Shape: {final.shape}")
    print(f"  → Normal windows: {normal_windows}")
    print(f"  → Attack windows: {attack_windows} ({attack_windows/len(final)*100:.1f}%)")
    print(f"  → Attack index range: {min(attack_indices)} to {max(attack_indices)}")
    print(f"  → 60/40 split analysis:")
    print(f"    Training (first 60%): {attacks_in_train} attack windows")
    print(f"    Testing  (last 40%):  {attacks_in_test} attack windows")
    
    if attacks_in_test == 0:
        print(f"\n  ⚠️  WARNING: No attacks in test split! Consider adjusting window_seconds.")
    else:
        print(f"\n  ✅ Test split has {attacks_in_test} attack windows — AUC will be valid!")
    
    os.makedirs('data/processed', exist_ok=True)
    out_path = 'data/processed/beth_edges.csv'
    final.to_csv(out_path)
    print(f"\n[BETH] Saved preprocessed data to {out_path}")

if __name__ == "__main__":
    window_sec = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    preprocess_beth(window_seconds=window_sec)
