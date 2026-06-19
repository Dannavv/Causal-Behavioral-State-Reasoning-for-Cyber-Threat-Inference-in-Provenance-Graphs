import pandas as pd
import numpy as np

def generate_synthetic_baseline(num_windows=100):
    """
    Generates a synthetic time-series baseline of normal system activity.
    Edges like 'svchost->dns', 'chrome->dns', 'word->file'
    """
    np.random.seed(42)
    time_index = pd.date_range('2026-06-14 00:00:00', periods=num_windows, freq='1min')
    
    data = {
        'svchost->dns': np.random.poisson(lam=10, size=num_windows),
        'chrome->dns': np.random.poisson(lam=5, size=num_windows),
        'word->file': np.random.poisson(lam=2, size=num_windows),
        'word->powershell': np.zeros(num_windows), # Normal: never happens
        'powershell->registry': np.random.poisson(lam=0.1, size=num_windows)
    }
    
    return pd.DataFrame(data, index=time_index)

def inject_anomaly(df, anomaly_time_idx):
    """
    Injects an APT-like causal anomaly (e.g., word spawning powershell)
    """
    print(f"Injecting anomaly at index {anomaly_time_idx}...")
    df.iloc[anomaly_time_idx, df.columns.get_loc('word->powershell')] = 1
    df.iloc[anomaly_time_idx+1, df.columns.get_loc('powershell->registry')] += 5
    return df

if __name__ == "__main__":
    print("ZeroCausal - Synthetic Anomaly Verification")
    print("Generating baseline data...")
    baseline_df = generate_synthetic_baseline()
    
    print("Baseline sample:")
    print(baseline_df.head())
    
    print("\nInjecting synthetic APT attack (word -> powershell -> registry)...")
    attack_df = inject_anomaly(baseline_df.copy(), anomaly_time_idx=50)
    
    print("\nIn an actual run, this data would be fed into 03_zero_causal_detector.py")
    print("The PCMCI algorithm will flag the 'word->powershell' edge because its")
    print("interventional probability P(powershell | do(word)) is 0 in the baseline.")
    
    # We can invoke the template detector functions here once fully integrated.
    print("Verification script ready.")
