import pandas as pd
import numpy as np
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import warnings
warnings.filterwarnings("ignore")

def build_time_series_from_edges(edges_df, time_window='1min'):
    """
    Converts an event edge list into a multivariate time series.
    Each column represents a specific edge type (e.g., 'word.exe->powershell.exe').
    Rows are time windows, values are counts of that edge.
    """
    if edges_df.empty:
        return pd.DataFrame()
        
    edges_df['timestamp'] = pd.to_datetime(edges_df['timestamp'], format='ISO8601')
    
    # Create a unique string identifier for each edge type
    edges_df['edge_type'] = edges_df['src_type'] + ":" + edges_df['src_id'] + " -> " + \
                            edges_df['action'] + " -> " + edges_df['dst_type'] + ":" + edges_df['dst_id']
                            
    # Pivot to get time series of counts
    ts_df = edges_df.groupby([pd.Grouper(key='timestamp', freq=time_window), 'edge_type']).size().unstack(fill_value=0)
    return ts_df

def run_pcmci_discovery(ts_df, tau_max=2, alpha_level=0.01):
    """
    Runs PCMCI on the multivariate time series to find the causal baseline graph.
    """
    print(f"Running PCMCI on shape {ts_df.shape} with tau_max={tau_max}...")
    var_names = ts_df.columns.tolist()
    data = ts_df.astype(float).values
    
    dataframe = pp.DataFrame(data, datatime=np.arange(len(ts_df)), var_names=var_names)
    
    # Using Partial Correlation for continuous count data (can use CMIknn for non-linear)
    cond_ind_test = ParCorr(significance='analytic')
    
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
    
    # Run PCMCI
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha_level)
    
    return pcmci, results

def calculate_anomaly_score(new_event_batch_df, pcmci_baseline, results, alpha_level=0.01):
    """
    Computes CAS for new events against the baseline causal graph.
    """
    # This is a template. Real implementation will check if the new batch introduces
    # edges that violate the p-value matrices found in `results['p_matrix']`.
    print("Calculating Causal Anomaly Score for new batch...")
    anomalies = []
    
    # Dummy logic for template
    for idx, row in new_event_batch_df.iterrows():
        # Example check
        cas = np.random.uniform(0, 1) # Placeholder for interventional prob calculation
        p_val = np.random.uniform(0, 0.05) # Placeholder
        if p_val < alpha_level:
            anomalies.append({
                'event': row.to_dict(),
                'CAS': cas,
                'p_value': p_val,
                'is_anomaly': True
            })
            
    return anomalies

if __name__ == "__main__":
    import os
    print("ZeroCausal Detector initialized.")
    csv_path = "optc_edges.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run stage 02 first.")
    else:
        print(f"Loading {csv_path}...")
        edges_df = pd.read_csv(csv_path)
        print(f"Loaded {len(edges_df)} edges.")
        
        # Use 1-second windows to ensure we get enough time steps from a short snapshot
        ts_df = build_time_series_from_edges(edges_df, time_window='1s')
        print(f"Built time series with shape: {ts_df.shape} (Time steps x Edge Types)")
        
        # We need at least a few time steps for PCMCI
        if ts_df.shape[0] > 5:
            # Drop edge types that never happen or are too sparse to avoid singular matrix
            ts_df = ts_df.loc[:, (ts_df.sum(axis=0) > 5)]
            print(f"Filtered to {ts_df.shape[1]} frequent edge types for stable causal discovery.")
            
            if ts_df.shape[1] > 0:
                pcmci, results = run_pcmci_discovery(ts_df, tau_max=1, alpha_level=0.01)
                print("\nPCMCI Discovery Complete.")
                print("Significant causal links discovered (p < 0.01):")
                # Print non-zero entries in p_matrix
                p_matrix = results['p_matrix']
                val_matrix = results['val_matrix']
                var_names = ts_df.columns.tolist()
                
                for i in range(len(var_names)):
                    for j in range(len(var_names)):
                        for tau in range(1, p_matrix.shape[2]):
                            if p_matrix[i, j, tau] < 0.01:
                                print(f"[{var_names[i]}] --(tau={tau}, p={p_matrix[i,j,tau]:.4f})--> [{var_names[j]}]")
            else:
                print("Not enough frequent edges after filtering.")
        else:
            print("Not enough time steps for causal discovery. Try extracting more events.")
