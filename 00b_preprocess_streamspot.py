import pandas as pd
import numpy as np
import os

def preprocess_streamspot():
    print("Reading StreamSpot all.tsv...")
    # StreamSpot format: source_id, source_type, dest_id, dest_type, edge_type, graph_id
    df = pd.read_csv("data/raw/streamspot/all.tsv", sep='\t', header=None, 
                     names=['source_id', 'source_type', 'dest_id', 'dest_type', 'edge_type', 'graph_id'])
    
    print("Constructing edge representations...")
    # Provenance edge: source_type -> edge_type -> dest_type
    df['edge'] = df['source_type'].astype(str) + '->' + df['edge_type'].astype(str) + '->' + df['dest_type'].astype(str)
    
    print("Binning into sequential time windows...")
    # StreamSpot is a sequential stream of graphs without exact timestamps.
    # We chunk the stream into fixed-size windows to simulate continuous streaming (1000 edges per window).
    window_size = 1000
    df['timestamp'] = np.arange(len(df)) // window_size
    
    # In StreamSpot, there are 6 scenarios (100 graphs each).
    # The drive-by download attack corresponds to graph_ids 300 to 399.
    df['is_attack'] = ((df['graph_id'] >= 300) & (df['graph_id'] < 400)).astype(int)
    
    print("Pivoting to edge-count matrix...")
    # Group by window and edge to get counts
    counts = df.groupby(['timestamp', 'edge']).size().reset_index(name='count')
    pivot = counts.pivot(index='timestamp', columns='edge', values='count').fillna(0)
    
    # Max attack label in each window
    labels = df.groupby('timestamp')['is_attack'].max()
    
    final = pd.concat([labels, pivot], axis=1)
    
    os.makedirs('data/processed', exist_ok=True)
    out_path = 'data/processed/streamspot_edges.csv'
    final.to_csv(out_path)
    print(f"Saved preprocessed StreamSpot data to {out_path} with shape {final.shape}")

if __name__ == "__main__":
    preprocess_streamspot()
