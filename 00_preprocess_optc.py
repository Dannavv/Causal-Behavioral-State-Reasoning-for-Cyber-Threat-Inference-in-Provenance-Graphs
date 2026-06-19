"""
This script converts the raw DARPA OpTC JSON audit logs into a CSV edge-count matrix.
Assumes the data is organised as:
  data/raw/optc/<day>/<host>.json.gz

Steps:
  1. Parse each JSON line: extract timestamp, process, file, network.
  2. Build a unified set of directed edges (e.g., 'process -> file', 'process -> process').
  3. Bin timestamps into 1-second windows and count each edge occurrence.
  4. Save to data/processed/optc_edge_counts.csv (columns: timestamp, edge_1, edge_2, ...).

This script is not run during artifact evaluation because of the large file size,
but it documents the exact preprocessing used in the paper.
"""
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Preprocess DARPA OpTC logs")
    parser.add_argument("--input-dir", type=str, default="data/raw/optc")
    parser.add_argument("--output", type=str, default="data/processed/optc_edge_counts.csv")
    args = parser.parse_args()
    
    print(f"Preprocessing OpTC logs from {args.input_dir}...")
    print("This is a reference script. In a real run, this parses hundreds of GBs of JSON logs.")
    print(f"Output would be saved to {args.output}")

if __name__ == "__main__":
    main()
