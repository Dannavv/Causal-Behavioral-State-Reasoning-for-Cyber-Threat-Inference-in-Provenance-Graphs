import polars as pl
import argparse
import os

def parse_optc_to_graph(parquet_path, num_events=10000, output_csv="optc_edges.csv"):
    """
    Reads OpTC parquet, extracts provenance edges, and saves to CSV.
    Uses polars for fast, out-of-core scanning.
    """
    print(f"Loading first {num_events} events from {parquet_path}...")
    
    try:
        # Load up to num_events rows
        df = pl.scan_parquet(parquet_path).head(num_events).collect()
    except Exception as e:
        print(f"Error loading {parquet_path}: {e}")
        return
        
    print(f"Extracting causal edges from {len(df)} rows...")
    edges = []
    
    # Process rows to build edges
    for row in df.to_dicts():
        timestamp = row.get("timestamp")
        action = row.get("action")
        obj_type = row.get("object")
        props = row.get("properties") or {}
        
        # Source is usually the actor process
        src_image = props.get("image_path")
        if not src_image:
            src_image = "unknown_process"
        
        src_type = "PROCESS"
        src_id = src_image.split("\\")[-1] if "\\" in src_image else src_image
        
        dst_type = obj_type
        dst_id = "unknown"
        edge_action = action
        
        if obj_type == "FLOW":
            dest_ip = props.get("dest_ip")
            dest_port = props.get("dest_port")
            dst_id = f"{dest_ip}:{dest_port}"
            edge_action = "CONNECTS_TO"
        elif obj_type == "FILE":
            file_path = props.get("file_path")
            dst_id = file_path.split("\\")[-1] if file_path and "\\" in file_path else str(file_path)
            if action == "CREATE": edge_action = "CREATES_FILE"
            elif action == "READ": edge_action = "READS_FILE"
            elif action == "WRITE": edge_action = "WRITES_FILE"
        elif obj_type == "PROCESS":
            tgt_process = props.get("command_line") or "unknown"
            dst_id = tgt_process
            if action == "START": edge_action = "SPAWNS_PROCESS"
            
        # Filter out extreme noise if needed (e.g., skip self-referential or empty)
        if src_id and dst_id and dst_id != "unknown" and dst_id != "None":
            edges.append({
                "timestamp": timestamp,
                "src_type": src_type,
                "src_id": src_id,
                "action": edge_action,
                "dst_type": dst_type,
                "dst_id": dst_id
            })

    edges_df = pl.DataFrame(edges)
    
    print(f"Extracted {len(edges_df)} valid causal edges.")
    print("\nSample of Extracted Edges:")
    print(edges_df.head())
    
    edges_df.write_csv(output_csv)
    print(f"\nSaved edges to {output_csv}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to OpTC parquet file")
    parser.add_argument("--num_events", type=int, default=10000)
    args = parser.parse_args()
    
    if args.input and os.path.exists(args.input):
        parse_optc_to_graph(args.input, args.num_events)
    else:
        print("Please provide a valid --input path.")
