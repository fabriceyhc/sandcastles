import pandas as pd
import os

def split_csv_by_approx_size(
    input_csv_path, 
    max_bytes=100 * 1024 * 1024,  # 100 MB
    small_chunk_rows=1000
):
    """
    Splits the CSV into multiple chunks, each trying not to exceed `max_bytes`.
    Reads the CSV in smaller chunks (small_chunk_rows) and accumulates until near max_bytes.
    
    Returns a list of chunk file paths created.
    """
    base_dir = os.path.dirname(input_csv_path)
    filename = os.path.basename(input_csv_path)
    file_root, file_ext = os.path.splitext(filename)
    
    output_paths = []
    part_idx = 1
    collector = []  # will hold small DataFrame chunks
    current_size = 0

    reader = pd.read_csv(input_csv_path, chunksize=small_chunk_rows)

    for df_small in reader:
        # Convert to CSV in memory (string), measure size
        csv_str = df_small.to_csv(index=False, header=False)  
        chunk_size = len(csv_str.encode('utf-8'))  

        if current_size + chunk_size > max_bytes:
            # Write out the existing collector as one chunk
            chunk_path = os.path.join(base_dir, f"{file_root}_part{part_idx}{file_ext}")
            _write_collector_as_csv(collector, chunk_path)
            output_paths.append(chunk_path)
            part_idx += 1

            # Reset collector
            collector = []
            current_size = 0

        # Add new small chunk
        collector.append(df_small)
        current_size += chunk_size

    # If there's anything left in collector after the loop, flush it
    if collector:
        chunk_path = os.path.join(base_dir, f"{file_root}_part{part_idx}{file_ext}")
        _write_collector_as_csv(collector, chunk_path)
        output_paths.append(chunk_path)

    return output_paths


def _write_collector_as_csv(collector, output_path):
    """
    Helper function to write a list of DataFrames to CSV (with header in the first chunk).
    """
    # Concatenate everything
    df = pd.concat(collector, ignore_index=True)
    # Write with header (we assume same columns throughout)
    df.to_csv(output_path, index=False, header=True)


if __name__ == "__main__":

    # python -m attack.scripts.split

    # NOTE: This script currently does not delete the original traces automatically.
    #       I'm proposing to keep it that way to avoid accidents and tradgedy. 
    #       Manually remove the originals for check-in for now. 

    import glob

    traces = glob.glob("./attack/traces/*")
    
    for trace in traces:
        chunk_paths = split_csv_by_approx_size(trace)
        print(f"Created chunk files for {trace}:")
        for cp in chunk_paths:
            print(cp)