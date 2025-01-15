import pandas as pd
import os

def split_csv_by_approx_size(
    input_csv_path, 
    max_bytes=90 * 1024 * 1024,  # 90 MB
    small_chunk_rows=1000
):
    """
    Splits the CSV into multiple chunks, each trying not to exceed `max_bytes`.
    - Reads the CSV in smaller chunks (small_chunk_rows) and accumulates them
      until adding another chunk would exceed the max_bytes threshold.
    - If the file is already smaller than max_bytes, skip splitting.

    Returns a list of chunk file paths created (or a single path if no split).
    """

    # 1. Check if the file is already below the threshold
    file_size = os.path.getsize(input_csv_path)
    if file_size < max_bytes:
        # Skip splitting, return the original file path
        return [input_csv_path]

    # 2. Proceed with splitting
    base_dir = os.path.dirname(input_csv_path)
    filename = os.path.basename(input_csv_path)
    file_root, file_ext = os.path.splitext(filename)
    
    output_paths = []
    part_idx = 1

    # This collector accumulates dataframes until we exceed max_bytes
    collector = []
    current_size = 0

    # Read the file in smaller chunks via chunksize
    reader = pd.read_csv(input_csv_path, chunksize=small_chunk_rows)
    
    for df_small in reader:
        # Convert the small chunk to CSV in memory (string) to estimate its size
        csv_str = df_small.to_csv(index=False, header=False)  
        chunk_size = len(csv_str.encode('utf-8'))  

        # If adding this small chunk would exceed the limit, flush what we have
        if current_size + chunk_size > max_bytes and collector:
            chunk_path = os.path.join(base_dir, f"{file_root}_part{part_idx}{file_ext}")
            _write_collector_as_csv(collector, chunk_path)
            output_paths.append(chunk_path)

            part_idx += 1
            # Reset our collector and size
            collector = []
            current_size = 0

        # Add the new small chunk
        collector.append(df_small)
        current_size += chunk_size

    # 3. After the loop, if anything remains in the collector, write it out
    if collector:
        chunk_path = os.path.join(base_dir, f"{file_root}_part{part_idx}{file_ext}")
        _write_collector_as_csv(collector, chunk_path)
        output_paths.append(chunk_path)

    return output_paths


def _write_collector_as_csv(collector, output_path):
    """
    Helper function to write a list of DataFrames to a single CSV,
    including a header on the output.
    """
    # Concatenate the dataframes
    df = pd.concat(collector, ignore_index=True)
    # Write them as CSV with header
    df.to_csv(output_path, index=False, header=True)


if __name__ == "__main__":

    # python -m attack.scripts.split

    # NOTE: This script currently does not delete the original traces automatically.
    #       I'm proposing to keep it that way to avoid accidents and tradgedy. 
    #       Manually remove the originals for check-in for now. 

    import glob

    traces = glob.glob("./attack/traces/*")

    traces = [trace for trace in traces if "Adaptive" in trace and "WordMutator" in trace]
    
    for trace in traces:
        chunk_paths = split_csv_by_approx_size(trace)
        print(f"Created chunk files for {trace}:")
        for cp in chunk_paths:
            print(cp)