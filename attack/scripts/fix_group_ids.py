import os
import glob
import pandas as pd

def assign_unique_group_ids(df):
    df['step_num'] = pd.to_numeric(df['step_num'], errors='coerce')
    
    # Check if there is any row with step_num == -1
    if (df['step_num'] == -1).any():
        df['new_group'] = (df['step_num'] == -1).astype(int)
    else:
        # If no row with -1 found, use step_num == 0
        df['new_group'] = (df['step_num'] == 0).astype(int)
        
    df['group_id'] = df['new_group'].cumsum()
    df.drop(columns=['new_group'], inplace=True)
    return df

def load_partitioned_data(base_dir, watermark_str, mutator_str, oracle_str):
    """Load all CSV parts with tracking of original file paths"""
    pattern = os.path.join(base_dir, f"{oracle_str}_{watermark_str}_{mutator_str}*")
    csv_files = glob.glob(pattern)

    # Sort them so part1 < part2 < part3, etc. (if you have chunked files)
    # extract the part number
    part = lambda filename: int(filename.split("_part")[-1].split(".")[0]) if "_part" in filename else 0
    # sort by the base name and then by the part number by converting the filename to a tuple
    cmp = lambda filename: (filename.split("_part")[0], part(filename))
    csv_files.sort(key=cmp)

    dfs = []
    for path in csv_files:
        df = pd.read_csv(path)
        df['__original_part_path__'] = path  # Track original file path
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def save_partitions(combined_df):
    """Save processed data back to original partition files"""
    if combined_df.empty:
        return

    for path in combined_df['__original_part_path__'].unique():
        # Filter and clean partition data
        part_df = combined_df[combined_df['__original_part_path__'] == path]
        part_df = part_df.drop(columns=['__original_part_path__'])
        
        # Save to original path
        part_df.to_csv(path, index=False)

def compute_new_column(combined_df):
    """Example computation that uses full dataset"""
    # Replace this with your actual column computation
    combined_df = combined_df.copy()
    
    combined_df = assign_unique_group_ids(combined_df)
    
    return combined_df

def process_partitions(base_dir, watermark_str, mutator_str, oracle_str):
    """Main processing pipeline"""
    # Load data with original path tracking
    combined_df = load_partitioned_data(base_dir, watermark_str, mutator_str, oracle_str)
    
    if combined_df.empty:
        print("No matching files found")
        return

    # Add your computed columns here
    processed_df = compute_new_column(combined_df)
    
    # Save back to original partitions
    save_partitions(processed_df)
    print(f"Processed {len(processed_df)} rows across " 
          f"{len(processed_df['__original_part_path__'].unique())} files")


if __name__ == "__main__":

    # python -m attack.scripts.fix_group_ids

    oracles = [
        # "DiffOracle",
        "InternLMOracle"
    ]

    watermark_types = [
        "Adaptive",
        "KGW",
        "SIR",
        "GPT4o_unwatermarked",
    ]

    mutators = [
        # "DocumentMutator",
        # "Document1StepMutator",
        # "Document2StepMutator",
        # "SentenceMutator",
        # "SpanMutator",
        "WordMutator",
        "EntropyWordMutator",
    ]

    for oracle in oracles:

        for watermark_type in watermark_types:
            
            for mutator in mutators:

                print(f"[MAIN] Processing {watermark_type} + {mutator}")

                # process_partitions("./attack/traces", watermark_type, mutator, oracle)
                process_partitions("./attack/traces/annotated", watermark_type, mutator, oracle)