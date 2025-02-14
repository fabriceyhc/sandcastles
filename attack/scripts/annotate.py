import pandas as pd
import os
import glob
import traceback
import shutil
import torch
import re
import logging
from guidance import models 

# ---------------------------------------------------------------------
# Additional imports from your original environment
# ---------------------------------------------------------------------
from extractors import FluencyMetric, GrammarMetric, EditsMetric
# import (
#     ArmoRMOracle,
#     OffsetBiasOracle,
#     DiffOracle
# )
from extractors.diversity import UniqueBigramsDiversity

# ---------------------------------------------------------------------
# Configure logger
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants and required columns
# ---------------------------------------------------------------------
REQUIRED_COLUMNS = [
    # "armolm_quality", "offsetbias_quality", "difforacle_quality",
    "words_edited", "perplexity", "grammar_errors", "unique_bigrams"
] 

# ---------------------------------------------------------------------
# Helper: parse filename
# ---------------------------------------------------------------------
def parse_filename(path):
    """
    Extract components from filename: {uuid}_{watermark}_{mutator}_part{N}.csv
    Example: 1234_Adaptive_SentenceMutator_part2.csv
    """
    basename = os.path.basename(path)
    pattern = r"^(.+)_(Adaptive|KGW|SIR|GPT4o_unwatermarked)_(.*?)_*part(\d+)\.csv$"
    match = re.match(pattern, basename)
    if match:
        return {
            "uuid": match.group(1),
            "watermark": match.group(2),
            "mutator": match.group(3),
            "part_num": int(match.group(4))
        }
    return None

# ---------------------------------------------------------------------
# Helper: assign a cross-file group_id
# ---------------------------------------------------------------------
def assign_crossfile_group_ids(combined_df):
    """
    Compute a group ID for rows across files.
    We assume that step_num == -1 indicates the start of a new group.
    
    NOTE: The original code re-sorted the dataframe after computing group_id,
    which inadvertently reorders rows from their original (file) order.
    Removing that sort preserves the correct group order.
    """
    combined_df['new_group'] = (combined_df['step_num'] == -1).astype(int)
    combined_df['group_id'] = combined_df['new_group'].cumsum()
    combined_df.drop(columns=['new_group'], inplace=True)
    return combined_df

# ---------------------------------------------------------------------
# Helper: load partitioned data
# ---------------------------------------------------------------------
def load_partitioned_data(watermark, mutator):
    pattern = f"*_{watermark}_{mutator}_*.csv"  # Matches both with and without "part"
    source_dir = "./attack/traces/"
    annotated_dir = "./attack/traces/annotated"
    
    source_files = glob.glob(os.path.join(source_dir, pattern))
    annotated_files = glob.glob(os.path.join(annotated_dir, pattern))
    
    logger.debug(f"Looking for {watermark}-{mutator} files in:")
    logger.debug(f"- Source: {source_dir} -> Found: {len(source_files)}")
    logger.debug(f"- Annotated: {annotated_dir} -> Found: {len(annotated_files)}")

    # Combine file lists
    all_files = source_files + annotated_files
    if not all_files:
        logger.warning(f"No files found for {watermark}-{mutator}")
        return None, []
    
    # **** FIX: Sort files by the part number so that they load in the correct order ****
    all_files = sorted(
        all_files,
        key=lambda f: parse_filename(f)['part_num'] if parse_filename(f) is not None else 0
    )
    
    dfs = []
    valid_files = []
    required_columns = ["step_num", "current_text", "mutated_text"]
    
    for f in all_files:
        try:
            df = pd.read_csv(f)
            # Check for critical columns
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                logger.error(f"Skipping {f} - missing columns: {missing}")
                continue
                
            df['__filepath__'] = f
            dfs.append(df)
            valid_files.append(f)
            logger.debug(f"Loaded {f} with {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Failed to load {f}: {str(e)}")
    
    if not dfs:
        logger.warning(f"No valid data for {watermark}-{mutator}")
        return None, []
    
    combined_df = pd.concat(dfs, ignore_index=True)

    # IMPORTANT: Assign cross-file group IDs here (without re-sorting)
    if "group_id" not in combined_df.columns:
        combined_df = assign_crossfile_group_ids(combined_df)

    return combined_df, valid_files

# ---------------------------------------------------------------------
# Helper: check if dataframe is fully annotated
# ---------------------------------------------------------------------
def is_fully_annotated(df):
    """Check if all required columns exist and have no null values."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            return False
    return True

# ---------------------------------------------------------------------
# Helper: evaluate a specific column
# ---------------------------------------------------------------------
def evaluate_column(df, column, mutator):
    """
    Compute missing values for a specific column using the corresponding metric.
    """
    metric = None
    N = 10
    if "Word" in mutator:
        N = 100
    try:
        if column == "words_edited":
            print(f"Loading EditsMetric()")
            metric = EditsMetric()
            df = metric.evaluate_dataframe(df, "current_text", "mutated_text", "words_edited")
        elif column == "perplexity":
            print(f"Loading FluencyMetric()")
            metric = FluencyMetric()
            df = metric.evaluate_dataframe(df, "mutated_text", "perplexity", N)
        elif column == "grammar_errors":
            print(f"Loading GrammarMetric()")
            metric = GrammarMetric()
            df = metric.evaluate_dataframe(df, "mutated_text", "grammar_errors", N)
        elif column == "unique_bigrams":
            print(f"Calculating UniqueBigramsDiversity...")
            metric = UniqueBigramsDiversity()
            df = metric.evaluate_dataframe(df, "mutated_text", "unique_bigrams")
        # elif column == "armolm_quality":
        #     print(f"Loading ArmoRMOracle()")
        #     metric = ArmoRMOracle()
        #     df = metric.score_dataframe(df, "prompt", "mutated_text", "armolm_quality", N)
        # elif column == "offsetbias_quality":
        #     print(f"Loading OffsetBiasOracle()")
        #     metric = OffsetBiasOracle()
        #     df = metric.score_dataframe(df, "prompt", "mutated_text", "offsetbias_quality", N)
        # elif column == "difforacle_quality":
        #     print(f"Loading DiffOracle(Meta-Llama-3.1-70B-Instruct-q8_0.gguf)")
        #     llm = models.LlamaCpp(
        #         model="/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-q8_0.gguf",
        #         echo=False,
        #         n_gpu_layers=-1,
        #         n_ctx=4096*3
        #     )
        #     metric = DiffOracle(llm=llm)
        #     df = metric.score_dataframe(df, "prompt", "current_text", "mutated_text", "difforacle_quality", N)
        
        # Clean up
        del metric
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error in {column}: {str(e)}\n{traceback.format_exc()}")
    return df

# ---------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------
def process_watermark_mutator_group(watermark, mutator):
    combined_df, source_files = load_partitioned_data(watermark, mutator)
    if combined_df is None:
        logger.warning(f"Skipping {watermark}-{mutator} - no loadable data")
        return

    logger.info(f"Processing {watermark}-{mutator} | {len(combined_df)} rows from {len(source_files)} files")
    
    try:
        # Fill mutated_text if missing
        combined_df["mutated_text"] = combined_df["mutated_text"].fillna(combined_df["current_text"])
        
        # Track which required columns are initially missing
        initial_missing = [col for col in REQUIRED_COLUMNS if col not in combined_df.columns]
        if initial_missing:
            logger.info(f"Missing columns to compute: {initial_missing}")

        # Compute each needed column if missing or partially missing
        for col in REQUIRED_COLUMNS:
            if col not in combined_df.columns:
                logger.info(f"Computing {col}...")
                pre_count = len(combined_df)
                combined_df = evaluate_column(combined_df, col, mutator)
                logger.debug(f"{col} computation kept {len(combined_df)}/{pre_count} rows")
                
        # Check if fully annotated
        if is_fully_annotated(combined_df):
            logger.info(f"Successfully annotated all columns for {watermark}-{mutator}")
        else:
            missing = [col for col in REQUIRED_COLUMNS if col not in combined_df.columns]
            logger.warning(f"Still missing after computation: {missing}")

        # Save annotated results
        annotated_dir = "./attack/traces/annotated"
        os.makedirs(annotated_dir, exist_ok=True)

        for orig_path in source_files:
            part_df = combined_df[combined_df['__filepath__'] == orig_path].copy()
            part_df = part_df.drop(columns=['__filepath__'])
            
            dest_path = os.path.join(annotated_dir, os.path.basename(orig_path))
            
            # Write out the CSV file (overwriting if needed)
            part_df.to_csv(dest_path, index=False)
            
            # Remove the old file if it wasn’t already in the annotated folder
            if orig_path != dest_path and os.path.exists(orig_path):
                os.remove(orig_path)

            logger.info(f"Wrote annotated file with {len(part_df)} rows -> {dest_path}")

    except Exception as e:
        logger.error(f"Fatal error processing {watermark}-{mutator}: {str(e)}")
        logger.error(traceback.format_exc())

# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def main():
    watermark_types = ["Adaptive", "KGW", "SIR", "GPT4o_unwatermarked"]
    mutators = [
        "Document1StepMutator", "Document2StepMutator", "DocumentMutator", 
        "SentenceMutator", "SpanMutator", "WordMutator", "EntropyWordMutator"
    ]

    watermark_types.reverse()
    mutators.reverse()

    for watermark in watermark_types:
        for mutator in mutators:
            # Skip over incomplete traces
            if ((watermark == "GPT4o_unwatermarked" and mutator == "DocumentMutator") or
                # (watermark == "KGW" and mutator == "EntropyWordMutator") or
                (watermark == "Adaptive" and mutator == "DocumentMutator")
                ):
                print(f"Skipping {watermark} + {mutator}")
                continue

            print(f"Processing {watermark} - {mutator}")
            try:
                process_watermark_mutator_group(watermark, mutator)
            except Exception as e:
                print(f"Failed processing {watermark}-{mutator}: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0 python -m attack.scripts.annotate

    main()