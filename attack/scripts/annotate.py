<<<<<<< HEAD
# RUN: CUDA_VISIBLE_DEVICES=6,7 python -m attack.scripts.annotate

=======
>>>>>>> 3f0c8be7a83fecd1c1feec63b76b111bba8c21b6
import pandas as pd
import re
import os
import torch
import logging
import glob
import traceback
<<<<<<< HEAD
from extractors import FluencyMetric, GrammarMetric, EditsMetric, InternLMQualityMetric, QualityMetric
=======
import shutil  # for moving files

from extractors import FluencyMetric, GrammarMetric, EditsMetric
from attack.oracles import (
    ArmoRMOracle,
    INFORMOracle,
    InternLMOracle,
    QRMOracle,
    SkyworkOracle
) # NOTE: Weshould only annotate with the TOP 3 from our quality oracle analysis

>>>>>>> 3f0c8be7a83fecd1c1feec63b76b111bba8c21b6
from watermark.get_watermark import get_watermark, cleanup_resources
from distinguisher.utils import parse_filename

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# These columns must be present and non-null to consider the CSV "fully annotated"
REQUIRED_COLUMNS = ["words_edited", "perplexity", "grammar_errors", "internlm_quality"] # TODO: update this with the other quality oracles when we decide which ones they are. 

def assign_unique_group_ids(df):
    df['new_group'] = (df['step_num'] == 0).astype(int)
    df['group_id'] = df['new_group'].cumsum()
    return df

def is_fully_annotated(df):
    """
    Return True if df has all REQUIRED_COLUMNS and no nulls in those columns.
    """
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            return False
        if df[col].isnull().any():
            return False
    return True

def evaluate_column(df, column):
    """
    Load the specific metric *only* for the given 'column', evaluate the df, then unload.
    """
    # If you have columns that require the same metric, you might unify them here.
    # For your code, each column maps to a unique metric class.
    try:
        if column == "words_edited":
            metric = EditsMetric()
            # Evaluate
            df = metric.evaluate_dataframe(
                df,
                current_text_column="current_text",
                mutated_text_column="mutated_text",
                new_column="words_edited"
            )
            del metric
            torch.cuda.empty_cache()

        elif column == "perplexity":
            metric = FluencyMetric()
            df = metric.evaluate_dataframe(
                df, 
                text_column="mutated_text", 
                new_column="perplexity"
            )
            del metric
            torch.cuda.empty_cache()

        elif column == "grammar_errors":
            metric = GrammarMetric()
            df = metric.evaluate_dataframe(
                df,
                text_column="mutated_text", 
                new_column="grammar_errors"
            )
            del metric
            torch.cuda.empty_cache()

        elif column == "internlm_quality":
            metric = InternLMOracle()
            df = metric.score_dataframe(
                df, 
                prompt_column="prompt", 
                text_column="mutated_text", 
                new_column="internlm_quality"
            )
            del metric
            torch.cuda.empty_cache()
    except Exception as e:
        print("=" * 50, f" {column} ", "=" * 50)
        print(traceback.format_exc())

    return df

def annotate_if_missing(df):
    """
    For each required column:
      - If it's missing or contains null, load and run that metric only.
      - Then free GPU memory.
    """
    # A bit of housekeeping for text columns (similar to your original code)
    if "mutated_text" in df.columns:
        df["mutated_text"] = df["mutated_text"].fillna(df["current_text"])
    if "current_text" in df.columns:
        df['current_text'] = df["mutated_text"].shift(1)
        df["current_text"] = df["current_text"].fillna(df["mutated_text"])

    # Go column by column; only instantiate the relevant metric if needed.
    for col in REQUIRED_COLUMNS:
        if col not in df.columns or df[col].isnull().any():
            df = evaluate_column(df, col)

    return df

def main():
    # Get all .csv traces in ./attack/traces/
    traces = glob.glob("./attack/traces/*.csv")

    for trace in traces:
        log.info(f"Trace: {trace}")

        # Attempt to parse metadata (optional)
        try:
            o, w, m, s = parse_filename(trace)
            log.info(f"Oracle: {o}, Watermarker: {w}, Mutator: {m}, Steps: {s}")
        except:
            # If parse_filename fails, ignore
            pass

        # Read the DataFrame
        df = pd.read_csv(trace)
        df = assign_unique_group_ids(df)

        # Check if already fully annotated
        if is_fully_annotated(df):
            annotated_path = os.path.join("./attack/traces/annotated", os.path.basename(trace))
            log.info(f"Already fully annotated. Moving to: {annotated_path}")
            shutil.move(trace, annotated_path)
            continue
        else:
            # Otherwise, annotate
            log.info("Not fully annotated. Annotating now...")
            df = annotate_if_missing(df)

            # Now that we've done it, save to ./attack/traces/annotated/
            base_name = os.path.splitext(os.path.basename(trace))[0]
            annotated_path = os.path.join("./attack/traces/annotated", base_name)

            log.info(f"Saving new annotated file to: {annotated_path}")
            df.to_csv(annotated_path, index=False)
            
            # Optional: remove or keep the original - NOTE: Keeping for now to avoid catastrophe
            # os.remove(trace)

if __name__ == "__main__":
    # python -m attack.scripts.annotate
    main()
