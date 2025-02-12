# CUDA_VISIBLE_DEVICES=1 python -m generate.annotate

import pandas as pd
from extractors import FluencyMetric, GrammarMetric, EditsMetric
from attack.oracles import InternLMOracle
import traceback
import torch
import nltk
from nltk.tokenize import word_tokenize

# Ensure you have the necessary NLTK resources
nltk.download('punkt')

def evaluate_column(df, column):
    """
    Compute missing values for a specific column using the corresponding metric.
    """
    metric = None
    try:
        if column == "perplexity":
            print(f"Loading FluencyMetric()")
            metric = FluencyMetric()
            df = metric.evaluate_dataframe(df, "text", "perplexity")
        elif column == "grammar_errors":
            print(f"Loading GrammarMetric()")
            metric = GrammarMetric()
            df = metric.evaluate_dataframe(df, "text", "grammar_errors")
        elif column == "internlm_quality":
            print(f"Loading InternLMOracle()")
            metric = InternLMOracle()
            df = metric.score_dataframe(df, "prompt", "text", "internlm_quality")
        elif column == "word_count":
            print(f"Calculating word_count...")
            df["word_count"] = df["text"].dropna().apply(lambda text: len(word_tokenize(text)))
        else:
            print("WRONG COLUMN!")
        
        # Clean up
        del metric
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error in {column}: {str(e)}\n{traceback.format_exc()}")
    return df

REQUIRED_COLUMNS = ["perplexity", "grammar_errors", "internlm_quality", "word_count"] 

watermarks = ["unwatermarked"]

for watermark in watermarks:

    print(f"Running for: {watermark}")

    in_path = f"./data/texts/entropy_control_{watermark}.csv"

    df = pd.read_csv(in_path)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df = evaluate_column(df, col)

    out_path = f"./data/texts/entropy_control_{watermark}.csv"   
    df.to_csv(out_path, index=False) 
    print(f"Saved to: {out_path}")
