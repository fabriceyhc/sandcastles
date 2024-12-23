# RUN: CUDA_VISIBLE_DEVICES=3,4 python -m attack.annotate

import pandas as pd
import re
import os
import logging
import glob
import traceback
from extractors import FluencyMetric, GrammarMetric, EditsMetric, InternLMQualityMetric
from watermark.get_watermark import get_watermark

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

def assign_unique_group_ids(df):
    df['new_group'] = (df['step_num'] == 0).astype(int)
    df['group_id'] = df['new_group'].cumsum()
    return df

def get_support(df):
    return (df['step_num'] == 0).sum()

def get_max_step_count(df):
    return df['step_num'].max()

def main():
    # Initialize metric extractors
    fluency = FluencyMetric()
    grammar = GrammarMetric()
    quality = InternLMQualityMetric()
    edits   = EditsMetric()

    traces = glob.glob(f"./attack/traces/*.csv")

    for trace in traces:
        log.info(f"Trace: {trace}")

        o, w, m, s = os.path.basename(trace).split("_")[:4]
        s = int(s.replace("n-steps=", ""))

        log.info(f"Oracle: {o}")
        log.info(f"Watermarker: {w}")
        log.info(f"Mutator: {m}")
        log.info(f"Steps: {s}")

        output_file = trace

        if "annotated" not in output_file:
            suffix = re.search(r"results(.*)\.csv", output_file).group(1)
            output_file = output_file.replace(f"results{suffix}.csv", f"results_annotated{suffix}.csv")
        
            if output_file in traces:
                print(f"\tWould-be output file {output_file} already exists")
                continue

        log.info(f"Output File: {output_file}")
        
        df = assign_unique_group_ids(pd.read_csv(trace))

        # Ensure that there's no NaN values in the mutated and current text columns
        df["mutated_text"] = df["mutated_text"].fillna(df["current_text"])
        df['current_text'] = df['mutated_text'].shift(1)
        df["current_text"] = df["current_text"].fillna(df["mutated_text"])

        # step_num,mutation_num,prompt,current_text,mutated_text,current_text_len,mutated_text_len,length_issue,quality_analysis,quality_preserved,watermark_detected,watermark_score,backtrack,total_time,mutator_time,oracle_time
        if "words_edited" not in df.columns:
            try:
                df = edits.evaluate_dataframe(df, current_text_column="current_text", mutated_text_column="mutated_text", new_column="words_edited")
            except:
                print(f"{'=' * 50} words_edited {'=' * 50}")
                print(traceback.format_exc())
        if "perplexity" not in df.columns:
            try:
                df = fluency.evaluate_dataframe(df, text_column="mutated_text", new_column="perplexity")
            except:
                print(f"{'=' * 50} perplexity {'=' * 50}")
                print(traceback.format_exc())
        if "grammar_errors" not in df.columns:    
            try:
                df = grammar.evaluate_dataframe(df, text_column="mutated_text", new_column="grammar_errors")
            except:
                print(f"{'=' * 50} grammar_errors {'=' * 50}")
                print(traceback.format_exc())
        if "internlm_quality" not in df.columns:
            try:
                df = quality.evaluate_dataframe(df, prompt_column="prompt", text_column="mutated_text", new_column="internlm_quality")
            except:
                print(f"{'=' * 50} internlm_quality {'=' * 50}")
                print(traceback.format_exc()) 

        mask = df['watermark_score'].isna()
        if mask.any():
            watermark = get_watermark(w)

            def detect_watermark(row):
                if pd.isna(row['watermark_score']):
                    is_detected, score = watermark.detect(row['mutated_text'])
                    row['watermark_detected'] = is_detected
                    row['watermark_score'] = score
                return row

            df = df.apply(detect_watermark, axis=1)
        
        log.info(df)
        log.info(output_file)
        df.to_csv(output_file, index=False)

        break

if __name__ == "__main__":

    # python -m attack.scripts.annotate

    main()