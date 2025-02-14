import os
import numpy as np
import pandas as pd
import ast

if __name__ == "__main__":

    # python -m attack.analysis.mutator_statistics

    from attack.utils import load_all_csvs

    watermark_types = ["Adaptive", "KGW", "SIR"]
    mutators = ["DocumentMutator", "Document1StepMutator", "Document2StepMutator", 
            "SentenceMutator", "SpanMutator", "WordMutator", "EntropyWordMutator"]

    # step_num	mutation_num	prompt	current_text	mutated_text	current_text_len	mutated_text_len	
    # length_issue	quality_analysis	quality_preserved	watermark_detected	watermark_score	backtrack	
    # total_time	mutator_time	oracle_time	group_id	armolm_quality	words_edited	perplexity	
    # difforacle_quality	offsetbias_quality	grammar_errors

    results = []
    for watermark in watermark_types:
        for mutator in mutators:
            print(f"[MAIN] Processing {watermark} + {mutator}...")
            df = load_all_csvs("./attack/traces/annotated", watermark, mutator)
            
            if df.empty:
                df = load_all_csvs("./attack/traces", watermark, mutator)
                if df.empty:
                    print(f"[MAIN] No traces found for {watermark} + {mutator}")
                    continue

            def extract_score(d):
                try:
                    d = ast.literal_eval(d)
                    if isinstance(d, dict):
                        return d.get('original_score_B', np.nan)
                except:
                    return np.nan
                return np.nan

            df['internlm_quality'] = df['quality_analysis'].apply(extract_score) # df[df["quality_preserved"].astype(bool) == True]['quality_analysis'].apply(extract_score)

            results.append({
                "watermark": watermark,
                "mutator": mutator, 
                "mean_perplexity": df['perplexity'].mean() if 'perplexity' in df.columns else None,
                "mean_words_edited": df['words_edited'].mean() if 'words_edited' in df.columns else None,
                "mean_grammar_errors": df['grammar_errors'].mean() if 'grammar_errors' in df.columns else None,
                "mean_internlm_quality": df['internlm_quality'].mean() if 'internlm_quality' in df.columns else None,
                "mean_approval_rate": df['quality_preserved'].astype(bool).mean() if 'quality_preserved' in df.columns else None,
            })


    df = pd.DataFrame(results)

    save_path = "./attack/analysis/csv/mutator_statistics.csv"
    df.to_csv(save_path, index=False)
    print(f"[MAIN] Saved mutator statistics to {save_path}")
