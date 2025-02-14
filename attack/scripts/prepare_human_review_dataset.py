import pandas as pd
import os
import re
import uuid
import ast

def process_dataframe(df, threshold):
    
    results = []
    
    for group_id, group in df.groupby("group_id"):

        first_row = group.loc[group["step_num"] == 0].head(1)
        final_row = group.loc[(group["quality_preserved"] == True) & (group["watermark_score"].notna())].tail(1)
        
        if first_row.empty:
            print(f"[MAIN] Missing step_num == 0 for group_id={group_id}")
            continue

        if final_row.empty:
            print(f"[MAIN] No successful mutations found for group_id={group_id}")
            continue
        
        final_row_instance = final_row.iloc[0]
        if final_row_instance['watermark_score'] > threshold:
            print(f"[MAIN] Skipping group {group_id} with watermark_score {final_row_instance['watermark_score']} > threshold {threshold}")
            continue
        
        first_row = first_row.iloc[0]
        final_row = final_row_instance
        
        # Extract required values
        prompt = first_row["prompt"]
        initial_text = first_row["current_text"]
        attacked_text = final_row["mutated_text"]
        mutation_num = final_row["mutation_num"]
        final_steps = final_row["step_num"]

        if mutation_num == 0:
            continue
        
        # Extract scores safely
        initial_quality_analysis = ast.literal_eval(first_row["quality_analysis"])
        final_quality_analysis = ast.literal_eval(final_row["quality_analysis"])
        
        initial_score = initial_quality_analysis.get("original_score_A", None)
        attacked_score = final_quality_analysis.get("original_score_B", None)
        
        # Append the processed data
        results.append({
            "row_id": str(uuid.uuid4()),
            "group_id": group_id,
            "prompt": prompt,
            "initial_text": initial_text,
            "attacked_text": attacked_text,
            "mutation_num": mutation_num, 
            "final_steps": final_steps,
            "initial_InternLMOracle_score": initial_score,
            "attacked_InternLMOracle_score": attacked_score
        })
    
    return pd.DataFrame(results)

def obscure_texts(df):

    # Ensure required columns exist
    if 'initial_text' in df.columns and 'attacked_text' in df.columns:
        # Create shuffled text_A and text_B assignments
        import numpy as np

        shuffled = np.random.permutation(len(df))
        text_A = np.where(shuffled % 2 == 0, df['initial_text'], df['attacked_text'])
        text_B = np.where(shuffled % 2 == 0, df['attacked_text'], df['initial_text'])

        # Create the mutated_text column indicating which text was the attacked_text
        mutated_text = np.where(shuffled % 2 == 0, 'text_B', 'text_A')

        # Create the modified DataFrame
        df_modified = df.copy()
        df_modified['text_A'] = text_A
        df_modified['text_B'] = text_B
        df_modified['mutated_text'] = mutated_text

        # Drop the original columns
        df_modified.drop(columns=['initial_text', 'attacked_text'], inplace=True)

    return df_modified

 
def main():

    from attack.utils import load_all_csvs

    watermarks = ["Adaptive", "KGW", "SIR"]
    mutators = [
        "Document1StepMutator", "Document2StepMutator", "DocumentMutator", 
        "SentenceMutator", "SpanMutator", "WordMutator", "EntropyWordMutator"
    ]

    unwatermarked_mean_std = {
        "Adaptive": (49.42577, 3.365801),
        "KGW": (-0.82778, 1.047094772),
        "SIR": (0.077541, 0.068233825),
    }
   
    print(f"[MAIN] Try loading existing datasets...")
    try:

        final_df = pd.read_csv("./data/final_review/full_dataset_len=425.csv")
        final_df_subsampled = pd.read_csv("./data/final_review/subsampled_dataset_len=208.csv")

        print(f"[MAIN] Success. Skipping recomputation...")

    except:

        print(f"[MAIN] Failed to load existing datasets...")

        dfs = []

        for watermark in watermarks:

            if watermark not in unwatermarked_mean_std:
                print(f"[MAIN] Skipping {watermark} - no threshold data available.")
                continue

            score_mean, score_std = unwatermarked_mean_std[watermark]
            threshold = score_mean + 2 * score_std

            for mutator in mutators:

                trace_df = load_all_csvs("./attack/traces/annotated", watermark, mutator)
                
                if trace_df.empty:
                    print(f"[MAIN] No traces found for {watermark} + {mutator}")
                    continue

                # Apply Adaptive watermark score filtering
                if watermark == "Adaptive":
                    trace_df = trace_df[~trace_df['watermark_score'].between(-0.0001, 0.0001)]

                print(f"[MAIN] Processing traces for {watermark} + {mutator}")

                first_vs_final_df = process_dataframe(trace_df, threshold)
                first_vs_final_df["watermark"] = [watermark] * len(first_vs_final_df)
                first_vs_final_df["mutator"] = [mutator] * len(first_vs_final_df)
                
                dfs.append(first_vs_final_df)

        final_df = pd.concat(dfs, axis=0)
        final_df = final_df.dropna(subset=["initial_InternLMOracle_score", "attacked_InternLMOracle_score"])

        final_df.to_csv(f"./data/final_review/full_dataset_len={len(final_df)}.csv", index=False)

        # Subsample
        N = 20
        final_df_subsampled = final_df.groupby(['watermark', 'mutator'], group_keys=False).apply(lambda x: x.sample(min(len(x), N)))
        final_df_subsampled.to_csv(f"./data/final_review/subsampled_dataset_len={len(final_df_subsampled)}.csv", index=False)

    final_df_obscured = obscure_texts(final_df)
    final_df_obscured.to_csv(f"./data/final_review/full_dataset_len={len(final_df_obscured)}_obscured.csv", index=False)

    final_df_subsampled_obscured = obscure_texts(final_df_subsampled)
    final_df_subsampled_obscured.to_csv(f"./data/final_review/subsampled_dataset_len={len(final_df_subsampled_obscured)}_obscured.csv", index=False)

if __name__ == "__main__":

    # python -m attack.scripts.prepare_human_review_dataset

    main()
