import pandas as pd
import os
import re
import uuid
import ast

def process_dataframe(df):
    
    results = []
    
    for group_id, group in df.groupby("group_id"):

        first_row = group.loc[group["step_num"] == 0].head(1)
        final_row = group.loc[group["quality_preserved"] == True & group["watermark_score"].notna()].tail(1)
        
        if first_row.empty:
            print(f"first_row: {first_row}")
            print(f"final_row: {final_row}")
            print(f"group: {group[['group_id', 'step_num', 'quality_preserved', 'watermark_score']]}")
            raise "Missing row!"

        if final_row.empty:
            print(f"[MAIN] No successful mutations found for group_id={group_id}")
            continue
        
        first_row = first_row.iloc[0]
        final_row = final_row.iloc[0]
        
        # Extract required values
        prompt = first_row["prompt"]
        initial_text = first_row["current_text"]
        attacked_text = final_row["mutated_text"]
        mutation_num = final_row["mutation_num"]
        final_steps = final_row["step_num"]
        
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
    
    # Convert results to DataFrame and return
    return pd.DataFrame(results)

# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def main():

    from attack.utils import load_all_csvs

    watermarks = ["Adaptive", "KGW", "SIR"]
    mutators = [
        "Document1StepMutator", "Document2StepMutator", "DocumentMutator", 
        "SentenceMutator", "SpanMutator", "WordMutator", "EntropyWordMutator"
    ]

    # watermarks.reverse()
    # mutators.reverse()

    dfs = []

    for watermark in watermarks:

        for mutator in mutators:

            trace_df = load_all_csvs("./attack/traces/annotated", watermark, mutator)
            
            if trace_df.empty:
                print(f"[MAIN] No traces found for {watermark} + {mutator}")
                continue

            print(f"[MAIN] Processing traces for {watermark} + {mutator}")

            first_vs_final_df = process_dataframe(trace_df)
            first_vs_final_df["watermark"] = [watermark] * len(first_vs_final_df)
            first_vs_final_df["mutator"] = [mutator] * len(first_vs_final_df)
            
            dfs.append(first_vs_final_df)

    final_df = pd.concat(dfs, axis=0)
    final_df = final_df.dropna(subset=["initial_InternLMOracle_score", "attacked_InternLMOracle_score"])

    print(final_df)

    final_df.to_csv(f"./data/final_review/full_dataset_len={len(final_df)}.csv", index=False)

    # SUBSAMPLE

    N = 10  # Maximum number of rows per group
    final_df_subsampled = final_df.groupby(['watermark', 'mutator'], group_keys=False).apply(lambda x: x.sample(min(len(x), N)))
    print(final_df_subsampled)

    final_df_subsampled.to_csv(f"./data/final_review/subssampled_dataset_len={len(final_df_subsampled)}.csv", index=False)


if __name__ == "__main__":

    # python -m attack.scripts.prepare_human_review_dataset

    main()
