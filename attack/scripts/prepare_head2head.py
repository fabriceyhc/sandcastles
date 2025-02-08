import pandas as pd
import os
import re
import uuid
import ast

def process_dataframe(df, oracle):
    
    results = []
    
    for group_id, group in df.groupby("group_id"):
        

        first_row = group.loc[group["step_num"] == 0].head(1)
        final_row = group.loc[group["quality_preserved"] == True].tail(1)
        
        if first_row.empty or final_row.empty:
            raise "Missing row!"
        
        first_row = first_row.iloc[0]
        final_row = final_row.iloc[0]
        
        # Extract required values
        prompt = first_row["prompt"]
        initial_text = first_row["current_text"]
        attacked_text = final_row["mutated_text"]
        mutation_num = final_row["mutation_num"]
        
        # Extract scores safely
        initial_quality_analysis = ast.literal_eval(first_row["quality_analysis"])
        final_quality_analysis = ast.literal_eval(final_row["quality_analysis"])
        
        initial_score = initial_quality_analysis.get("original_score_A", True)
        attacked_score = final_quality_analysis.get("original_score_B" if oracle == "InternLMOracle" else "quality_preserved", None)
        
        # Append the processed data
        results.append({
            "row_id": str(uuid.uuid4()),
            "group_id": group_id,
            "prompt": prompt,
            "initial_text": initial_text,
            "attacked_text": attacked_text,
            f"initial_{oracle}_score": initial_score,
            f"attacked_{oracle}_score": attacked_score
        })
    
    # Convert results to DataFrame and return
    return pd.DataFrame(results)


if __name__ == "__main__":

    # python -m attack.scripts.prepare_head2head

    from attack.utils import load_all_csvs

    w = "GPT4o_unwatermarked"
    m = "SentenceMutator"
            
    internlm_trace_df   = load_all_csvs("./attack/traces/annotated", w, m, "InternLMOracle")
    difforacle_trace_df = load_all_csvs("./attack/traces", w, m, "DiffOracle")
    print(f"Original InternLM + {w} + {m} attack trace: {internlm_trace_df.shape}")
    print(f"Original DiffOracle + {w} + {m} attack trace: {difforacle_trace_df.shape}")
    
    internlm_trace_df = process_dataframe(internlm_trace_df, "InternLMOracle").iloc[::]
    difforacle_trace_df = process_dataframe(difforacle_trace_df, "DiffOracle")
    print(f"Trimmed InternLM + {w} + {m} attack pairs: {internlm_trace_df.shape}")
    print(f"Trimmed DiffOracle + {w} + {m} attack pairs: {difforacle_trace_df.shape}")

    combined_df = pd.concat([internlm_trace_df, difforacle_trace_df], axis=0)
    print(f"Combined (InternLM + DiffOracle) + {w} + {m} attack pairs: {combined_df.shape}")

    combined_df.to_csv("./data/head2head/DiffOracle_vs_InternLM_smackdown_answers.csv", index=False)

    columns_to_keep = combined_df.columns[:5]
    combined_df = combined_df[columns_to_keep]
    combined_df["human_oracle_is_quality_preserved"] = ['Undecided'] * len(combined_df)
    combined_df["human_oracle_name"] = [''] * len(combined_df)
    combined_df.sample(frac=1).to_csv("./data/head2head/DiffOracle_vs_InternLM_smackdown_test.csv", index=False)