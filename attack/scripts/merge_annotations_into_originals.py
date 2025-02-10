import pandas as pd

def merge_cols(df_to_keep, df_with_cols, cols_to_copy, cols_to_join):
    df_to_keep = df_to_keep.drop(columns=[col for col in cols_to_copy if col in df_to_keep.columns], errors='ignore')
    merged_df = df_to_keep.merge(df_with_cols[cols_to_join + cols_to_copy], on=cols_to_join, how='left') 
    return merged_df

# step_num,mutation_num,prompt,current_text,mutated_text,current_text_len,mutated_text_len,length_issue,quality_analysis,
# quality_preserved,watermark_detected,watermark_score,backtrack,total_time,mutator_time,oracle_time,group_id,armolm_quality,
# words_edited,perplexity,grammar_errors,offsetbias_quality

if __name__ == "__main__":

    # python -m attack.scripts.merge_annotations_into_originals

    import glob

    cols_to_join = ['step_num', 'mutation_num', 'prompt', 'current_text', 'mutated_text']
    cols_to_copy = ['watermark_score', 'armolm_quality', 'words_edited', 'perplexity', 'grammar_errors', 'offsetbias_quality']

    paths_to_keep = glob.glob("./attack/traces/InternLMOracle_SIR_Document2StepMutator_n-steps=100_attack_results*.csv")
    paths_with_cols = glob.glob("./attack/traces/annotated/InternLMOracle_SIR_Document2StepMutator_n-steps=100_attack_results*.csv")

    paths_to_keep = sorted(paths_to_keep)
    paths_with_cols = sorted(paths_with_cols)

    for path_to_keep, path_with_cols in zip(paths_to_keep, paths_with_cols):

        df_to_keep = pd.read_csv(path_to_keep)
        df_with_cols = pd.read_csv(path_with_cols)

        print(f"df_to_keep.shape: {df_to_keep.shape}")
        print(f"df_with_cols.shape: {df_with_cols.shape}")

        merged_df = merge_cols(df_to_keep, df_with_cols, cols_to_copy, cols_to_join)
        print(merged_df.columns)
        print(merged_df)

        merged_df.to_csv(path_to_keep.replace(".csv", "_new.csv"), index=False)