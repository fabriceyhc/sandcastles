# RUN: CUDA_VISIBLE_DEVICES=4 python -m attack.attack_metrics

import pandas as pd
import matplotlib.pyplot as plt
from attack.utils import load_all_csvs
def assign_unique_group_ids(df):
    df['new_group'] = (df['step_num'] == 0).astype(int)
    df['group_id'] = df['new_group'].cumsum()
    return df

def get_support(df):
    return (df['step_num'] == 0).sum()

def get_successfully_attacked_support(df, watermark_threshold=0.0):
    df = get_successully_attacked_rows(df, watermark_threshold)
    return len(df['group_id'].unique())

def get_max_step_count(df):
    return df['step_num'].max()

def get_successully_attacked_rows(df, watermark_threshold=0.0):
    successful_df = df[(df['quality_preserved'] == True) & 
                       (df['watermark_score'] < watermark_threshold)]
    successful_df = successful_df.sort_values(by='step_num').groupby('group_id').first().reset_index()
    return successful_df

def get_mean_step_count_to_break_watermark(df, watermark_threshold=0.0):
    successful_df = get_successully_attacked_rows(df, watermark_threshold)
    if successful_df.empty:
        return None
    return successful_df["step_num"].mean()

def get_attack_success_rate(df, watermark_threshold=0.0):
    success_count = len(get_successully_attacked_rows(df, watermark_threshold))
    divisor = get_support(df)
    if divisor == 0:
        return None
    success_rate = success_count / divisor
    return success_rate

def get_mean_mutation_time(df):
    return df["mutator_time"].mean()

def get_mean_oracle_time(df):
    return df["oracle_time"].mean()

def get_mean_change_in_z_scores(df, watermark_threshold=0.0):
    quality_preserved_df = df[df['quality_preserved'] == True].copy()
    quality_preserved_df = quality_preserved_df.sort_values(by='step_num')
    z_score_changes = []
    for group_id, group_df in quality_preserved_df.groupby('group_id'):
        first_success_idx = group_df[group_df['watermark_score'] < watermark_threshold].index.min()
        if pd.notna(first_success_idx):
            group_df = group_df.loc[:first_success_idx]  # Consider only steps before the threshold
        else:
            group_df = group_df  # If no success, consider the whole group
        group_df['watermark_score_change'] = group_df['watermark_score'].diff()
        z_score_changes.extend(group_df['watermark_score_change'].dropna().tolist())
    if z_score_changes:
        mean_change = sum(z_score_changes) / len(z_score_changes)
    else:
        mean_change = None
    return mean_change

def get_mean_total_time_for_successful_attacks(df, watermark_threshold=0.0):
    successful_df = get_successully_attacked_rows(df, watermark_threshold)
    successful_group_ids = successful_df['group_id'].unique()  
    total_times = []
    for group_id in successful_group_ids:
        group_df = df[df['group_id'] == group_id]
        first_success_idx = group_df[group_df['watermark_score'] < watermark_threshold].index.min()
        total_time_before_success = group_df.loc[:first_success_idx, 'total_time'].sum()
        total_times.append(total_time_before_success)
    if total_times:
        mean_total_time = sum(total_times) / len(total_times)
    else:
        mean_total_time = None
    return mean_total_time

def get_original_and_final_text_comparisons(df, watermark_threshold=0.0):
    original_text_df = df[df['step_num'] == 0][['group_id', 'prompt', 'current_text']]
    final_text_df = get_successully_attacked_rows(df, watermark_threshold)[['group_id', 'mutated_text']]
    comparison_df = pd.merge(original_text_df, final_text_df, on='group_id', how='inner')
    comparison_df.rename(columns={'current_text': 'original_text', 'mutated_text': 'final_mutated_text'}, inplace=True)
    return comparison_df


if __name__ == "__main__":

    import os
    import glob 

    mutators = [
    "DocumentMutator",
    "Document1StepMutator",
    "Document2StepMutator",
    "SentenceMutator",
    "SpanMutator",
    "WordMutator",
    "EntropyWordMutator",
]
    watermarks = ["Adaptive", "KGW"]

    cutoffs = {
        "Adaptive" : [20, 30, 40, 50, 60, 70],
        "KGW" :  [0, 0.5, 1, 2, 3, 6]
    }
    labels = ["Average Time to Success", "Average Attack Success", "Average Steps to Success", "Average score change"]
    titles = ["Time to Success vs Cutoff", "Attack Success Rate vs Cutoff", "Steps to Sucess vs Cutoff", "Score change vs Cutoff"]
  

    mutator_markers = ["s", "D", "X", "o", "^", "p"]
    for idx, watermarker in enumerate(watermarks):
        print(f"Plotting {watermarker}")
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        # plt.subplots_adjust(left=.1, bottom=None, right=None, top=None, wspace=.3, hspace=None)
        axsl = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
        for idm, mutator in enumerate(mutators):
            print(f"\tPlotting for {mutator}")
            lmean_time = []
            lsuccess_rate = []
            lstep_count = []
            lscore_change = []
            total = [lmean_time, lsuccess_rate, lstep_count, lscore_change]
            df = load_all_csvs("./attack/traces", watermarker, mutator)
            if df.empty:
                continue
            for cutoff in cutoffs[watermarker]:
                    try:
                        mean_time = get_mean_total_time_for_successful_attacks(df, cutoff)
                        success_rate = get_attack_success_rate(df, cutoff)
                        step_count = get_mean_step_count_to_break_watermark(df, cutoff)
                        score_change = get_mean_change_in_z_scores(df, cutoff)
                        lmean_time.append(mean_time)
                        lsuccess_rate.append(success_rate)
                        lstep_count.append(step_count)
                        lscore_change.append(score_change)
                    except:
                        pass
            if len(total[0]) == 0:
                continue
            for i in range(4):
                # print(cutoffs[watermarker])
                # print(total[i])
                axsl[i].plot(cutoffs[watermarker], total[i], marker=mutator_markers[idm], label=mutator)
                axsl[i].set_xlabel("Watermark Detection Score")
                axsl[i].set_ylabel(labels[i])
                axsl[i].set_title(titles[i])
                axsl[i].legend(loc="upper right")
        fig.savefig(f"./attack/{watermarker}.png")