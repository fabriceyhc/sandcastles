import pandas as pd
import matplotlib.pyplot as plt
from attack.utils import load_all_csvs
def assign_unique_group_ids(df):
    df['new_group'] = (df['step_num'] == 0).astype(int)
    df['group_id'] = df['new_group'].cumsum()
    return df

def get_support(df):
    return (df['step_num'] == -1).sum()

def get_successfully_attacked_support(df, watermark_threshold=0.0):
    df = get_successully_attacked_rows(df, watermark_threshold)
    return len(df['group_id'].unique())

def get_max_step_count(df):
    return df['step_num'].max()

def get_successully_attacked_rows(df, watermark_threshold=0.0):
    print("TEST TEST TEST", df[(df['group_id'] == 69) & (df['step_num'] == -1)]['entropy'])
    successful_df = df[(df['quality_preserved'] == True) & 
                       (df['watermark_score'] < watermark_threshold)]
    print("2 TEST2TEST", successful_df[(successful_df['group_id'] == 69) & (successful_df['step_num'] == -1)]['entropy'])
    successful_df = successful_df.sort_values(by='step_num').groupby('group_id').first().reset_index()
    return successful_df

def get_mean_step_count_to_break_watermark(df, watermark_threshold=0.0):
    successful_df = get_successully_attacked_rows(df, watermark_threshold)
    if successful_df.empty:
        return None
    return successful_df["step_num"].mean()

def get_attack_success_rate(df, watermark_threshold=0.0):
    success_count = get_successfully_attacked_support(df, watermark_threshold)
    divisor = get_support(df)
    print("success: ", success_count)
    print("total: ", divisor)
    # if success_count > divisor:
    #     print(get_successully_attacked_rows(df, watermark_threshold)['group_id'])
    #     print(df[df['step_num'] == -1]['group_id'])
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
        first_success_idx = group_df[(group_df['watermark_score'] < watermark_threshold)].index.min()
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

def prompt_to_entropy(prompt):
    if "story" in prompt:
        if len(prompt) == 23: return 1
        if len(prompt) == 49: return 2
        if len(prompt) == 65: return 3
        if len(prompt) == 98: return 4
        if len(prompt) == 123: return 5
        if len(prompt) == 209: return 6
        if len(prompt) == 235: return 7
        if len(prompt) == 263: return 8
        if len(prompt) == 340: return 9
        if len(prompt) == 492: return 10
    elif "essay" in prompt:
        if len(prompt) == 65: return 1
        if len(prompt) == 107: return 2
        if len(prompt) == 152: return 3
        if len(prompt) == 194: return 4
        if len(prompt) == 253: return 5
        if len(prompt) == 285: return 6
        if len(prompt) == 331: return 7
        if len(prompt) == 341: return 8
        if len(prompt) == 399: return 9
        if len(prompt) == 471: return 10
    elif "news" in prompt:
        if len(prompt) == 30: return 1
        if len(prompt) == 60: return 2
        if len(prompt) == 131: return 3
        if len(prompt) == 206: return 4
        if len(prompt) == 272: return 5
        if len(prompt) == 356: return 6
        if len(prompt) == 442: return 7
        if len(prompt) == 514: return 8
        if len(prompt) == 566: return 9
        if len(prompt) == 664: return 10

def row_to_entropy(row):
    if "story" in row["prompt"]:
        match len(row["prompt"]):
            case 23: return 1
            case 49: return 2
            case 65: return 3
            case 98: return 4
            case 123: return 5
            case 209: return 6
            case 235: return 7
            case 263: return 8
            case 340: return 9
            case 492: return 10
    elif "essay" in row["prompt"]:
        match len(row["prompt"]):
            case 65: return 1
            case 107: return 2
            case 152: return 3
            case 194: return 4
            case 253: return 5
            case 285: return 6
            case 331: return 7
            case 341: return 8
            case 399: return 9
            case 471: return 10
    elif "news" in row["prompt"]:
        match len(row["prompt"]):
            case 30: return 1
            case 60: return 2
            case 131: return 3
            case 206: return 4
            case 272: return 5
            case 356: return 6
            case 442: return 7
            case 514: return 8
            case 566: return 9
            case 664: return 10
def success_vs_entropy_per_mutator():
    watermarks = ["Adaptive", "KGW", "SIR"]
    mutators = [
        "DocumentMutator",
        "Document1StepMutator",
        "Document2StepMutator",
        "SentenceMutator",
        "SpanMutator",
        "WordMutator",
        "EntropyWordMutator",
    ]
    mutator_markers = ["s", "D", "X", "o", "^", "p", ".", "<", ">"]
    cutoffs = {
        "Adaptive" : 80,
        "KGW" : .50,
        "SIR": .20,

    }
    fig, axs = plt.subplots(2, 3, figsize=(32, 12))
    fig.suptitle(f"Attack Success Rate vs Entropy Level")
    # plt.subplots_adjust(left=.1, bottom=None, right=None, top=None, wspace=.3, hspace=None)
    axsl = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]]
    for idx, watermarker in enumerate(watermarks):
        print(f"Plotting {watermarker}")
        
        for idm, mutator in enumerate(mutators):
            print(f"\tPlotting for {mutator}")
            df_total = load_all_csvs("./attack/traces/annotated", watermarker, mutator)
            # for idx, row in df.iterrows():
            #     print(len(row['prompt']))
            #     print(prompt_to_entropy(row['prompt']))
            
            
            data = {}
            if df_total.empty:
                continue

            if "Adaptive" in watermarker:
                df_total = df_total[df_total['watermark_score'] != 0]
            df_total['entropy'] = df_total['prompt'].apply(prompt_to_entropy)
           
            for entropy in range(1,11):
                df = df_total[df_total["entropy"] == entropy]
                # print("TEST TEST TEST", df[(df['group_id'] == 69) & (df['step_num'] == -1)]['entropy'])
                print("graphing entropy: ", entropy)
                # mean_time = get_mean_total_time_for_successful_attacks(df, cutoff)
                success_rate = get_attack_success_rate(df, cutoffs[watermarker])
                # step_count = get_mean_step_count_to_break_watermark(df, cutoff)
                # score_change = get_mean_change_in_z_scores(df, cutoff)
                data[entropy] = success_rate
            axsl[idx].plot(data.keys(), data.values(), marker=mutator_markers[idm], label=mutator)
        axsl[idx].set_xlabel("Entropy Level")
        axsl[idx].set_ylabel("Attack Success Rate")
        axsl[idx].set_title(f"Attack Success Rate vs Entropy Level for {watermarker}")
        # Move legend outside the plot
        axsl[idx].legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    fig.subplots_adjust(right=0.8, wspace=0.8, hspace=0.3)
    fig.savefig(f"./attack/analysis/figs/success_vs_entropy.png")
if __name__ == "__main__":
    success_vs_entropy_per_mutator()

        