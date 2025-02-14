from attack.utils import load_all_csvs
import pandas as pd
def get_successfully_attacked_support(df, watermark_threshold=0.0):
    df = get_successully_attacked_rows(df, watermark_threshold)
    return len(df['group_id'].unique())
def get_successully_attacked_rows(df, watermark_threshold=0.0):
    print("TEST TEST TEST", df[(df['group_id'] == 69) & (df['step_num'] == -1)]['entropy'])
    successful_df = df[(df['quality_preserved'] == True) & 
                       (df['watermark_score'] < watermark_threshold)]
    print("2 TEST2TEST", successful_df[(successful_df['group_id'] == 69) & (successful_df['step_num'] == -1)]['entropy'])
    successful_df = successful_df.sort_values(by='step_num').groupby('group_id').first().reset_index()
    return successful_df
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
        
df_total = load_all_csvs("./attack/traces/annotated", "Adaptive", "WordMutator")
df_total = df_total[df_total['watermark_score'] != 0]
print(df_total[df_total['step_num'] == -1]['prompt'][:50])
df_total['entropy'] = df_total['prompt'].apply(prompt_to_entropy)
df = df_total[df_total["entropy"] == 4]
print(df[df['step_num'] == 1000]['group_id'])
sucess = get_successully_attacked_rows(df, 80)
for idx, row in sucess.iterrows():
    print(row['group_id'], row['prompt'][0:50], prompt_to_entropy(row['prompt']))
print(sucess)

