import pandas as pd
from matplotlib import pyplot as plt

tdf = pd.read_csv(f'results/stationary_distribution_full.csv')

def summary(df, prefix=""):
    df[f'{prefix}correct'] = (df['Origin'] == df[f'{prefix}choice']).astype(float)
    df[f'{prefix}flipped_correct'] = (df['Origin'] == df[f'{prefix}flipped_choice']).astype(float)
    df[f'{prefix}avg_correct'] = (df[f'{prefix}correct']+df[f'{prefix}flipped_correct'])/2
    print(f"Correct: {df[f'{prefix}correct'].mean()}, Flipped Correct: {df[f'{prefix}flipped_correct'].mean()}, Avg Correct: {df[f'{prefix}avg_correct'].mean()}")


experiments = 8
prompts = 10
trials = experiments * prompts

n_rows = [1,3]
n_cols = [2,3,4,5]

fig, axes = plt.subplots(len(n_rows), len(n_cols), figsize=(40, 30))

tdf = pd.read_csv(f'results/test3.csv')

def summary(df, prefix=""):
    df[f'{prefix}correct'] = (df['Origin'] == df[f'{prefix}choice']).astype(float)
    df[f'{prefix}flipped_correct'] = (df['Origin'] == df[f'{prefix}flipped_choice']).astype(float)
    df[f'{prefix}avg_correct'] = (df[f'{prefix}correct']+df[f'{prefix}flipped_correct'])/2
    # print(f"Correct: {df[f'{prefix}correct'].mean()}, Flipped Correct: {df[f'{prefix}flipped_correct'].mean()}, Avg Correct: {df[f'{prefix}avg_correct'].mean()}")

for i in tdf['id'].unique():
    t_row = i // prompts
    t_col = i % prompts
    if t_row not in n_rows or t_col not in n_cols:
        continue
    row = n_rows.index(t_row)
    col = n_cols.index(t_col)
    df = tdf[tdf['id'] == i].copy(deep=True)
    print(i)
    summary(df)
    tmp1 = df[df['Origin'] == 'A']['avg_correct'].reset_index(drop=True)
    tmp2 = df[df['Origin'] == 'B']['avg_correct'].reset_index(drop=True)
    data = (tmp1+tmp2)/2
    window = 10
    rolling_mean = data.rolling(window=window).mean()[window-1:].reset_index()
    axes[row, col].plot(rolling_mean, color='orange')
    if row == 0:
        axes[row, col].set_title(f'difficulty={df.iloc[0]["entropy"]}', fontsize=20, pad=20)
    if col == 0:
        axes[row, col].set_ylabel(f'{df.iloc[0]["m_str"]}, origin={df.iloc[0]["compare_against_original"]}', fontsize=20, labelpad=40)
    axes[row, col].axhline(y=0.5, color='red', linestyle='--')
    axes[row,col].set_ylim(-0.1, 1.1)
    axes[row,col].set_xlim(0, None)

plt.tight_layout()
plt.savefig('results/test2.png')