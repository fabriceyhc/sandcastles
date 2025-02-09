import pandas as pd

def analyze_oracle_quality(answers_df, answer_key_df):
    # Merge the two dataframes on 'row_id'
    merged_df = answers_df.merge(answer_key_df, on="row_id", suffixes=("_answers", "_key"))
    
    # Group "Attacked Text Better" and "Tie" together
    merged_df["human_oracle_is_quality_preserved"] = merged_df["human_oracle_is_quality_preserved"].replace({"Attacked Text Better": "Tie"})
    
    # Count occurrences of each value in 'human_oracle_is_quality_preserved' per oracle type
    oracle_counts = merged_df.groupby("oracle")["human_oracle_is_quality_preserved"].value_counts().unstack(fill_value=0)
    
    # Rename columns
    oracle_counts = oracle_counts.rename(columns={
        "Tie": "Humans Agreed Quality was Preserved",
        "Initial Text Better": "Humans Disagreed Quality was Preserved"
    })
    
    # Calculate percentages
    oracle_percentages = oracle_counts.div(oracle_counts.sum(axis=1), axis=0) * 100
    
    # Combine counts and percentages into a single dataframe for display
    oracle_stats = oracle_counts.astype(str) + " (" + oracle_percentages.round(2).astype(str) + "%)"
    
    return oracle_stats

answers_df = pd.read_csv("./data/head2head/DiffOracle_vs_InternLM_smackdown_test_annotated.csv", encoding="ISO-8859-1")
answer_key_df = pd.read_csv("./data/head2head/DiffOracle_vs_InternLM_smackdown_answers.csv", encoding="ISO-8859-1")

# Apply function
df = analyze_oracle_quality(answers_df, answer_key_df)

print(df)

df.to_csv("./data/head2head/head2head_results.csv")
