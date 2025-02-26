# python -m distinguisher.build_dataset_of_failed_distinguishes

import os
import glob
import pandas as pd

# Set the directory containing the CSV files
input_directory = "./distinguisher/results"

# Define a pattern to match specific CSV files (e.g., only files starting with "llama3")
file_pattern = os.path.join(input_directory, "llama3.1-70B-long*SimpleDistinguisher.csv")

# Output file
output_file = "./distinguisher/data/failed_distinguishes_for_llama3.1-70B-long.csv"

# List to collect data from all files
all_disagreements = []

total_distinguishes = 0

# Iterate through matched CSV files using glob
for file_path in glob.glob(file_pattern):
    print(f"Processing: {file_path}")
    
    # Load the CSV file
    df = pd.read_csv(file_path)

    total_distinguishes += len(df)
    
    # Assign group_ids based on each time Num == 0
    df['group_id'] = (df['Num'] == 0).cumsum()
    
    # Extract the parent rows (where Num=0) and their origin_A and origin_B
    parent_df = df[df['Num'] == 0][['group_id', 'origin_A', 'origin_B']]
    
    # Merge the parent information back into the main DataFrame
    df = df.merge(parent_df, on='group_id', how='left', suffixes=('_old', ''))
    df.drop(df.filter(regex='_old$').columns, axis=1, inplace=True)
    
    # Find rows where Choice, Flipped_Choice, and Origin have any disagreement
    disagreement_mask = (df['choice'] != df['flipped_choice']) | (df['choice'] != df['Origin']) | (df['flipped_choice'] != df['Origin'])
    disagreement_df = df[disagreement_mask]
    
    # Add the disagreement rows to the list
    all_disagreements.append(disagreement_df)

# Concatenate all results
if all_disagreements:
    final_df = pd.concat(all_disagreements, ignore_index=True)
    num_failed = len(final_df)
    
    final_df.to_csv(output_file, index=False)
    print(f"Saved output to {output_file}")

    print(f"Total Rows Distinguished: {total_distinguishes}")
    print(f"Total Rows Failed: {num_failed}")
    print(f"Percent Correct: {1 - (num_failed / total_distinguishes)}")
else:
    print("No disagreements found in any matching file.")