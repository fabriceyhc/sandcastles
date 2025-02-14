import glob
import pandas as pd
import os

oracle_str = "InternLMOracle"
watermark_str = "KGW"
mutator_str = "EntropyWordMutator"
base_dir = "./attack/traces/annotated"
def add_missing_step_minus_one(group):
    # Check if -1 is missing in this group
    if -1 not in group['step_num'].values:
        # Duplicate the row where step_num is 0 and modify it
        if 0 not in group['step_num'].values:
            return group
        row0 = group[group['step_num'] == 0].iloc[0].copy()
        row0['step_num'] = -1
        row0['mutation_num'] = -1

        row0['mutated_text'] = ''
        row0['mutated_text_len'] = -1
        row0['quality_analysis'] = {}
        # Create a DataFrame for the new row and concatenate it to the group
        new_row = pd.DataFrame([row0])
        group = pd.concat([group, new_row], ignore_index=True)
    # Sort the group so that the rows are in order by step_num
    return group.sort_values('step_num')

file_paths = f"{oracle_str}_{watermark_str}_{mutator_str}?*"
pattern = os.path.join(base_dir, file_paths)

# Get all matching CSV files
csv_files = glob.glob(pattern)


# Sort them so part1 < part2 < part3, etc. (if you have chunked files)
# extract the part number
part = lambda filename: int(filename.split("_part")[-1].split(".")[0]) if "_part" in filename else 0
# sort by the base name and then by the part number by converting the filename to a tuple
cmp = lambda filename: (filename.split("_part")[0], part(filename))
csv_files.sort(key=cmp)

# Read each CSV into a list of DataFrames
dataframes = []
for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    df = df.groupby('group_id', group_keys=False).apply(add_missing_step_minus_one)
    # print(df)
    df.to_csv(csv_path, index=False)


