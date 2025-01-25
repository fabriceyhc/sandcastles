# ./impossibility-watermark> CUDA_VISIBLE_DEVICES=7 python -m distinguisher.evaluate

from distinguisher.models import (SimpleDistinguisher, AggressiveSimple, SimpleGPT, SimplestGPT, LogicGPT, LogicSimple, LogicSimplest)
from distinguisher.utils import process_attack_traces, extract_unique_column_value, get_id_tuples, split_dataframe, get_model, parse_filename
import pandas as pd
import os
import datasets
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)


class AttackParser():
    def __init__(self, file, df=None):
        if file is not None:
            df = pd.read_csv(file)
        df = df[(df['quality_preserved'] == True) & (df['length_issue'] == False)]
        end = df[df['mutation_num'] == df['mutation_num'].max()].tail(1)['step_num']
        df = df[df['step_num'] <= end.values[0]]
        df = df.drop_duplicates(subset=['mutation_num'], keep='last').reset_index(drop=True)

        if df.at[0, 'mutated_text'] != df.at[0, 'mutated_text']: # NaN check
            df.at[0, 'mutated_text'] = df.at[0, 'current_text']

        # check for consistency
        for i, row in df.iterrows():
            if i == 0:
                assert row['step_num'] == -1, "First row is not step -1"
                continue  # Skip the first row

            # Check current_text matches mutated_text from the previous row
            prev_row = df.loc[i-1]
            if row['current_text'] != prev_row['mutated_text']:
                log.error(f"Row {i} current_text mismatch. Current: {row.to_dict()}, Previous: {prev_row.to_dict()}")
                assert row['current_text'] == prev_row['mutated_text'], f"Row {i} does not match previous row"

            # Check row index matches mutation_num
            if (i - 1) != row['mutation_num']:
                log.error(f"Row {i} mutation_num mismatch. Current: {row.to_dict()}, Previous: {prev_row.to_dict()}")
                assert i == row['mutation_num'], f"Row {i} does not match mutation_num"
                
        self.prompt = df.loc[0, 'prompt']
        self.response = df.loc[0, 'current_text']
        self.df = df['mutated_text']

    def get_prompt(self):
        return self.prompt
    
    def get_response(self):
        return self.response
    
    def __len__(self):
        return len(self.df)
    
    def get_nth(self, n):
        return self.df[n]

def distinguish_attacks(sd, df, length_of_df, prefix, subsample=1):
    o_str = extract_unique_column_value(df, "o_str")
    m_str = extract_unique_column_value(df, "m_str")
    w_str = extract_unique_column_value(df, "w_str")
    if m_str == "SentenceMutator":
        subsample *= 3
    if m_str == "SpanMutator":
        subsample *= 5
    if m_str == "WordMutator":
        subsample *= 20
    prompt_type = extract_unique_column_value(df, "prompt_type")
    ids = get_id_tuples(length_of_df)

    output_path=f"distinguisher/results/{prefix}_{o_str}_{w_str}_{m_str}_{prompt_type}_{sd.__class__.__name__}.csv"

    log.info(f"Oracle: {o_str}")
    log.info(f"Mutator: {m_str}")
    log.info(f"Watermarker: {w_str}")
    log.info(f"Prompt Type: {prompt_type}")
    log.info(f"Distinguisher: {sd.__class__.__name__}")

    if os.path.isfile(output_path):
        log.info(f"Path {output_path} already exists, returning...")
        return

    distinguish_count = 0

    for i, (attack1_id, attack2_id, entropy) in enumerate(ids):
        origins = {
            'A': AttackParser(None, df.iloc[attack1_id]['attack_data']),
            'B': AttackParser(None, df.iloc[attack2_id]['attack_data'])
        }
        if sd is not None:
            sd.set_origin(origins['A'].get_response(), origins['B'].get_response())
            log.info(f"Processing attack pair {i+1}/{len(ids)}")
        for origin, attack in origins.items():
            dataset = []
            for n in range(len(attack)):
                dataset.append({
                    "P": attack.get_nth(n),
                    "Num": n,
                    "Origin": origin,
                })
            # dataset = dataset[-5:] # TODO: Remove this line when running on full dataset
            dataset = dataset[::subsample]

            if sd is None:
                distinguish_count += len(dataset)
                continue
            log.info(f"Counter: {distinguish_count}")
            distinguish_count += len(dataset)
            dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
            dataset = sd.distinguish(dataset).to_pandas()
            dataset["prompt"] = attack.get_prompt()
            dataset["o_str"] = o_str
            dataset["w_str"] = w_str
            dataset["m_str"] = m_str
            dataset["prompt_type"] = prompt_type
            dataset["entropy"] = entropy
            dataset["attack1_id"] = attack1_id
            dataset["attack2_id"] = attack2_id
            dataset["origin_A"] = ""
            dataset["origin_B"] = ""
            dataset.iloc[0, dataset.columns.get_loc("origin_A")] = origins['A'].get_response()
            dataset.iloc[0, dataset.columns.get_loc("origin_B")] = origins['B'].get_response()
            dataset.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    log.info(f"Finished Counter: {distinguish_count}")
    return distinguish_count

# Need to adjust for different datasets
def split_to_parts(df, length_of_df):
    unique_values = df['m_str'].unique()
    dfs_by_m_str = {value: df[df['m_str'] == value].copy() for value in unique_values}

    further_split_dfs = {}
    for key, dataframe in dfs_by_m_str.items():
        split_dfs = split_dataframe(dataframe, length_of_df)

        # Add "prompt_type" column to each split
        split_dfs[0]['prompt_type'] = 'paris'
        split_dfs[1]['prompt_type'] = 'space'
        split_dfs[2]['prompt_type'] = 'news'

        further_split_dfs[key] = {
            f'{key}_part1': split_dfs[0],
            f'{key}_part2': split_dfs[1],
            f'{key}_part3': split_dfs[2]
        }

    # Verify each further split has length_of_df rows and order is preserved
    for _, splits in further_split_dfs.items():
        for part_key, part_df in splits.items():
            assert len(part_df) == length_of_df, f"{part_key} does not have {length_of_df} rows."

    parts = [part for splits in further_split_dfs.values() for part in splits.values()]
    return parts

def main():
    # Directory containing attack traces
    attack_trace_dir = "attack/traces/"

    # Filter to parse only files that evaluate as True
    def filter_func(filename):
        if not filename.endswith(".csv"):
            return False
        if "annotated" in filename:
            return False
        o_str, w_str, m_str, n_steps = parse_filename(filename)
        # print(filename, w_str, m_str)
        # return w_str in ["GPT4o_unwatermarked", "KGW", "Adaptive"] and m_str in ["WordMutator", "SpanMutator", "SentenceMutator"]
        return w_str in ["Adaptive"] and m_str in ["WordMutator", "SpanMutator", "SentenceMutator"]

    for filename in filter(filter_func, os.listdir(attack_trace_dir)):
        log.info(filename)
    
    all_attacks = process_attack_traces(attack_trace_dir, filter_func)

    # TODO: Use the lambda function to filter the attacks
    # Reading all the traces then filtering takes unnecessarily long and too much RAM

    df = pd.DataFrame(all_attacks)
    length_of_df = 30
    parts = split_to_parts(df, length_of_df)
    log.info(f"Number of parts: {len(parts)}")

    total_len = 0
    for i, part in enumerate(parts):
        log.info(f"Processing part {i+1}/{len(parts)}")
        part_len = distinguish_attacks(None, part, length_of_df, "dryrun", 4)
        total_len += part_len
        log.info(f"Part length: {part_len}\n")
    log.info(f"Total length: {total_len}")

    llm, distinguisher_persona = get_model("llama3.1-70B")

    for sd in [SimpleDistinguisher(llm, distinguisher_persona)]:
        for i, part in enumerate(parts):
            log.info(f"Processing part {i+1}/{len(parts)}")
            distinguish_attacks(sd, part, length_of_df, "llama3.1-70B-full", 4)

if __name__ == "__main__":
    main()