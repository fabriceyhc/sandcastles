# ./impossibility-watermark> CUDA_VISIBLE_DEVICES=7 python -m distinguisher.evaluate

from guidance import models
from distinguisher.models import (AggressiveSimple, SimpleGPT, ReasoningDistinguisher, SimpleDistinguisher)
from distinguisher.utils import all_attacks, extract_unique_column_value
import pandas as pd
import os
import datasets
from dotenv import load_dotenv, find_dotenv
import logging
from itertools import combinations

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)

# load_dotenv(find_dotenv())
# chatgpt = models.OpenAI("gpt-4o-mini")


def split_dataframe(df, chunk_size):
    return [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]

class AttackParser():
    def __init__(self, file, df=None):
        if file is not None:
            df = pd.read_csv(file)
        df = df[(df['quality_preserved'] == True) & (df['length_issue'] == False)]
        end = df[df['mutation_num'] == df['mutation_num'].max()].tail(1)['step_num']
        df = df[df['step_num'] <= end.values[0]]
        df = df.drop_duplicates(subset=['mutation_num'], keep='last').reset_index(drop=True)

        df.at[0, 'mutated_text'] = df.at[0, 'current_text']

        # check for consistency
        for i, row in df.iterrows():
            if i == 0:
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
                
        self.response = df.loc[0, 'current_text']
        self.df = df['mutated_text']
    
    def get_response(self):
        return self.response
    
    def __len__(self):
        return len(self.df)
    
    def get_nth(self, n):
        return self.df[n]

def get_id_tuples(num=10):
    """
    Generates unique pairs of numbers within groups of three from 1 to `num`.
    Each pair is annotated with the group number it belongs to.

    Parameters:
    num (int): The maximum number to generate pairs up to. Default is 30.

    Returns:
    list of tuples: Each tuple contains two numbers forming a pair and the group number.
    """
    # Initialize an empty list to store the pairs
    pairs = []
    
    # Use enumerate to keep track of the group number, starting at 0
    for group_num, start in enumerate(range(0, num, 3), start=1):
        # Define the current group of three numbers
        group = range(start, start + 3)
        
        # Generate all unique pairs within the group
        for pair in combinations(group, 2):
            # Check if the second number in the pair does not exceed 'num'
            if pair[1] <= num:
                # Append the pair along with the group number to the list
                pairs.append((*pair, group_num))
    
    return pairs

def distinguish_attacks(sd, df, length_of_df):
    o_str = extract_unique_column_value(df, "o_str")
    m_str = extract_unique_column_value(df, "m_str")
    w_str = extract_unique_column_value(df, "w_str")
    prompt_type = extract_unique_column_value(df, "prompt_type")
    ids = get_id_tuples(length_of_df)

    output_path=f"distinguisher/results/long_{o_str}_{w_str}_{m_str}_{prompt_type}_{sd.__class__.__name__}.csv"

    log.info(f"Oracle: {o_str}")
    log.info(f"Mutator: {m_str}")
    log.info(f"Watermarker: {w_str}")
    log.info(f"Prompt Type: {prompt_type}")

    if os.path.isfile(output_path):
        log.info(f"Path {output_path} already exists, returning...")
        return

    for attack1_id, attack2_id, entropy in ids:
        
        origin_A = AttackParser(None, df.iloc[attack1_id]['attack_data'])
        origin_B = AttackParser(None, df.iloc[attack2_id]['attack_data'])
        dataset = []
        for n in range(min(len(origin_A), len(origin_B))):
            dataset.append({
                "P": origin_A.get_nth(n),
                "Num": n,
                "Origin": "A",
            })
            dataset.append({
                "P": origin_B.get_nth(n),
                "Num": n,
                "Origin": "B",
            })
        sd.set_origin(origin_A.get_response(), origin_B.get_response())
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
        dataset = sd.distinguish(dataset).to_pandas()
        dataset["o_str"] = o_str
        dataset["w_str"] = w_str
        dataset["m_str"] = m_str
        dataset["prompt_type"] = prompt_type
        dataset["entropy"] = entropy
        dataset["attack1_id"] = attack1_id
        dataset["attack2_id"] = attack2_id
        dataset.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

def main():

    distinguisher_persona = \
    """
    You are an expert in analyzing the similarity of responses.
    You specialize in detecting whether one response has been derived from another by carefully analyzing the content and structure of the text.
    Your expertise in linguistic analysis allows you to distinguish which responses are the most closely related.
    Your goal is to provide a clear, concise, and accurate assessment of the provided instructions.
    # """

    llm = models.LlamaCpp(
        model="/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf",
        echo=False,
        n_gpu_layers=-1,
        n_ctx=4096
    )

    df = pd.DataFrame(all_attacks)

    # TODO: Adjust this to pick which attacks to distinguish.
    # df = df[df['w_str'].isin(['GPT4o_small'])] # GPT4o_unwatermarked, KGW, Adaptive,
    # df = df[df['w_str'].isin(['GPT4o_unwatermarked', 'KGW', 'Adaptive'])]
    # df = df[df['m_str'].isin(["WordMutator", "SpanMutator", "SentenceMutator"])]
    # df = df[df['m_str'].isin(["SentenceMutator"])] # Document1StepMutator, Document2StepMutator, DocumentMutator
    df = df[df['n_steps'] == '500']

    length_of_df = 30
    # sd = AggressiveSimple(llm, distinguisher_persona, None, None)
    # sd = ReasoningDistinguisher(llm, distinguisher_persona, None, None)
    sd = SimpleDistinguisher(llm, distinguisher_persona, None, None)

    unique_values = df['m_str'].unique()
    dfs_by_m_str = {value: df[df['m_str'] == value].copy() for value in unique_values}

    # log.info(f"DFs by m_str: {dfs_by_m_str}")

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

    # log.info(f"Further split: {further_split_dfs}")

    # Verify each further split has length_of_df rows and order is preserved
    for _, splits in further_split_dfs.items():
        for part_key, part_df in splits.items():
            assert len(part_df) == length_of_df, f"{part_key} does not have {length_of_df} rows."

    parts = [part for splits in further_split_dfs.values() for part in splits.values()]
    log.info(f"Number of parts: {len(parts)}")

    for part in parts:
        distinguish_attacks(sd, part, length_of_df)

if __name__ == "__main__":
    main()