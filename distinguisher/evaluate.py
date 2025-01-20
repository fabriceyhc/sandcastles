# ./impossibility-watermark> CUDA_VISIBLE_DEVICES=7 python -m distinguisher.evaluate

from guidance import models
from distinguisher.models import (AggressiveSimple, SimpleGPT, ReasoningDistinguisher, SimpleDistinguisher, SimplestGPT)
from distinguisher.utils import process_attack_traces, extract_unique_column_value, get_id_tuples, split_dataframe
import pandas as pd
import os
import datasets
from dotenv import load_dotenv, find_dotenv
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

def distinguish_attacks(sd, df, length_of_df, prefix):
    o_str = extract_unique_column_value(df, "o_str")
    m_str = extract_unique_column_value(df, "m_str")
    w_str = extract_unique_column_value(df, "w_str")
    prompt_type = extract_unique_column_value(df, "prompt_type")
    ids = get_id_tuples(length_of_df)

    output_path=f"distinguisher/results/{prefix}_{o_str}_{w_str}_{m_str}_{prompt_type}_{sd.__class__.__name__}.csv"

    log.info(f"Oracle: {o_str}")
    log.info(f"Mutator: {m_str}")
    log.info(f"Watermarker: {w_str}")
    log.info(f"Prompt Type: {prompt_type}")

    if os.path.isfile(output_path):
        log.info(f"Path {output_path} already exists, returning...")
        return

    for i, (attack1_id, attack2_id, entropy) in enumerate(ids):
        log.info(f"Processing attack pair {i+1}/{len(ids)}")
        origins = {
            'A': AttackParser(None, df.iloc[attack1_id]['attack_data']),
            'B': AttackParser(None, df.iloc[attack2_id]['attack_data'])
        }
        sd.set_origin(origins['A'].get_response(), origins['B'].get_response())
        for origin, attack in origins.items():
            dataset = []
            for n in range(len(attack)):
                dataset.append({
                    "P": attack.get_nth(n),
                    "Num": n,
                    "Origin": origin,
                })
            dataset = dataset[-5:]
            dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
            dataset = sd.distinguish(dataset).to_pandas()
            dataset["prompt"] = attack.get_prompt()
            dataset["origin_A"] = origins['A'].get_response()
            dataset["origin_B"] = origins['B'].get_response()
            dataset["o_str"] = o_str
            dataset["w_str"] = w_str
            dataset["m_str"] = m_str
            dataset["prompt_type"] = prompt_type
            dataset["entropy"] = entropy
            dataset["attack1_id"] = attack1_id
            dataset["attack2_id"] = attack2_id
            dataset.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

def main():
    # Directory containing attack traces
    attack_trace_dir = "attack/traces/"

    # Filter to parse only files that evaluate as True
    filter_func = lambda x: "distinguisher" in x

    all_attacks = process_attack_traces(attack_trace_dir, filter_func)

    # TODO: Use the lambda function to filter the attacks
    # Reading all the traces then filtering takes unnecessarily long and too much RAM

    df = pd.DataFrame(all_attacks)

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

    load_dotenv(find_dotenv())
    chatgpt = models.OpenAI("gpt-4o-mini")

    length_of_df = 30
    # sd = AggressiveSimple(llm, distinguisher_persona, None, None)
    # sd = ReasoningDistinguisher(llm, distinguisher_persona, None, None)
    # sd = SimpleGPT(llm, distinguisher_persona, None, None)
    sd = SimplestGPT(llm, distinguisher_persona, None, None)
    # sd = SimplestGPT(chatgpt, None, None, None)

    unique_values = df['m_str'].unique()
    dfs_by_m_str = {value: df[df['m_str'] == value].copy() for value in unique_values}

    # log.info(f"DFs by m_str: {dfs_by_m_str}")

    further_split_dfs = {}
    for key, dataframe in dfs_by_m_str.items():
        split_dfs = split_dataframe(dataframe, length_of_df)

        # Add "prompt_type" column to each split
        split_dfs[0]['prompt_type'] = 'test'

        further_split_dfs[key] = {
            f'{key}_part1': split_dfs[0],
        }

    # log.info(f"Further split: {further_split_dfs}")

    # Verify each further split has length_of_df rows and order is preserved
    for _, splits in further_split_dfs.items():
        for part_key, part_df in splits.items():
            assert len(part_df) == length_of_df, f"{part_key} does not have {length_of_df} rows."

    parts = [part for splits in further_split_dfs.values() for part in splits.values()]
    log.info(f"Number of parts: {len(parts)}")

    for i, part in enumerate(parts):
        log.info(f"Processing part {i+1}/{len(parts)}")
        distinguish_attacks(sd, part, length_of_df, "llama3.1-8B")

if __name__ == "__main__":
    main()