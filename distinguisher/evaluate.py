# ./impossibility-watermark> CUDA_VISIBLE_DEVICES=0,1,2 python -m distinguisher.evaluate

from distinguisher.models import (SimpleDistinguisher, AggressiveSimple, SimpleGPT, SimplestGPT, LogicGPT, LogicSimple, LogicSimplest)
from distinguisher.utils import get_attack_traces, extract_unique_column_value, get_id_tuples, get_model, prompt_to_entropy, prompt_to_type
from itertools import combinations
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
        df['step_num'] = pd.to_numeric(df['step_num'], errors='coerce')
        df = df[(df['quality_preserved'] == True) & (df['length_issue'] == False)]
        end = df[df['mutation_num'] == df['mutation_num'].max()].tail(1)['step_num']
        df = df[df['step_num'] <= end.values[0]]
        df = df.drop_duplicates(subset=['mutation_num'], keep='last').reset_index(drop=True)

        if df.at[0, 'mutated_text'] != df.at[0, 'mutated_text']: # NaN check
            df.at[0, 'mutated_text'] = df.at[0, 'current_text']

        # check for consistency
        for i, row in df.iterrows():
            if i == 0:
                # assert row['step_num'] == -1, "First row is not step -1"
                continue  # Skip the first row

            # # Check current_text matches mutated_text from the previous row
            # prev_row = df.loc[i-1]
            # if row['current_text'] != prev_row['mutated_text']:
            #     log.error(f"Row {i} current_text mismatch. Current: {row['step_num']}, Previous: {prev_row['step_num']}")
            #     assert row['current_text'] == prev_row['mutated_text'], f"Row {i} does not match previous row"

            # # Check row index matches mutation_num
            # if (i - 1) != row['mutation_num']:
            #     log.error(f"Row {i} mutation_num mismatch. Current: {row['step_num']}, Previous: {prev_row['step_num']}")
            #     assert i == row['mutation_num'], f"Row {i} does not match mutation_num"
                
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

def distinguish_attacks(sd, df, prefix, subsample, groups=10):
    o_str = extract_unique_column_value(df, "o_str")
    m_str = extract_unique_column_value(df, "m_str")
    w_str = extract_unique_column_value(df, "w_str")
    if m_str in ["DocumentMutator"]:
        subsample = max(subsample//2, 1)
    if m_str == "SentenceMutator":
        subsample *= 3
    if m_str == "SpanMutator":
        subsample *= 5
    if m_str in ["WordMutator", "EntropyWordMutator"]:
        subsample *= 20
    prompt_type = extract_unique_column_value(df, "prompt_type")
    ids = get_id_tuples(df, groups)

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
    tot_pairs = 0

    for entropy in range(1, groups+1):
        pairs = ids[entropy]
        tot_pairs += len(pairs)
        if len(pairs) == 0:
            log.warning(f"Entropy level {entropy}: no pairs found")
            continue
        if len(pairs) == 1:
            log.warning(f"Entropy level {entropy}: only one pair found")
            continue
        for i, (attack1_id, attack2_id) in enumerate(pairs):
            origins = {
                'A': AttackParser(None, df[df['attack_num'] == attack1_id].iloc[0]['attack_data']),
                'B': AttackParser(None, df[df['attack_num'] == attack2_id].iloc[0]['attack_data'])
            }
            if sd is not None:
                sd.set_origin(origins['A'].get_response(), origins['B'].get_response())
                log.info(f"Processing attack pair {i+1}/{len(pairs)}")
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
    return distinguish_count, tot_pairs

# Need to adjust for different datasets
def split_to_parts(df):
    df['prompt_type'] = df['attack_data'].apply(lambda x: prompt_to_type(x['prompt'].values[0]))
    df['entropy'] = df['attack_data'].apply(lambda x: prompt_to_entropy(x['prompt'].values[0]))
    # split dfs based on prompt type
    unique_types = df['prompt_type'].unique()
    split_dfs = [df[df['prompt_type'] == t] for t in unique_types]
    # assert len(split_dfs) == 3, f"Expected 3 splits, got {len(split_dfs)}"
    return split_dfs

def evaluate(distinguisher, parts, prefix, subsample):
    """
    Given a distinguisher and a list of parts, evaluate the distinguisher on each part.
    Parts should be a list of dataframes.
    Each part is defined by a 4-tuple (oracle, watermark type, mutator, prompt type).
    Each part is itself a list of attack traces. Here we use 30 per part, (10 entropy levels, 3 per).
    Within each entropy level, each pair is distinguished i.e. A1 against (A1, A2) and (A1, A3), etc.
    This gives 60 experiments per part.
    """
    total_len = 0
    for i, part in enumerate(parts):
        log.info(f"Processing part {i+1}/{len(parts)}")
        part_len, pairs = distinguish_attacks(distinguisher, part, prefix, subsample)
        total_len += part_len
        log.info(f"Distinguishes per experiment: {part_len/pairs/2:.2f}\n")
    log.info(f"Total length: {total_len}")

def main():
    oracle = "InternLMOracle"

    # Select watermark types and mutators to evaluate
    # experiments = {
    #     "GPT4o_unwatermarked": ["EntropyWordMutator", "Document1StepMutator", "Document2StepMutator"],
    #     "Adaptive": ["EntropyWordMutator", "Document2StepMutator"],
    #     "KGW": ["EntropyWordMutator", "Document1StepMutator", "Document2StepMutator"],
    #     "SIR": ['DocumentMutator'],
    # }
    # experiments = {
    #     "Adaptive": [ "Document1StepMutator"],
    #     "SIR": ['Document1StepMutator', 'Document2StepMutator'],
    # }
    experiments = {
        "Adaptive": ["DocumentMutator"],
    }

    # Construct parts
    parts = []
    for watermark_type in experiments:
        for mutator in experiments[watermark_type]:
            attack_trace_dir = "./attack/traces/"
            df = get_attack_traces(attack_trace_dir, oracle, watermark_type, mutator)
            if len(df) == 0:
                attack_trace_dir = "./attack/traces/annotated/"
                df = get_attack_traces(attack_trace_dir, oracle, watermark_type, mutator)
                if len(df) == 0:
                    log.warning(f"No attack traces found for {oracle}, {watermark_type}, {mutator}. Skipping.")
                    continue
            log.info(f"Loaded {len(df)} attack traces for {oracle}, {watermark_type}, {mutator} from {attack_trace_dir}")
            split_df = split_to_parts(df)
            if len(split_df) < 3:
                log.warning(f"Expected 3 parts for {oracle}, {watermark_type}, {mutator}, got {len(split_df)}: {','.join([x['prompt_type'].unique()[0] for x in split_df])}")
            parts.extend(split_df)
    log.info(f"Total parts: {len(parts)}")

    # Evaluate distinguishers
    evaluate(None, parts, "dryrun", 4)

    llm, distinguisher_persona = get_model("llama3.1-70B")
    for sd in [SimpleDistinguisher(llm, distinguisher_persona)]:
        evaluate(sd, parts, "llama3.1-70B-full", 4)

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=5,7 python -m distinguisher.evaluate

    main()
