from guidance import models
from distinguisher.models import (AggressiveSimple, SimpleGPT)
from distinguisher.utils import all_attacks
import pandas as pd
import os
import datasets
from dotenv import load_dotenv, find_dotenv

class AttackParser():
    def __init__(self, file, df=None):
        if file is not None:
            df = pd.read_csv(file)
        df = df[(df['quality_preserved'] == True) & (df['length_issue'] == False)]
        end = df[df['mutation_num'] == df['mutation_num'].max()].tail(1)['step_num']
        df = df[df['step_num'] <= end.values[0]]
        df = df.drop_duplicates(subset=['mutation_num'], keep='last').reset_index(drop=True)

        # check for consistency
        for i, row in df.iterrows():
            if i == 0:
                continue
            assert row['current_text'] == df.loc[i-1, 'mutated_text'], f"Row {i} does not match previous row"
            assert i == row['mutation_num'], f"Row {i} does not match mutation_num"
        
        self.response = df.loc[0, 'current_text']
        self.df = df['mutated_text']
    
    def get_response(self):
        return self.response
    
    def __len__(self):
        return len(self.df)
    
    def get_nth(self, n):
        return self.df[n]
        
    
def get_file(entropy, output_num, attack_id):
    csv_file_directory = f"results/stationary_distribution/robustness_analysis/entropy_{entropy}/"
    first_perturbed_csv_filename = f"output_{output_num}/corpuses/attack_{attack_id}.csv"
    csv_file_path = os.path.join(csv_file_directory, first_perturbed_csv_filename)
    return csv_file_path


# load_dotenv(find_dotenv())
# chatgpt = models.OpenAI("gpt-4o-mini")

distinguisher_persona = \
"""
You are an expert in analyzing the similarity of responses.
You specialize in detecting whether one response has been derived from another by carefully analyzing the content and structure of the text.
Your expertise in linguistic analysis allows you to distinguish which responses are the most closely related.
Your goal is to provide a clear, concise, and accurate assessment of the provided instructions.
"""

llm = models.LlamaCpp(
    model="/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-8B-Instruct-q8_0.gguf",
    echo=False,
    n_gpu_layers=-1,
    n_ctx=2048
)

sd = AggressiveSimple(llm, distinguisher_persona, None, None)
for i in range(8):
    for entropy in range(10):
        attack1 = all_attacks[i*30+entropy*3]
        attack2 = all_attacks[i*30+entropy*3+1]
        origin_A = AttackParser(None, attack1["attack_data"])
        origin_B = AttackParser(None, attack2["attack_data"])
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
        dataset["o_str"] = attack1["o_str"]
        dataset["w_str"] = attack1["w_str"]
        dataset["m_str"] = attack1["m_str"]
        dataset["compare_against_original"] = attack1["compare_against_original"]
        dataset["entropy"] = entropy
        dataset["id"] = i*10+entropy
        output_path='distinguisher/results/stationary_distribution_full.csv'
        dataset.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

# ./impossibility-watermark> CUDA_VISIBLE_DEVICES=7 python -m distinguisher.evaluate