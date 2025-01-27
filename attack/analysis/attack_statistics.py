import pandas as pd
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from scipy.optimize import curve_fit
from attack.utils import load_all_csvs

mutators = ["Document1StepMutator", "Document2StepMutator", "SentenceMutator", "SpanMutator", "EntropyWordMutator", "WordMutator"]
watermarks = ["GPT4o_unwatermarked", "Adaptive", "KGW", "SIR"]
watermark_thresholds = {
    "Adaptive" : 70,
    "KGW" : .50,
    "SIR": .20,
}
annotated_watermarks = ["KGW", "Adaptive", "SIR"]
quality_watermarks = ["KGW", "GPT4o_unwatermarked"]


def attack_statistics():
    for watermarker in annotated_watermarks:
        print(f"Analyzing {watermarker}")
        table = []
        for mutator in mutators:
            print(f"\tAnalyzing {mutator}")
            data = {"mutator": mutator,
                     "avg_pre_attack_watermark": 0, "avg_post_attack_watermark": 0, "avg_mutation_time": 0, 
                     "avg_pre_attack_quality": 0, "avg_post_attack_quality": 0, "avg_success_rate": 0,
                     }
            
            trace_df = load_all_csvs("./attack/traces/annotated", watermarker, mutator)

            if trace_df.empty:
                print(f"\t\tNo annotated data: Skipping {watermarker} {mutator}")
                continue
            elif "nicolai_quality" not in trace_df.columns:
                print(f"\t\tNo Nicolai Quality: Skipping {watermarker} {mutator}")
                continue
            
            trace_df['trace_num'] = (trace_df['step_num'] == -1).cumsum()

            num_instances = 0

            for run_num, attack in trace_df.groupby('trace_num'):
                num_instances += 1

                data["avg_mutation_time"] += attack["mutator_time"].mean()

                attack = attack[attack["watermark_score"] != 0]
                attack = attack[attack["quality_preserved"] == True]

                initial_row = attack[attack["step_num"] == -1]
                final_row = attack[attack.watermark_score == attack.watermark_score.min()]

                data["avg_pre_attack_watermark"] += initial_row["watermark_score"].iloc[0]
                data["avg_pre_attack_quality"] += initial_row["nicolai_quality"].iloc[0]
                data["avg_post_attack_watermark"] += final_row["watermark_score"].iloc[0]
                data["avg_post_attack_quality"] += final_row["nicolai_quality"].iloc[0]
                data["avg_success_rate"] += 1 if final_row["watermark_score"].iloc[0] <= watermark_thresholds[watermarker] else 0

            data["avg_pre_attack_watermark"] /= num_instances
            data["avg_pre_attack_quality"] /= num_instances
            data["avg_post_attack_watermark"] /= num_instances
            data["avg_post_attack_quality"] /= num_instances
            data["avg_mutation_time"] /= num_instances
            data["avg_success_rate"] /= num_instances

            table.append(data)
        table = pd.DataFrame(table)
        table.to_csv(f"./attack/analysis/csv/{watermarker}_statistics.csv", index=False)
                


            


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

def row_to_entropy(row):
    if "story" in row["prompt"]:
        match len(row["prompt"]):
            case 23: return 1
            case 49: return 2
            case 65: return 3
            case 98: return 4
            case 123: return 5
            case 209: return 6
            case 235: return 7
            case 263: return 8
            case 340: return 9
            case 492: return 10
    elif "essay" in row["prompt"]:
        match len(row["prompt"]):
            case 65: return 1
            case 107: return 2
            case 152: return 3
            case 194: return 4
            case 253: return 5
            case 285: return 6
            case 331: return 7
            case 341: return 8
            case 399: return 9
            case 471: return 10
    elif "news" in row["prompt"]:
        match len(row["prompt"]):
            case 30: return 1
            case 60: return 2
            case 131: return 3
            case 206: return 4
            case 272: return 5
            case 356: return 6
            case 442: return 7
            case 514: return 8
            case 566: return 9
            case 664: return 10


def num_steps(mutator):
    if "Document" in mutator:
        return 100
    if "Word" in mutator:
        return 1000
    if "Span" in mutator:
        return 250
    if "Sentence" in mutator:
        return 150


if __name__ ==  "__main__":

    # python -m attack.analysis.attack_statistics
    attack_statistics()