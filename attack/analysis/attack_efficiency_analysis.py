import pandas as pd
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

mutators = ["Document1StepMutator", "Document2StepMutator", "SentenceMutator", "SpanMutator", "EntropyWordMutator", "WordMutator"]
watermarks = ["GPT4o_unwatermarked", "Llama-3.1_unwatermarked", "Adaptive", "UMD"]


# How to split across mutator, watermark scheme, and entropy:
# Watermark scheme: 4 different options. Should probably create separate plots for each
# Mutator: 6 different mutators. Use different markers or linestyles?
#   circle, down triangle, square, plus, diamond, cross
mutator_markers = ["s", "D", "X", "o", "^", "p"]
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html
# Entropy: ranges 1-10. Use a color range?
entropy_colors = cm.plasma(np.linspace(0,1,10))


def plot_sliding_window_success_rates():
    # dimensions to consider: watermark type, mutator, entropy level

    for idx, watermarker in enumerate(watermarks):
        print(f"Plotting {watermarker}")
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        axs = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]]
        plt.suptitle(watermarker)

        for idm, mutator in enumerate(mutators):
            print(f"\tPlotting for {mutator}")
            traces = sorted(glob.glob(f"./attack/traces/?*{watermarker}_{mutator}?*"))
            
            if "Document" in mutator:
                window_size = 10
            if "Word" in mutator:
                window_size = 100
            if "Span" in mutator:
                window_size = 25
            if "Sentence" in mutator:
                window_size = 15
            
            final_data = [pd.DataFrame(columns=["step_num", "quality_preserved"]) for e in range(1,11)]
            num_traces_per_entropy = [0] * 10

            for trace in traces:
                trace_df = pd.read_csv(trace)
                # NOTE: This will contain trace data for multiple attack runs, and not necessarily from step 0/-1!!!!
                for prompt, group in trace_df.groupby('prompt'):
                    # Thank ChatGPT for this genius idea 
                    group['trace_num'] = (group['step_num'] == -1).cumsum()

                    for run_num, attack in group.groupby('trace_num'):
                        attack["quality_preserved"] = attack["quality_preserved"].rolling(window=window_size, min_periods=1).mean()
                        entropy = prompt_to_entropy(prompt)
                        color = entropy_colors[entropy-1]

                        final_data[entropy-1] = pd.concat([final_data[entropy-1], attack[["step_num", "quality_preserved"]]]).groupby("step_num", as_index=False).sum()
                        #axs[idm].plot(attack["step_num"], attack["quality_preserved"], alpha=.8, color=color)
                        num_traces_per_entropy[entropy-1] += 1
                        
            for i, entropy_df in enumerate(final_data):
                color = entropy_colors[i]
                axs[idm].plot(entropy_df["step_num"], entropy_df["quality_preserved"]/num_traces_per_entropy[i], alpha=.8, color=color)

            axs[idm].set_xlabel("Step Number")
            axs[idm].set_ylabel(f"Rolling Success Rate (window size = {window_size})")
            axs[idm].set_ylim(-.05, 1.05)
            axs[idm].set_title(mutator)

            # TODO: set legend for colors 
            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=1, vmax=10))
            cbar = plt.colorbar(sm, ax=axs[idm], orientation='vertical')
            cbar.set_label('Entropy Level')
        
        plt.savefig(f"./attack/analysis/rolling_success_{watermarker}.png")


    

def plot_oracle_failure_rates_on_human_data():
    # this data already exists somewhere
    pass

def plot_quality_degredation_vs_zscore():
    # split across watermark, mutator, entropy
    pass

def plot_estimated_watermark_breaking():
    # split across watermark, mutator, entropy?
    pass


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

if __name__ ==  "__main__":
    plot_sliding_window_success_rates()
             
