import pandas as pd
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from attack.utils import load_all_csvs

mutators = ["Document1StepMutator", "Document2StepMutator", "SentenceMutator", "SpanMutator", "EntropyWordMutator", "WordMutator"]
watermarks = ["GPT4o_unwatermarked", "Adaptive", "KGW", "SIR"]
watermark_thresholds = {
    "GPT4o_unwatermarked", 
    "Llama-3.1_unwatermarked", 
    "Adaptive", 
    "KGW", 
    "SIR"
}
annotated_watermarks = ["KGW", "Adaptive", "SIR"]
quality_watermarks = ["KGW", "GPT4o_unwatermarked"]


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
            
            window_size = num_steps(mutator)//10

            axs[idm].set_xlabel("Step Number")
            axs[idm].set_ylabel(f"Rolling Success Rate (window size = {window_size})")
            axs[idm].set_ylim(-.05, 1.05)
            axs[idm].set_title(mutator)

            # set legend for colors 
            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=1, vmax=10))
            cbar = plt.colorbar(sm, ax=axs[idm], orientation='vertical')
            cbar.set_label('Entropy Level')
            
            final_data = [pd.DataFrame(columns=["step_num", "quality_preserved"]) for e in range(1,11)]
            num_traces_per_entropy = [0] * 10

            trace_df = load_all_csvs("./attack/traces", watermarker, mutator, annotated=False)
            if trace_df.empty:
                print(f"\t\tEmpty dataframe! Skipping {watermarker}, {mutator}")
                continue
            trace_df['trace_num'] = (trace_df['step_num'] == -1).cumsum()

            for run_num, attack in trace_df.groupby('trace_num'):
                attack["quality_preserved"] = attack["quality_preserved"].rolling(window=window_size, min_periods=1).mean()
                entropy = prompt_to_entropy(attack["prompt"].iloc[0])
                color = entropy_colors[entropy-1]

                final_data[entropy-1] = pd.concat([final_data[entropy-1], attack[["step_num", "quality_preserved"]]]).groupby("step_num", as_index=False).sum()
                num_traces_per_entropy[entropy-1] += 1

                        
            for i, entropy_df in enumerate(final_data):
                color = entropy_colors[i]
                axs[idm].plot(entropy_df["step_num"], entropy_df["quality_preserved"]/num_traces_per_entropy[i], alpha=.8, color=color)

        
        plt.savefig(f"./attack/analysis/figs/rolling_success_{watermarker}.png")




def plot_quality_degredation_vs_zscore():
    for idx, watermarker in enumerate(annotated_watermarks):
        print(f"Plotting {watermarker}")
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        axs = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]]
        plt.suptitle(watermarker)

        for idm, mutator in enumerate(mutators):
            print(f"\tPlotting for {mutator}")
            traces = sorted(glob.glob(f"./attack/traces/?*{watermarker}_{mutator}?*annotated?*"))
            
            # key is (prompt, trace_num) -> data
            initial_data = {}
            final_data = {}

            trace_num_from_prompt = defaultdict(int)

            for trace in traces:
                trace_df = pd.read_csv(trace)

                if "internlm_quality" not in trace_df.columns or "watermark_score" not in trace_df.columns:
                    print(f"Skipping {trace}") 
                    continue
                
                if "Adaptive" in trace:
                    trace_df = trace_df[trace_df["watermark_score"] != 0]

                trace_df = trace_df[trace_df["quality_preserved"] == True]
                trace_df = trace_df[["step_num", "watermark_score", "internlm_quality", "prompt"]]

                # NOTE: This will contain trace data for multiple attack runs, and not necessarily from step 0/-1!!!!
                for prompt, group in trace_df.groupby('prompt'):
                    # Thank ChatGPT for this genius idea 
                    group['trace_num'] = (group['step_num'] == -1).cumsum()

                    for run_num, attack in group.groupby('trace_num'):
                        trace_num_from_prompt[prompt] += 1
                        trace_num = trace_num_from_prompt[prompt]
                        
                        # case 1: `attack` contains full attack trace 
                        #   initial row is present
                        #   min zscore row is present
                        # case 2: `attack` contains beginning of trace
                        #   initial row is present
                        #   min zscore row may or may not be present
                        # case 3: `attack` contains end of trace
                        #   initial row is not present
                        #   min zscore row may or may not be present

                        initial_row = attack[attack["step_num"] == -1]
                        min_zscore_row = attack[attack.watermark_score == attack.watermark_score.min()]
                        key = (prompt, trace_num)

                        if len(initial_row) != 0:
                            initial_data[key] = [initial_row["internlm_quality"].iloc[0], initial_row["watermark_score"].iloc[0]]
                        if key not in final_data or final_data[key][1] > min_zscore_row["watermark_score"].iloc[0]:
                            final_data[key] = [min_zscore_row["internlm_quality"].iloc[0], min_zscore_row["watermark_score"].iloc[0]]
            
            initial_data = pd.DataFrame.from_dict(initial_data, orient="index", columns=["quality", "zscores"])
            final_data = pd.DataFrame.from_dict(final_data, orient="index", columns=["quality", "zscores"])

            axs[idm].scatter(final_data["zscores"], final_data["quality"], label="Minimum z-score")
            axs[idm].scatter(initial_data["zscores"], initial_data["quality"], label="Initial")

            if "Adaptive" in trace:
                axs[idm].set_xlabel("Adaptive Score")
            else:
                axs[idm].set_xlabel("Z-score")
            axs[idm].set_ylabel(f"InternLM Quality")
            #axs[idm].set_ylim(-.05, 1.05)
            axs[idm].set_title(mutator)
            axs[idm].legend()
        
        plt.savefig(f"./attack/analysis/figs/zscore_vs_quality_{watermarker}.png")



def plot_estimated_watermark_breaking():
    # dimensions to consider: watermark type, mutator, entropy level

    for idx, watermarker in enumerate(annotated_watermarks):
        print(f"Plotting {watermarker}")
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        axs = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]]
        plt.suptitle(watermarker)

        for idm, mutator in enumerate(mutators):
            print(f"\tPlotting for {mutator}")
            
            final_data = [pd.DataFrame(columns=["step_num", "watermark_score"]) for e in range(1,11)]
            step_num_count_per_entropy = [defaultdict(int) for i in range(10)]
            # for each entropy: a dictionary for the number of instances of each step
            # this is because there's an error in the adaptive traces where the number of instances of each
            # step size is not uniform. We don't even watermark most rows, but the number
            num_traces_per_entropy = [0] * 10

            axs[idm].set_xlabel("Step Number")
            if "Adaptive" in watermarker:
                axs[idm].set_ylabel(f"Adaptive Score")
            else:
                axs[idm].set_ylabel(f"Z-score")
            axs[idm].set_title(mutator)

            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=1, vmax=10))
            cbar = plt.colorbar(sm, ax=axs[idm], orientation='vertical')
            cbar.set_label('Entropy Level')
        
            trace_df = load_all_csvs("./attack/traces", watermarker, mutator)

            if "watermark_score" not in trace_df.columns:
                print(f"\t\tNo watermark scores! Skipping {watermarker}, {mutator}")
                continue
            if "Adaptive" in watermarker:
                trace_df = trace_df[trace_df["watermark_score"] != 0]

            trace_df = trace_df[["step_num", "prompt", "watermark_score", "quality_preserved"]]
            if "Adaptive" not in watermarker:
                trace_df = trace_df[trace_df["quality_preserved"] == True]
            trace_df['trace_num'] = (trace_df['step_num'] == -1).cumsum()

            for run_num, attack in trace_df.groupby('trace_num'):
                entropy = prompt_to_entropy(attack["prompt"].iloc[0])

                if "Adaptive" in watermarker:
                    stride = num_steps(mutator)//10

                    def round_stride(x):
                        if x == stride*10:
                            return x
                        if x % stride < stride // 2:
                            return (x // stride) * stride - 1 
                        else:
                            return (x // stride + 1) * stride - 1

                    attack["step_num"] = attack["step_num"].apply(round_stride)
                    
                    for idx, step in attack.iterrows():
                        step_num_count_per_entropy[entropy-1][step["step_num"]] += 1 

                else:
                    num_traces_per_entropy[entropy-1] += 1
                
                final_data[entropy-1] = pd.concat([final_data[entropy-1], attack[["step_num", "watermark_score"]]]).groupby("step_num", as_index=False).sum()
                    
            for i, entropy_df in enumerate(final_data):
                color = entropy_colors[i]

                if "Adaptive" in watermarker:
                    entropy_df["watermark_score"] = entropy_df.apply(
                        lambda row: row["watermark_score"] / step_num_count_per_entropy[i][row["step_num"]],
                        axis=1)
                    axs[idm].plot(entropy_df["step_num"], entropy_df["watermark_score"], alpha=.8, color=color)
                else:
                    entropy_df["watermark_score"] = entropy_df["watermark_score"]/num_traces_per_entropy[i]
                    entropy_df["watermark_score"] = entropy_df["watermark_score"].rolling(window=num_steps(mutator), min_periods=1).mean()
                    axs[idm].plot(entropy_df["step_num"], entropy_df["watermark_score"], alpha=.8, color=color)
            
            if "Adaptive" in watermarker:
                axs[idm].set_ylim(45, 105)
            else:
                axs[idm].set_ylim(bottom=-1)

            
        plt.savefig(f"./attack/analysis/figs/estimated_breaking_{watermarker}.png")



def sanity_checks(metric):
    # dimensions to consider: watermark type, mutator, entropy level

    for idx, watermarker in enumerate(watermarks):
        print(f"Plotting {watermarker}")
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        axs = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]]
        plt.suptitle(watermarker)

        for idm, mutator in enumerate(mutators):
            print(f"\tPlotting for {mutator}")

            axs[idm].set_xlabel("Step Number")
            axs[idm].set_title(mutator)

            # set legend for colors 
            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=1, vmax=10))
            cbar = plt.colorbar(sm, ax=axs[idm], orientation='vertical')
            cbar.set_label('Entropy Level')
            

            trace_df = load_all_csvs("./attack/traces", watermarker, mutator, annotated=False)
            if metric not in trace_df.columns:
                print(f"\t\tDataframe missing {metric}! Skipping {watermarker}, {mutator}")
                continue
            trace_df['trace_num'] = (trace_df['step_num'] == -1).cumsum()

            for run_num, attack in trace_df.groupby('trace_num'):
                entropy = prompt_to_entropy(attack["prompt"].iloc[0])
                color = entropy_colors[entropy-1]
                axs[idm].plot(attack["step_num"], attack[metric], alpha=.8, color=color)

        
        plt.savefig(f"./attack/analysis/figs/sanity_checks/{metric}_{watermarker}.png")

def sanity_checks_rolling(metric):
    for idx, watermarker in enumerate(watermarks):
        print(f"Plotting {watermarker}")
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        axs = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]]
        plt.suptitle(watermarker)

        for idm, mutator in enumerate(mutators):
            print(f"\tPlotting for {mutator}")
            
            final_data = [pd.DataFrame(columns=["step_num", metric]) for e in range(1,11)]
            num_traces_per_entropy = [0] * 10

            axs[idm].set_xlabel("Step Number")
            axs[idm].set_ylabel(metric)
            axs[idm].set_title(mutator)

            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=1, vmax=10))
            cbar = plt.colorbar(sm, ax=axs[idm], orientation='vertical')
            cbar.set_label('Entropy Level')
        
            trace_df = load_all_csvs("./attack/traces", watermarker, mutator)

            if metric not in trace_df.columns:
                print(f"\t\tMissing {metric}! Skipping {watermarker}, {mutator}")
                continue

            trace_df = trace_df[["step_num", "prompt", "quality_preserved", metric]]
            trace_df = trace_df[trace_df["quality_preserved"] == True]
            trace_df['trace_num'] = (trace_df['step_num'] == -1).cumsum()

            for run_num, attack in trace_df.groupby('trace_num'):
                entropy = prompt_to_entropy(attack["prompt"].iloc[0])

                num_traces_per_entropy[entropy-1] += 1
                
                final_data[entropy-1] = pd.concat([final_data[entropy-1], attack[["step_num", metric]]]).groupby("step_num", as_index=False).sum()
                    
            for i, entropy_df in enumerate(final_data):
                color = entropy_colors[i]

                entropy_df[metric] = entropy_df[metric].rolling(window=num_steps(mutator), min_periods=1).mean()
                axs[idm].plot(entropy_df["step_num"], entropy_df[metric]/num_traces_per_entropy[i], alpha=.8, color=color)
           
            
        plt.savefig(f"./attack/analysis/figs/sanity_checks/rolling_{metric}_{watermarker}.png")

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
    # plot_sliding_window_success_rates()
    # plot_quality_degredation_vs_zscore()
    plot_estimated_watermark_breaking()
    # sanity_checks_rolling("nicolai_quality")
