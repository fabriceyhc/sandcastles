import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from scipy.optimize import curve_fit
from attack.utils import load_all_csvs

mutators = ["Document1StepMutator", "Document2StepMutator", "SentenceMutator", "SpanMutator", "EntropyWordMutator", "WordMutator"]
watermarks = ["GPT4o_unwatermarked", "Adaptive", "KGW", "SIR"]
watermark_thresholds = {
    "Adaptive" : 60,
    "KGW" : .25,
    "SIR": .25,
}
annotated_watermarks = ["KGW", "Adaptive", "SIR"]
quality_watermarks = ["KGW", "GPT4o_unwatermarked"]


# https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html
# Entropy: ranges 1-10. Use a color range?
entropy_colors = cm.plasma(np.linspace(0,1,10))


def exponential(x, a, b):
    return a * np.e**(x / (-b*100)) 


def plot_estimated_watermark_breaking():
    # dimensions to consider: watermark type, mutator, entropy level

    for idx, watermarker in enumerate(annotated_watermarks):
        print(f"Plotting {watermarker}")
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        axs = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]]
        plt.suptitle(watermarker)

        for idm, mutator in enumerate(mutators):
            print(f"\tPlotting for {mutator}")
            
            final_data = pd.DataFrame(columns=["step_num", "watermark_score"]) 

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
                final_data = pd.concat([final_data, attack[["step_num", "watermark_score"]]])
              
            # rough estimates
            initial_point = 90 if "Adaptive" in watermarker else 4
            final_point = 70 if "Adaptive" in watermarker else 2

            params, cov = curve_fit(exponential, 
                final_data["step_num"].to_numpy(), final_data["watermark_score"].to_numpy(),
                p0=[initial_point, num_steps(mutator)/(100 * np.log(initial_point/final_point))]
                )

            threshold = watermark_thresholds[watermarker]
            crossing_point = np.log(params[0]/threshold) * 100 * params[1]
            print(f"\t\t{crossing_point} steps to {threshold}")
            print(f"\t\t{params}")

            x = np.linspace(0, num_steps(mutator), num_steps(mutator)//10)
            y = exponential(x, params[0], params[1])

            axs[idm].plot(x, y, alpha=.8)
            
            if "Adaptive" in watermarker:
                axs[idm].set_ylim(45, 105)
            else:
                axs[idm].set_ylim(-1.25, 6.25)

        
        plt.savefig(f"./attack/analysis/figs/estimated_breaking_trendline_{watermarker}.png")



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
