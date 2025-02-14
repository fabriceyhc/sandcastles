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

mutators = ["Document1StepMutator", "Document2StepMutator", "DocumentMutator", "SentenceMutator", "SpanMutator", "EntropyWordMutator", "WordMutator"]
watermarks = ["GPT4o_unwatermarked", "Adaptive", "KGW", "SIR"]
watermark_thresholds = {
    "Adaptive" : 70,
    "KGW" : .50,
    "SIR": .20,
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
        # fig, axs = plt.subplots(3, 3, figsize=(20, 12))
        # axs = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[2,0], axs[2,1]]

        fig = plt.figure(figsize=(26, 12))
        axs = [
            plt.subplot2grid((2,8), (0,1), colspan=2),            
            plt.subplot2grid((2,8), (0,3), colspan=2),           
            plt.subplot2grid((2,8), (0,5), colspan=2),

            plt.subplot2grid((2,8), (1,0), colspan=2),           
            plt.subplot2grid((2,8), (1,2), colspan=2),
            plt.subplot2grid((2,8), (1,4), colspan=2),           
            plt.subplot2grid((2,8), (1,6), colspan=2),
        ]
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


            
            trace_df = load_all_csvs("./attack/traces/annotated", watermarker, mutator)

            if trace_df.empty:
                print(f"\t\tEmpty dataframe! Skipping {watermarker}, {mutator}")
                continue
            trace_df['trace_num'] = (trace_df['step_num'] == -1).cumsum()

            pain = trace_df[trace_df['step_num'] == "======="]
            if not pain.empty: print(pain)
            try:
                trace_df = trace_df[trace_df['step_num'] <= num_steps(mutator)]
            except:            
                trace_df['step_num'] = pd.to_numeric(trace_df['step_num'])
                trace_df = trace_df[trace_df['step_num'] <= num_steps(mutator)]


            for run_num, attack in trace_df.groupby('trace_num'):
                attack["quality_preserved"] = attack["quality_preserved"].rolling(window=window_size, min_periods=1).mean()
                entropy = prompt_to_entropy(attack["prompt"].iloc[0])
                color = entropy_colors[entropy-1]

                final_data[entropy-1] = pd.concat([final_data[entropy-1], attack[["step_num", "quality_preserved"]]]).groupby("step_num", as_index=False).sum()
                num_traces_per_entropy[entropy-1] += 1

                        
            for i, entropy_df in enumerate(final_data):
                color = entropy_colors[i]
                axs[idm].plot(entropy_df["step_num"], entropy_df["quality_preserved"]/num_traces_per_entropy[i], alpha=.8, color=color)

        fig.tight_layout(pad=3)
        plt.savefig(f"./attack/analysis/figs/rolling_success_{watermarker}.png")
def plot_sliding_window_quality():
    # dimensions to consider: watermark type, mutator, entropy level

    for idx, watermarker in enumerate(watermarks):
        print(f"Plotting {watermarker}")
        # fig, axs = plt.subplots(3, 3, figsize=(20, 12))
        # axs = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[2,0], axs[2,1]]

        fig = plt.figure(figsize=(26, 12))
        axs = [
            plt.subplot2grid((2,8), (0,1), colspan=2),            
            plt.subplot2grid((2,8), (0,3), colspan=2),           
            plt.subplot2grid((2,8), (0,5), colspan=2),

            plt.subplot2grid((2,8), (1,0), colspan=2),           
            plt.subplot2grid((2,8), (1,2), colspan=2),
            plt.subplot2grid((2,8), (1,4), colspan=2),           
            plt.subplot2grid((2,8), (1,6), colspan=2),
        ]
        plt.suptitle(watermarker)

        for idm, mutator in enumerate(mutators):
            print(f"\tPlotting for {mutator}")
            
            window_size = num_steps(mutator)//10

            axs[idm].set_xlabel("Step Number")
            axs[idm].set_ylabel(f"Rolling Quality Score (window size = {window_size})")
            axs[idm].set_ylim(.1, .21)
            axs[idm].set_title(mutator)

            # set legend for colors 
            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=1, vmax=10))
            cbar = plt.colorbar(sm, ax=axs[idm], orientation='vertical')
            cbar.set_label('Entropy Level')
            
            final_data = [pd.DataFrame(columns=["step_num", "armolm_quality"]) for e in range(1,11)]
            num_traces_per_entropy = [0] * 10


            
            trace_df = load_all_csvs("./attack/traces/annotated", watermarker, mutator)

            if trace_df.empty:
                print(f"\t\tEmpty dataframe! Skipping {watermarker}, {mutator}")
                continue


            for run_num, attack in trace_df.groupby('group_id'):
                attack["armolm_quality"] = attack["armolm_quality"].rolling(window=window_size, min_periods=1).mean()
                entropy = prompt_to_entropy(attack["prompt"].iloc[0])
                color = entropy_colors[entropy-1]

                final_data[entropy-1] = pd.concat([final_data[entropy-1], attack[["step_num", "armolm_quality"]]]).groupby("step_num", as_index=False).sum()
                num_traces_per_entropy[entropy-1] += 1

                        
            for i, entropy_df in enumerate(final_data):
                color = entropy_colors[i]
                axs[idm].plot(entropy_df["step_num"], entropy_df["armolm_quality"]/num_traces_per_entropy[i], alpha=.8, color=color)

        fig.tight_layout(pad=3)
        plt.savefig(f"./attack/analysis/figs/rolling_quality_{watermarker}.png")




# def plot_quality_degredation_vs_zscore():
#     for idx, watermarker in enumerate(annotated_watermarks):
#         print(f"Plotting {watermarker}")
#         # fig, axs = plt.subplots(3, 3, figsize=(20, 12))
#         # axs = [axs[0,0], axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[2,0], axs[2,1]]

#         fig = plt.figure(figsize=(26, 12))
#         axs = [
#             plt.subplot2grid((2,8), (0,1), colspan=2),            
#             plt.subplot2grid((2,8), (0,3), colspan=2),           
#             plt.subplot2grid((2,8), (0,5), colspan=2),

#             plt.subplot2grid((2,8), (1,0), colspan=2),           
#             plt.subplot2grid((2,8), (1,2), colspan=2),
#             plt.subplot2grid((2,8), (1,4), colspan=2),           
#             plt.subplot2grid((2,8), (1,6), colspan=2),
#         ]


#         plt.suptitle(watermarker)

#         for idm, mutator in enumerate(mutators):
#             print(f"\tPlotting for {mutator}")
#             trace_df = load_all_csvs("./attack/traces/annotated", watermarker, mutator)
            
#             # key is (prompt, trace_num) -> data
#             initial_data = {}
#             final_data = {}

#             trace_num_from_prompt = defaultdict(int)

#             # for trace in traces:
#             #     trace_df = pd.read_csv(trace)

#             # if "internlm_quality" not in trace_df.columns or "watermark_score" not in trace_df.columns:
#             #     print(f"Skipping {trace}") 
#             #     continue
            
#             if "Adaptive" in watermarker:
#                 trace_df = trace_df[trace_df["watermark_score"] != 0]

#             trace_df = trace_df[trace_df["quality_preserved"] == True]
#             trace_df = trace_df[["step_num", "watermark_score", "armolm_quality", "prompt"]]

#             for id, group in trace_df.groupby("group_id"):

            
#             initial_data = pd.DataFrame.from_dict(initial_data, orient="index", columns=["quality", "zscores"])
#             final_data = pd.DataFrame.from_dict(final_data, orient="index", columns=["quality", "zscores"])

#             axs[idm].scatter(final_data["zscores"], final_data["quality"], label="Minimum z-score")
#             axs[idm].scatter(initial_data["zscores"], initial_data["quality"], label="Initial")

#             if "Adaptive" in watermarker:
#                 axs[idm].set_xlabel("Adaptive Score")
#             else:
#                 axs[idm].set_xlabel("Z-score")
#             axs[idm].set_ylabel(f"ArmoLM Quality")
#             #axs[idm].set_ylim(-.05, 1.05)
#             axs[idm].set_title(mutator)
#             axs[idm].legend()
        
#         fig.tight_layout(pad=3)
#         plt.savefig(f"./attack/analysis/figs/zscore_vs_quality_{watermarker}.png")

def plot_estimated_watermark_breaking():
    for watermarker in annotated_watermarks:
        fig = plt.figure(figsize=(26, 12))
        axs = [
            plt.subplot2grid((2,8), (0,1), colspan=2),            
            plt.subplot2grid((2,8), (0,3), colspan=2),           
            plt.subplot2grid((2,8), (0,5), colspan=2),

            plt.subplot2grid((2,8), (1,0), colspan=2),           
            plt.subplot2grid((2,8), (1,2), colspan=2),
            plt.subplot2grid((2,8), (1,4), colspan=2),           
            plt.subplot2grid((2,8), (1,6), colspan=2),
        ]

        plt.suptitle(watermarker)

        for idm, mutator in enumerate(mutators):
            ax = axs[idm]
            ax.set_title(mutator)

            print(f"\tPlotting for {mutator}")

            # Prepare containers for each entropy level (1..10)
            final_data = [
                pd.DataFrame(columns=["step_num", "watermark_score"])
                for _ in range(10)
            ]
            step_num_count_per_entropy = [defaultdict(int) for _ in range(10)]
            num_traces_per_entropy = [0] * 10

            ax = axs[idm]
            ax.set_xlabel("Step Number")
            if "Adaptive" in watermarker:
                ax.set_ylabel("Adaptive Score")
            else:
                ax.set_ylabel("Z-score")
            ax.set_title(mutator)

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=1, vmax=10))
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
            cbar.set_label('Entropy Level')

            # Load data
            trace_df = load_all_csvs("./attack/traces/annotated", watermarker, mutator)
            if "watermark_score" not in trace_df.columns:
                print(f"\t\tNo watermark scores! Skipping {watermarker}, {mutator}")
                continue

            # For Adaptive, remove rows with zero watermark_score
            if "Adaptive" in watermarker:
                trace_df = trace_df[trace_df["watermark_score"] != 0]

            trace_df = trace_df[["step_num", "prompt", "watermark_score", "quality_preserved"]]
            if "Adaptive" not in watermarker:
                trace_df = trace_df[trace_df["quality_preserved"] == True]

            # Group each run by 'trace_num'
            trace_df['trace_num'] = (trace_df['step_num'] == -1).cumsum()

            # Fill final_data[i] for each entropy i
            for run_num, attack in trace_df.groupby('trace_num'):
                # You’ll need your own function to convert prompt -> integer [1..10]
                entropy = prompt_to_entropy(attack["prompt"].iloc[0])

                if "Adaptive" in watermarker:
                    # Possibly adjust or bin steps for Adaptive
                    stride = num_steps(mutator)//10

                    def round_stride(x):
                        if x == stride * 10:
                            return x
                        if x % stride < stride // 2:
                            return (x // stride) * stride - 1
                        else:
                            return (x // stride + 1) * stride - 1

                    attack["step_num"] = attack["step_num"].apply(round_stride)
                    for idx2, step in attack.iterrows():
                        step_num_count_per_entropy[entropy-1][step["step_num"]] += 1
                else:
                    num_traces_per_entropy[entropy-1] += 1

                # Sum watermark_score by step_num for that entropy
                final_data[entropy-1] = pd.concat([
                    final_data[entropy-1],
                    attack[["step_num", "watermark_score"]]
                ]).groupby("step_num", as_index=False).sum()

            # Plot each entropy curve separately
            for i, entropy_df in enumerate(final_data):
                if entropy_df.empty:
                    continue

                # Average out scores
                if "Adaptive" in watermarker:
                    entropy_df["watermark_score"] = entropy_df.apply(
                        lambda row: row["watermark_score"] /
                                    step_num_count_per_entropy[i].get(row["step_num"], 1),
                        axis=1
                    )
                else:
                    if num_traces_per_entropy[i] > 0:
                        entropy_df["watermark_score"] /= num_traces_per_entropy[i]

                # Sort by step_num and optionally smooth for adaptive
                entropy_df.sort_values("step_num", inplace=True)
                
                if "Adaptive" not in watermarker:
                    entropy_df["watermark_score"] = entropy_df["watermark_score"].rolling(window=num_steps(mutator), min_periods=1).mean()

                # Plot the actual curve (per entropy)
                ax.plot(
                    entropy_df["step_num"],
                    entropy_df["watermark_score"],
                    alpha=0.8,
                    color=entropy_colors[i]
                )

            # Combine per-entropy data to get a single DataFrame
            all_entropy_df = pd.concat(final_data)
            if not all_entropy_df.empty:
                # Average over step_num
                mean_df = (
                    all_entropy_df.groupby("step_num", as_index=False)["watermark_score"]
                    .mean()
                    .sort_values("step_num")
                )
                xdata = mean_df["step_num"].values
                ydata = mean_df["watermark_score"].values

                # Plot the mean curve
                ax.plot(xdata, ydata, color="black", alpha=0.7, label="_nolegend_")

                # Only fit if we have enough points
                if len(xdata) > 5:
                    p0 = [
                        max(ydata) - min(ydata),  # A0
                        50.0,                    # B0
                        min(ydata),              # C0
                    ]
                    bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

                    try:
                        popt, _ = curve_fit(exponential_offset, xdata, ydata,
                                            p0=p0, bounds=bounds)
                        A_fit, B_fit, C_fit = popt

                        # 1) Compute the crossing time from the fitted parameters
                        threshold = watermark_thresholds[watermarker]
                        crossing_point = None
                        if A_fit > 0 and threshold > C_fit and threshold < (A_fit + C_fit):
                            from math import log
                            cp = B_fit * log(A_fit / (threshold - C_fit))
                            crossing_point = max(cp, 0)
                        
                        # 2) Plot the fitted curve ONLY out to num_steps(mutator)
                        #    (so if crossing > 1000, the line ends at 1000, but we still label crossing=2438 steps)
                        x_max = num_steps(mutator)
                        x_fine = np.linspace(0, x_max, 200)
                        y_fine = exponential_offset(x_fine, A_fit, B_fit, C_fit)

                        # Show dotted line. Put crossing in the label either way.
                        if crossing_point is not None:
                            lbl = f"Est. steps to break @ {threshold}: {int(crossing_point)} steps"
                        else:
                            lbl = f"Est. steps to break @ {threshold}: Infinity"

                        ax.plot(
                            x_fine, y_fine,
                            linestyle="dotted", linewidth=3,
                            label=lbl
                        )
                        ax.legend()

                    except RuntimeError:
                        # Fit failed
                        ax.plot([], [], ' ', label=f"No break found (threshold={threshold})")
                        ax.legend()

                else:
                    ax.plot([], [], ' ', label="Not enough points to fit")
                    ax.legend()
            else:
                ax.plot([], [], ' ', label="No data found")
                ax.legend()

        if "Adaptive" in watermarker:
            global_min = 50
            global_max = 100
        else:
            global_min = 0
            global_max = 7
        
        for ax in axs:
            ax.set_ylim(global_min, global_max)

        fig.tight_layout(pad=3)
        plt.savefig(f"./attack/analysis/figs/estimated_breaking_{watermarker}.png")



def sanity_checks(metric):
    # dimensions to consider: watermark type, mutator, entropy level

    for idx, watermarker in enumerate(watermarks):
        print(f"Plotting {watermarker}")
        fig = plt.figure(figsize=(26, 12))
        axs = [
            plt.subplot2grid((2,8), (0,1), colspan=2),            
            plt.subplot2grid((2,8), (0,3), colspan=2),           
            plt.subplot2grid((2,8), (0,5), colspan=2),

            plt.subplot2grid((2,8), (1,0), colspan=2),           
            plt.subplot2grid((2,8), (1,2), colspan=2),
            plt.subplot2grid((2,8), (1,4), colspan=2),           
            plt.subplot2grid((2,8), (1,6), colspan=2),
        ]
        plt.suptitle(watermarker)

        for idm, mutator in enumerate(mutators):
            print(f"\tPlotting for {mutator}")

            axs[idm].set_xlabel("Step Number")
            axs[idm].set_ylabel(metric)
            axs[idm].set_title(mutator)

            # set legend for colors 
            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=1, vmax=10))
            cbar = plt.colorbar(sm, ax=axs[idm], orientation='vertical')
            cbar.set_label('Entropy Level')
            

            trace_df = load_all_csvs("./attack/traces/annotated", watermarker, mutator)
            if metric not in trace_df.columns:
                print(f"\t\tDataframe missing {metric}! Skipping {watermarker}, {mutator}")
                continue
            trace_df['trace_num'] = (trace_df['step_num'] == -1).cumsum()

            for run_num, attack in trace_df.groupby('trace_num'):
                entropy = prompt_to_entropy(attack["prompt"].iloc[0])
                color = entropy_colors[entropy-1]
                label = f"{watermarker},{mutator},{run_num}"
                axs[idm].plot(attack["step_num"], attack[metric], alpha=.8, color=color, label=label)

        
        fig.tight_layout(pad=3)
        plt.savefig(f"./attack/analysis/figs/sanity_checks/{metric}_{watermarker}.png")



def sanity_checks_rolling(metric, include_unannotated=False):
    for idx, watermarker in enumerate(watermarks):
        print(f"Plotting {watermarker}")
        fig = plt.figure(figsize=(26, 12))
        axs = [
            plt.subplot2grid((2,8), (0,1), colspan=2),            
            plt.subplot2grid((2,8), (0,3), colspan=2),           
            plt.subplot2grid((2,8), (0,5), colspan=2),

            plt.subplot2grid((2,8), (1,0), colspan=2),           
            plt.subplot2grid((2,8), (1,2), colspan=2),
            plt.subplot2grid((2,8), (1,4), colspan=2),           
            plt.subplot2grid((2,8), (1,6), colspan=2),
        ]
        plt.suptitle(watermarker)

        for idm, mutator in enumerate(mutators):
            print(f"\tPlotting for {mutator}")

            axs[idm].set_xlabel("Step Number")
            axs[idm].set_ylabel(metric)
            axs[idm].set_title(mutator)

            # set legend for colors 
            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=1, vmax=10))
            cbar = plt.colorbar(sm, ax=axs[idm], orientation='vertical')
            cbar.set_label('Entropy Level')

            if include_unannotated == None:
                # only unannotated
                trace_df = load_all_csvs("./attack/traces", watermarker, mutator)
            elif include_unannotated == False:
                # only annotated
                trace_df = load_all_csvs("./attack/traces/annotated", watermarker, mutator)
            else:
                # both annotated, unannotated
                trace_df = pd.concat([
                    load_all_csvs("./attack/traces", watermarker, mutator),
                    load_all_csvs("./attack/traces/annotated", watermarker, mutator)
                ], ignore_index=True)
            

            if metric not in trace_df.columns:
                print(f"\t\tDataframe missing {metric}! Skipping {watermarker}, {mutator}")
                continue
            trace_df['trace_num'] = (trace_df['step_num'] == -1).cumsum()

            for run_num, attack in trace_df.groupby('trace_num'):
                window_size = num_steps(mutator)//10
                attack[metric] = attack[metric].rolling(window=window_size, min_periods=1).mean()
                entropy = prompt_to_entropy(attack["prompt"].iloc[0])
                color = entropy_colors[entropy-1]
                label = f"{watermarker},{mutator},{run_num}"
                axs[idm].plot(attack["step_num"], attack[metric], alpha=.8, color=color, label=label)

        
        fig.tight_layout(pad=3)
        plt.savefig(f"./attack/analysis/figs/sanity_checks/rolling_{metric}_{watermarker}.png")


def sanity_checks_rolling_entropy(metric):
    for idx, watermarker in enumerate(watermarks):
        print(f"Plotting {watermarker}")
        fig = plt.figure(figsize=(26, 12))
        axs = [
            plt.subplot2grid((2,8), (0,1), colspan=2),            
            plt.subplot2grid((2,8), (0,3), colspan=2),           
            plt.subplot2grid((2,8), (0,5), colspan=2),

            plt.subplot2grid((2,8), (1,0), colspan=2),           
            plt.subplot2grid((2,8), (1,2), colspan=2),
            plt.subplot2grid((2,8), (1,4), colspan=2),           
            plt.subplot2grid((2,8), (1,6), colspan=2),
        ]
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
        
            trace_df = load_all_csvs("./attack/traces/annotated", watermarker, mutator)

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
           
            
        fig.tight_layout(pad=3)
        plt.savefig(f"./attack/analysis/figs/sanity_checks/rolling_entropy_{metric}_{watermarker}.png")

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


def exponential_offset(x, A, B, C):
    """
    A * exp(-x / B) + C

    Where:
    - A > 0 is the initial amplitude (roughly the difference between starting and offset).
    - B > 0 is the decay constant (larger means slower decay).
    - C is the asymptote or baseline (the curve flattens near y = C).
    """
    return A * np.exp(-x / B) + C


if __name__ ==  "__main__":

    # python -m attack.analysis.attack_efficiency_analysis
    plot_sliding_window_quality()
    # plot_sliding_window_success_rates()
    # plot_quality_degredation_vs_zscore()
    # plot_estimated_watermark_breaking()

    # sanity_checks_rolling("quality_preserved", include_unannotated=True)
