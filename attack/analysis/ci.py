import numpy as np
import pandas as pd
import scipy.stats as st

from attack.utils import load_all_csvs

def num_steps(mutator):
    if "Document" in mutator:
        return 100
    if "Word" in mutator:
        return 1000
    if "Span" in mutator:
        return 250
    if "Sentence" in mutator:
        return 150

# ------------------------------------------------------------------------
# 1) Define functions for confidence intervals
# ------------------------------------------------------------------------

def t_conf_interval(data, confidence=0.95):
    """
    Compute a t-based confidence interval for the mean of 'data'.
    """
    data = np.asarray(data)
    n = len(data)
    mean_ = np.mean(data)
    sem_ = st.sem(data)  # standard error of the mean
    dof = n - 1
    alpha = 1 - confidence
    t_crit = st.t.ppf(1 - alpha/2, dof)
    margin = t_crit * sem_
    return mean_ - margin, mean_ + margin

def bootstrap_conf_interval(data, confidence=0.95, n_boot=10_000, random_state=None):
    """
    Compute a bootstrap confidence interval (percentile method).
    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data)
    n = len(data)
    boot_means = []
    for _ in range(n_boot):
        sample_indices = rng.integers(0, n, size=n)  # draw with replacement
        sample = data[sample_indices]
        boot_means.append(sample.mean())
    
    boot_means = np.sort(boot_means)
    alpha = 1 - confidence
    lower_idx = int((alpha / 2) * n_boot)
    upper_idx = int((1 - alpha / 2) * n_boot)
    return boot_means[lower_idx], boot_means[upper_idx]


# ------------------------------------------------------------------------
# 2) Main Script
# ------------------------------------------------------------------------

if __name__ == "__main__":

    # python -m attack.analysis.ci

    watermark_types = [
        "Adaptive",
        "KGW",
        "SIR",
    ]

    mutators = [
        "DocumentMutator",
        "Document1StepMutator",
        "Document2StepMutator",
        "SentenceMutator",
        "SpanMutator",
        "WordMutator",
        "EntropyWordMutator",
    ]

    all_results = []

    for watermark_type in watermark_types:
        for mutator in mutators:
            
            trace_df = load_all_csvs("./attack/traces/annotated", watermark_type, mutator)
            
            if trace_df.empty:
                print(f"[MAIN] No traces found for {watermark_type} + {mutator}")
                continue

            if "watermark_score" not in trace_df.columns:
                print(f"\t\tNo watermark scores! Skipping {watermark_type}, {mutator}")
                continue

            # ----------------------------------------------------------------
            # (A) Filter out rows that do not have valid watermark scores
            # ----------------------------------------------------------------
            trace_df = trace_df.dropna(subset=["watermark_score"])
            # If you also want to remove zero scores:
            # trace_df = trace_df[trace_df["watermark_score"] != 0]

            # Keep relevant columns
            trace_df = trace_df[["step_num", "watermark_score", "quality_preserved"]]

            # Optionally, for non-Adaptive, keep only quality_preserved = True:
            # if "Adaptive" not in watermark_type:
            #     trace_df = trace_df[trace_df["quality_preserved"] == True]

            # Group each run by 'trace_num'
            trace_df["trace_num"] = (trace_df["step_num"] == -1).cumsum()

            # ----------------------------------------------------------------
            # (B) For each trace, pick the N/10 positions in its own timeline
            # ----------------------------------------------------------------
            n_segments = 10  # 10 intervals => fraction_index from 0..10
            total_ideal_steps = num_steps(mutator)  # e.g. 100, 1000, etc.

            collected_rows = []
            grouped = trace_df.groupby("trace_num", as_index=False)

            for trnum, grp in grouped:
                # Sort by step_num so that "time" is ascending
                grp = grp.sort_values("step_num", ascending=True).reset_index(drop=True)

                # Number of valid steps for this trace
                M = len(grp)
                if M == 0:
                    continue

                # fraction_index from 0..10 inclusive
                for i in range(n_segments + 1):
                    # 'ideal_step' is the step that fraction i *would* be at
                    # if the trace had exactly 'total_ideal_steps' steps
                    # (We round it to an integer.)
                    ideal_step = int(round(i * total_ideal_steps / n_segments))

                    # Find the actual row in this trace that corresponds
                    # to fraction i of the *observed* step range
                    # If i=0 => index=0; i=10 => index=M-1
                    idx = int(round(i * (M - 1) / n_segments))

                    if idx < 0 or idx >= M:
                        continue

                    row = grp.iloc[idx]
                    collected_rows.append({
                        "trace_num": trnum,
                        "fraction_index": i,
                        "ideal_step": ideal_step,      # The "intended" step
                        "actual_step": row["step_num"], # The actual step in this trace
                        "watermark_score": row["watermark_score"],
                        "watermark_type": watermark_type,
                        "mutator": mutator,
                    })

            # ----------------------------------------------------------------
            # (C) Now 'collected_rows' has one row for each fraction_index
            #     across all traces. We'll compute CIs by fraction_index.
            # ----------------------------------------------------------------
            if not collected_rows:
                print(f"[Warning] No valid rows for {watermark_type}, {mutator} after filtering.")
                continue

            frac_df = pd.DataFrame(collected_rows)

            # We'll group by fraction_index, computing means + CIs
            final_list = []
            for frac_idx, subgrp in frac_df.groupby("fraction_index"):
                data = subgrp["watermark_score"].values

                # t-based CI
                t_low, t_high = t_conf_interval(data, confidence=0.95)
                # bootstrap CI
                b_low, b_high = bootstrap_conf_interval(data, confidence=0.95,
                                                        n_boot=10_000,
                                                        random_state=42)

                # The 'ideal_step' is the same for all rows with the same i
                # (since i -> ideal_step is deterministic). We can just take .iloc[0].
                some_row = subgrp.iloc[0]
                final_list.append({
                    "watermark_type": watermark_type,
                    "mutator": mutator,
                    "fraction_index": frac_idx,
                    "ideal_step": some_row["ideal_step"],
                    "count_at_fraction": len(data),
                    "mean_score": data.mean(),
                    "t_lower": t_low,
                    "t_upper": t_high,
                    "b_lower": b_low,
                    "b_upper": b_high,
                })

            results_df = pd.DataFrame(final_list)
            print(f"=== Results for {watermark_type} + {mutator} ===")
            print(results_df)

            all_results.append(results_df)

    # ----------------------------------------------------------------
    # (D) Concatenate all (watermark_type, mutator) results and save
    # ----------------------------------------------------------------
    if all_results:
        all_scores_df = pd.concat(all_results, ignore_index=True)
        save_path = "./attack/analysis/csv/confidence_intervals.csv"
        all_scores_df.to_csv(save_path, index=False)
        print(f"Saved all confidence intervals to {save_path}")
    else:
        print("No data to save. all_results was empty.")
