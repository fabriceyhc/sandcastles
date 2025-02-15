import os
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class WatermarkMetricsEvaluator:
    def __init__(self, true_labels, watermark_scores):
        self.true_labels = np.array(true_labels)
        self.watermark_scores = np.array(watermark_scores)
        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.true_labels) != len(self.watermark_scores):
            raise ValueError("Input arrays must have equal length")
        if not np.isin(self.true_labels, [0, 1]).all():
            raise ValueError("True labels must be binary (0/1)")

    def compute_metrics(self, threshold, reverse=False):
        pred = self.watermark_scores >= threshold if not reverse \
               else self.watermark_scores <= threshold
        return self._calculate_metrics(pred)

    def find_optimal_threshold(self, metric='F1', reverse=False):
        sorted_scores, sorted_labels = self._get_sorted_data(reverse)
        total_p = self.true_labels.sum()
        total_n = len(self.true_labels) - total_p

        cum_tp = np.cumsum(sorted_labels)
        cum_fp = np.arange(1, len(sorted_labels)+1) - cum_tp

        thresholds = self._generate_thresholds(sorted_scores, reverse)
        metrics = self._compute_all_metrics(cum_tp, cum_fp, total_p, total_n)

        metric_map = {
            'F1': metrics['F1'],
            'TPR': metrics['TPR'],
            'TNR': metrics['TNR'],
            'ACC': metrics['ACC'],
            'P': metrics['P']
        }
        metric_key = metric.upper()
        metric_values = metric_map[metric_key]

        best_idx = np.nanargmax(metric_values)
        return thresholds[best_idx]

    def find_threshold_at_fpr(self, target_fpr, reverse=False):
        sorted_scores, sorted_labels = self._get_sorted_data(reverse)
        total_n = len(self.true_labels) - self.true_labels.sum()
        
        # Calculate midpoints between sorted scores
        thresholds = self._generate_midpoints(sorted_scores, reverse)
        
        # Calculate FPR for each threshold
        fpr_values = []
        for thresh in thresholds:
            pred = sorted_scores >= thresh if not reverse else sorted_scores <= thresh
            fp = np.sum((sorted_labels == 0) & pred)
            fpr = fp / total_n if total_n else 0.0
            fpr_values.append(fpr)
        
        # Find the highest threshold meeting FPR requirement
        valid_thresholds = [thresh for thresh, fpr in zip(thresholds, fpr_values) if fpr <= target_fpr]
        return valid_thresholds[0] if valid_thresholds else thresholds[-1]

    def _generate_midpoints(self, sorted_scores, reverse):
        """Generate optimal thresholds between data points"""
        if len(sorted_scores) < 1:
            return np.array([])
            
        # Create midpoints between consecutive scores
        midpoints = (sorted_scores[:-1] + sorted_scores[1:]) / 2
        
        # Add boundary thresholds
        if not reverse:
            return np.concatenate([[np.inf], midpoints, [-np.inf]])
        return np.concatenate([[-np.inf], midpoints, [np.inf]])

    def _get_sorted_data(self, reverse):
        sort_order = np.argsort(-self.watermark_scores) if not reverse \
                     else np.argsort(self.watermark_scores)
        return self.watermark_scores[sort_order], self.true_labels[sort_order]

    def _generate_thresholds(self, sorted_scores, reverse):
        # Create N+1 thresholds for N sorted scores
        if not reverse:
            return np.concatenate([[np.inf], sorted_scores])
        return np.concatenate([sorted_scores, [-np.inf]])

    def _compute_all_metrics(self, cum_tp, cum_fp, total_p, total_n):
        with np.errstate(divide='ignore', invalid='ignore'):
            # Initialize metrics with proper float dtype
            metrics = {
                'TPR': np.divide(cum_tp, total_p, 
                               out=np.zeros_like(cum_tp, dtype=np.float64),
                               where=total_p != 0),
                'FPR': np.divide(cum_fp, total_n,
                               out=np.zeros_like(cum_fp, dtype=np.float64),
                               where=total_n != 0),
                'P': np.divide(cum_tp, cum_tp + cum_fp,
                             out=np.zeros_like(cum_tp, dtype=np.float64),
                             where=(cum_tp + cum_fp) != 0),
                'ACC': (cum_tp + (total_n - cum_fp)) / len(self.true_labels),
                'TNR': np.divide(total_n - cum_fp, total_n,
                               out=np.zeros_like(cum_fp, dtype=np.float64),
                               where=total_n != 0)
            }

            # Fixed F1 calculation without .filled()
            denominator = 2 * cum_tp + cum_fp + (total_p - cum_tp)
            metrics['F1'] = np.divide(
                2 * cum_tp,
                denominator.astype(np.float64),
                out=np.zeros_like(denominator, dtype=np.float64),
                where=denominator != 0
            )
            
            return metrics

    def _calculate_metrics(self, predicted):
        TP = np.sum((self.true_labels == 1) & predicted)
        FP = np.sum((self.true_labels == 0) & predicted)
        TN = np.sum((self.true_labels == 0) & ~predicted)
        FN = np.sum((self.true_labels == 1) & ~predicted)

        def safe_div(a, b):
            return a / b if b else 0.0

        return {
            'TPR': safe_div(TP, TP + FN),
            'FPR': safe_div(FP, FP + TN),
            'TNR': safe_div(TN, TN + FP),
            'FNR': safe_div(FN, FN + TP),
            'P': safe_div(TP, TP + FP),
            'R': safe_div(TP, TP + FN),
            'F1': safe_div(2*TP, 2*TP + FP + FN),
            'ACC': safe_div(TP + TN, len(self.true_labels)),
            'ASR': 1-safe_div(2*TP, 2*TP + FP + FN)
        }

def safe_extract_internlm_quality(value, key):
    """
    Safely extract a score from a quality_analysis string.
    If value is NaN or parsing fails, returns None.
    """
    try:
        if pd.notnull(value):
            analysis = ast.literal_eval(value)
            return analysis.get(key, None)
    except Exception:
        return None
    return None

def process_attack_traces(df):
    # Drop rows with NaN in 'watermark_score' to avoid issues with idxmin()
    df = df.dropna(subset=['watermark_score'])

    # Calculate metrics from filtered data
    df_sorted = df.sort_values(by=['group_id', 'step_num'])
    
    # Get original total steps before any filtering
    if 'original_steps' not in df.columns:
        total_steps = df_sorted.groupby('group_id')['step_num'].max()
        df_sorted['original_steps'] = df_sorted['group_id'].map(total_steps)

    print(df.columns)
    
    result = df_sorted.groupby('group_id').agg(
        init_watermark_score=pd.NamedAgg(column='watermark_score', aggfunc='first'),
        min_watermark_score=pd.NamedAgg(column='watermark_score', aggfunc='min'),
        final_watermark_score=pd.NamedAgg(column='watermark_score', aggfunc='last'),
        total_attack_steps=pd.NamedAgg(column='original_steps', aggfunc='first'),
        min_score_step=pd.NamedAgg(
            column='watermark_score',
            aggfunc=lambda x: df_sorted.loc[x.idxmin(), 'step_num'] if not x.isna().all() else np.nan
        ),
        # # Extract the initial internLM quality score from quality_analysis
        # init_internLM_quality=pd.NamedAgg(
        #     column='quality_analysis', 
        #     aggfunc=lambda s: safe_extract_internlm_quality(s.iloc[0], "original_score_A")
        # ),
        # Extract the final internLM quality score from quality_analysis
        final_internLM_quality=pd.NamedAgg(
            column='quality_analysis', 
            aggfunc=lambda s: safe_extract_internlm_quality(s.iloc[-1], "original_score_B")
        ),
        # init_perplexity=pd.NamedAgg(column='perplexity', aggfunc='first'),
        final_perplexity=pd.NamedAgg(column='perplexity', aggfunc='last'),
        # init_grammar_errors=pd.NamedAgg(column='grammar_errors', aggfunc='first'),
        final_grammar_errors=pd.NamedAgg(column='grammar_errors', aggfunc='last'),
        # init_unique_bigrams=pd.NamedAgg(column='unique_bigrams', aggfunc='first'),
        final_unique_bigrams=pd.NamedAgg(column='unique_bigrams', aggfunc='last')
    ).reset_index()

    return result

def format_success_rates(df):
    # Define the metrics that vary by score_type
    metrics_to_pivot = ['ASR']
    
    # Pivot these metrics, so that each score_type (min/fin) becomes a separate column
    df_pivot = df.pivot_table(
        index=["watermark_type", "mutator", "threshold", "score_mean", "score_std"],
        columns="score_type",
        values=metrics_to_pivot,
    )
    
    # Flatten the MultiIndex in columns (e.g., ASR_fin, ASR_min, etc.)
    df_pivot.columns = [f"{metric}_{stype}" for metric, stype in df_pivot.columns]
    df_pivot.reset_index(inplace=True)
    
    # The new average internLM and watermark scores do not depend on score_type.
    # Get them by grouping and taking the first value for each group.
    # avg_cols = [
    #     'avg_init_internLM_quality', 'avg_final_internLM_quality',
    #     'avg_init_watermark_score', 'avg_final_watermark_score',
    #     'avg_init_perplexity', 'avg_final_perplexity',
    #     'avg_init_grammar_errors', 'avg_final_grammar_errors',
    #     'avg_init_unique_bigrams', 'avg_final_unique_bigrams'
    # ]
    avg_cols = [
        'avg_final_internLM_quality',
        'avg_final_watermark_score',
        'avg_final_perplexity',
        'avg_final_grammar_errors',
        'avg_final_unique_bigrams'
    ]
    avg_df = df.groupby(
        ["watermark_type", "mutator", "threshold", "score_mean", "score_std"]
    )[avg_cols].first().reset_index()
    
    # Merge the pivoted metrics with the average metrics
    df_final = pd.merge(
        df_pivot,
        avg_df,
        on=["watermark_type", "mutator", "threshold", "score_mean", "score_std"],
        how="left"
    )
    
    return df_final

def plot_f1_scores(df, std_index=2, save_path="./attack/analysis/figs/f1_scores.png"):
    """
    Plots ASR_fin and ASR_min scores for different watermarking schemes and mutators using seaborn,
    distinguishing them with different marker shapes.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        std_index (int): The index of the standard deviation threshold to use (default is 2).
        save_path (str): The path to save the image as .png
    """

    # Filter the data for the specified standard deviation (threshold index)
    std_data = df.groupby(['watermark_type', 'mutator']).nth(std_index).reset_index()

    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid", font_scale=1.2)

    # Create a figure and axis
    plt.figure(figsize=(14, 8))
    ax = plt.gca()

    # Get unique mutators and watermark types
    mutators = std_data['mutator'].unique()
    watermark_types = std_data['watermark_type'].unique()

    # Define an offset for each watermark type to avoid overlap
    offset = 0.1  # Adjust this value to control the spacing between lines
    x_positions = np.arange(len(mutators))  # Base x-axis positions for mutators

    # Define markers for differentiation
    marker_styles = {"ASR_fin": "o", "ASR_min": "s"}  # Circle for ASR_fin, Square for ASR_min

    # Plot ASR_fin and ASR_min for each watermark type and mutator
    for i, watermark_type in enumerate(watermark_types):
        subset = std_data[std_data['watermark_type'] == watermark_type]

        # Calculate offset x-positions for this watermark type
        x_offset = x_positions + (i - len(watermark_types) / 2) * offset

        # Plot ASR_fin and ASR_min as connected lines with different markers
        for j, mutator in enumerate(mutators):
            row = subset[subset['mutator'] == mutator]
            if not row.empty:
                ASR_fin = row['ASR_fin'].values[0]
                ASR_min = row['ASR_min'].values[0]

                ax.plot([x_offset[j], x_offset[j]], [ASR_fin, ASR_min], color="black", linestyle="--", alpha=0.5)
                ax.scatter(x_offset[j], ASR_fin, marker=marker_styles["ASR_fin"], s=100, label=f"{watermark_type} (ASR_fin)")
                ax.scatter(x_offset[j], ASR_min, marker=marker_styles["ASR_min"], s=100, label=f"{watermark_type} (ASR_min)")

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(mutators, rotation=45, ha='right', fontsize=12)

    # Add labels and title
    ax.set_xlabel('Mutator', fontsize=14)
    ax.set_ylabel('F1 Score', fontsize=14)
    ax.set_title(f'ASR_fin and ASR_min Scores for Different Watermarking Schemes and Mutators (Threshold = {std_index}σ)', 
                 fontsize=16, pad=20)

    # Simplify legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))  # Remove duplicates while preserving order
    unique_handles = [handles[labels.index(lab)] for lab in unique_labels]
    ax.legend(unique_handles, unique_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, title="Watermark & Score")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show plot
    plt.show()

def plot_f1_heatmaps(df, threshold_std=2, watermarker_order=None, mutator_order=None, save_path=None):
    """
    Generates heatmaps for ASR_fin and ASR_min at a given threshold (in standard deviations)
    and optionally saves them to a specified location.

    Parameters:
    - df: pandas DataFrame containing columns ["watermark_type", "mutator", "threshold", "score_mean", "score_std", "ASR_fin", "ASR_min"]
    - threshold_std: The number of standard deviations away from the mean to filter the threshold.
    - watermarker_order: List of watermarker names in desired order.
    - mutator_order: List of mutator names in desired order.
    - save_path: Directory where plots should be saved (if None, plots are only displayed).

    Returns:
    - Saves the heatmaps as images if `save_path` is provided.
    - Displays the heatmaps.
    """

    if watermarker_order is None:
        watermarker_order = sorted(df["watermark_type"].unique())
    if mutator_order is None:
        mutator_order = sorted(df["mutator"].unique())

    # Remove "Mutator" from mutator names
    df = df.copy()
    df["mutator"] = df["mutator"].str.replace("Mutator", "", regex=False)

    # Filter the threshold based on the given standard deviation
    tolerance = 1e-5
    df_filtered = df[(df["threshold"] - (df["score_mean"] + threshold_std * df["score_std"])).abs() < tolerance]

    # Ensure only one row per (mutator, watermark_type) by averaging F1 scores if duplicates exist
    df_filtered = df_filtered.groupby(["mutator", "watermark_type"], as_index=False).agg({"ASR_fin": "mean", "ASR_min": "mean"})

    # Ensure correct categorical ordering
    df_filtered["watermark_type"] = pd.Categorical(df_filtered["watermark_type"], categories=watermarker_order, ordered=True)
    df_filtered["mutator"] = pd.Categorical(df_filtered["mutator"], categories=mutator_order, ordered=True)

    # Pivot the data for heatmap plotting
    df_ASR_fin = df_filtered.pivot(index="mutator", columns="watermark_type", values="ASR_fin")
    df_ASR_min = df_filtered.pivot(index="mutator", columns="watermark_type", values="ASR_min")

    # Ensure save path exists if specified
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Plot ASR_fin heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_ASR_fin, annot=True, cmap="RdBu", fmt=".2f", vmin=0, vmax=1)
    plt.title(f"ASR_fin Heatmap (Threshold = {threshold_std}σ)")
    plt.xlabel("Watermarker")
    plt.ylabel("Mutator")
    if save_path:
        fin_path = os.path.join(save_path, f"ASR_fin_heatmap_{threshold_std}sigma.png")
        plt.savefig(fin_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Plot ASR_min heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_ASR_min, annot=True, cmap="RdBu", fmt=".2f", vmin=0, vmax=1)
    plt.title(f"ASR_min Heatmap (Threshold = {threshold_std}σ)")
    plt.xlabel("Watermarker")
    plt.ylabel("Mutator")
    if save_path:
        min_path = os.path.join(save_path, f"ASR_min_heatmap_{threshold_std}sigma.png")
        plt.savefig(min_path, dpi=300, bbox_inches="tight")
    plt.show()

    if save_path:
        print(f"Saved heatmaps to {save_path}")

def plot_f1_lineplot(df, save_path_base="./attack/analysis/figs"):
    """
    Plots separate line plots for each watermarking scheme, showing ASR_fin and ASR_min scores 
    against the number of standard deviations (0,1,2,3), with unique line styles and markers per mutator.
    
    ASR_fin is always solid, and ASR_min is always dotted. Lines are smoothed using cubic interpolation.
    All plots have the same Y-axis range from 0 to 1 for consistency. Markers are added to the legend.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        save_path_base (str): The base path to save the images as .png (each watermark type will have its own file).
    """
    import scipy.interpolate
    import matplotlib.lines as mlines

    # Remove "Mutator" from mutator names
    df = df.copy()
    df["mutator"] = df["mutator"].str.replace("Mutator", "", regex=False)

    # Define desired mutator order
    mutator_order = ["Word", "EntropyWord", "Span", "Sentence", "Document", "Document1Step", "Document2Step"]
    
    # Calculate standard deviation levels (0,1,2,3) based on thresholds
    df["std_level"] = df.groupby(["watermark_type", "mutator"]).cumcount()

    # Define darker color palettes for each watermark type
    watermark_palette = {
        "Adaptive": sns.color_palette("Blues_r", len(mutator_order)),  # _r => Reverse for better visibility
        "SIR": sns.color_palette("Reds_r", len(mutator_order)),
        "KGW": sns.color_palette("Greens_r", len(mutator_order)),
    }

    # Define unique markers for each mutator
    marker_styles = ["o", "s", "D", "^", "v", "X", "*"]

    # Set seaborn style
    sns.set(style="whitegrid", font_scale=1.2)

    # Generate separate plots for each watermark type
    for watermark_type in ["Adaptive", "SIR", "KGW"]:
        if watermark_type not in df["watermark_type"].unique():
            continue  # Skip if the watermark type is not present in the data

        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # Get subset of data for the specific watermark type
        subset_df = df[df["watermark_type"] == watermark_type]

        # Assign colors and markers to mutators based on predefined order
        available_mutators = [m for m in mutator_order if m in subset_df["mutator"].unique()]
        color_map = {mutator: color for mutator, color in zip(available_mutators, watermark_palette[watermark_type])}
        marker_map = {mutator: marker_styles[i % len(marker_styles)] for i, mutator in enumerate(available_mutators)}

        legend_handles = []  # Store legend entries

        # Plot each mutator separately with its assigned color, solid/dotted lines, and markers
        for mutator in available_mutators:
            subset = subset_df[subset_df["mutator"] == mutator]
            color = color_map[mutator]
            marker = marker_map[mutator]

            # Interpolation for smoothing
            if len(subset) > 3:  # Ensure enough points for cubic interpolation
                x_new = np.linspace(subset["std_level"].min(), subset["std_level"].max(), 100)
                ASR_fin_smooth = scipy.interpolate.interp1d(subset["std_level"], subset["ASR_fin"], kind="cubic")
                ASR_min_smooth = scipy.interpolate.interp1d(subset["std_level"], subset["ASR_min"], kind="cubic")
                
                # Plot smoothed ASR_fin (solid)
                ax.plot(x_new, ASR_fin_smooth(x_new), color=color, linestyle="-", linewidth=2)

                # Plot smoothed ASR_min (dotted)
                ax.plot(x_new, ASR_min_smooth(x_new), color=color, linestyle=":", linewidth=2)

            else:
                # Plot regular lines if not enough points for smoothing
                ax.plot(subset["std_level"], subset["ASR_fin"], color=color, linestyle="-", linewidth=2)
                ax.plot(subset["std_level"], subset["ASR_min"], color=color, linestyle=":", linewidth=2)

            # Add markers to original points for visibility
            ax.scatter(subset["std_level"], subset["ASR_fin"], color=color, marker=marker, s=80)
            ax.scatter(subset["std_level"], subset["ASR_min"], color=color, marker=marker, s=80, alpha=0.8)

            # Add entry to legend with both marker and line
            legend_handles.append(mlines.Line2D([], [], color=color, marker=marker, linestyle="-", markersize=8, label=mutator))

        # Axis labels and title
        ax.set_xlabel("Standard Deviations from Unwatermarked Mean", fontsize=14)
        ax.set_ylabel("F1 Score", fontsize=14)
        ax.set_title(f"F1 Scores for {watermark_type} Watermark", fontsize=16, pad=20)

        # Set Y-axis limits to ensure all plots are on the same scale
        ax.set_ylim(-0.02, 1.02)

        # Customize x-ticks
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0σ", "1σ", "2σ", "3σ"])

        # Set legend to enforce mutator order while skipping missing ones
        ax.legend(handles=legend_handles, title="Mutator", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plot
        save_path_for_watermark = f"{save_path_base}/f1_lineplot_{watermark_type}.png"
        plt.savefig(save_path_for_watermark, dpi=300, bbox_inches="tight")

        # Show plot
        plt.show()

    if save_path_base:
        print(f"Saved line plots to {save_path_base}")


if __name__ == "__main__":

    # python -m attack.analysis.success_rate_calculator

    from attack.utils import load_all_csvs

    watermark_types = ["Adaptive", "KGW", "SIR"]
    mutators = ["DocumentMutator", "Document1StepMutator", "Document2StepMutator", 
            "SentenceMutator", "SpanMutator", "WordMutator", "EntropyWordMutator"]

    unwatermarked_mean_std = {
        "Adaptive": (49.42577, 3.365801),
        "KGW": (-0.82778, 1.047094772),
        "SIR": (0.077541, 0.068233825),
    }

    num_std = 2

    results = []
    for watermark_type in watermark_types:
        score_mean, score_std = unwatermarked_mean_std[watermark_type]
        for mutator in mutators:
            print(f"\n[MAIN] {watermark_type} + {mutator} Results:")
            df = load_all_csvs("./attack/traces/annotated", watermark_type, mutator)
            
            if df.empty:
                print(f"[MAIN] No traces found for {watermark_type} + {mutator}")
                continue

            # Filter only for rows where quality is approved
            df = df[df['quality_preserved'] == True]

            # Store original steps before filtering
            total_steps = df.groupby('group_id')['step_num'].max()
            df['original_steps'] = df['group_id'].map(total_steps)

            if "Adaptive" in watermark_type:
                df = df[~df['watermark_score'].between(-0.0001, 0.0001)]
            
            attack_metrics = process_attack_traces(df)
            
            # Compute overall averages across groups for both internLM quality and watermark scores
            # avg_init_internLM_quality = attack_metrics['init_internLM_quality'].mean()
            avg_final_internLM_quality = attack_metrics['final_internLM_quality'].mean()
            # avg_init_watermark_score = attack_metrics['init_watermark_score'].mean()
            avg_final_watermark_score = attack_metrics['final_watermark_score'].mean()
            # avg_init_perplexity = attack_metrics['init_perplexity'].mean()
            avg_final_perplexity = attack_metrics['final_perplexity'].mean()
            # avg_init_grammar_errors = attack_metrics['init_grammar_errors'].mean()
            avg_final_grammar_errors = attack_metrics['final_grammar_errors'].mean()
            # avg_init_unique_bigrams = attack_metrics['init_unique_bigrams'].mean()
            avg_final_unique_bigrams = attack_metrics['final_unique_bigrams'].mean()

            print("Attack trace sample:")
            print(attack_metrics[['group_id', 'init_watermark_score', 'min_watermark_score', 
                                  'final_watermark_score', 'total_attack_steps', 'min_score_step',
                                  'final_internLM_quality',
                                  'final_perplexity',
                                  'final_grammar_errors',
                                  'final_unique_bigrams'
                                 ]].head().to_string(index=False))
            print("\nAverages for this combination:")
            # print(f"  Avg initial watermark score:  {avg_init_watermark_score:.4f}")
            print(f"  Avg final watermark score:    {avg_final_watermark_score:.4f}")
            # print(f"  Avg initial internLM quality: {avg_init_internLM_quality:.4f}")
            print(f"  Avg final internLM quality:   {avg_final_internLM_quality:.4f}")
            # print(f"  Avg initial perplexity:       {avg_init_perplexity:.4f}")
            print(f"  Avg final perplexity:         {avg_final_perplexity:.4f}")
            # print(f"  Avg initial grammar errors:   {avg_init_grammar_errors:.4f}")
            print(f"  Avg final grammar errors:     {avg_final_grammar_errors:.4f}")
            # print(f"  Avg initial unique bigrams:   {avg_init_unique_bigrams:.4f}")
            print(f"  Avg final unique bigrams:     {avg_final_unique_bigrams:.4f}\n")
            
            min_scores = attack_metrics['min_watermark_score'].tolist()
            fin_scores = attack_metrics['final_watermark_score'].tolist()
            true_labels = [1 for _ in range(len(min_scores))]

            for name, scores in [('min', min_scores), ('fin', fin_scores)]:

                evaluator = WatermarkMetricsEvaluator(true_labels, scores)

                for i in range(4):  # to explore 3 standard deviations from mean unwatermarked

                    threshold = score_mean + (i * score_std)
                    metrics = evaluator.compute_metrics(threshold)

                    results.append({
                        "watermark_type": watermark_type,
                        "mutator": mutator,
                        "score_type": name,
                        "score_mean": score_mean,
                        "score_std": score_std,
                        "num_std": i,
                        "threshold": threshold,
                        **metrics,
                        # "avg_init_internLM_quality": avg_init_internLM_quality,
                        "avg_final_internLM_quality": avg_final_internLM_quality,
                        # "avg_init_watermark_score": avg_init_watermark_score,
                        "avg_final_watermark_score": avg_final_watermark_score,
                        # "avg_init_perplexity": avg_init_perplexity,
                        "avg_final_perplexity": avg_final_perplexity,
                        # "avg_init_grammar_errors": avg_init_grammar_errors,
                        "avg_final_grammar_errors": avg_final_grammar_errors,
                        # "avg_init_unique_bigrams": avg_init_unique_bigrams,
                        "avg_final_unique_bigrams": avg_final_unique_bigrams
                    })


    df = pd.DataFrame(results)
    if num_std:
        df = df[df['num_std']==num_std]
    df.to_csv("./attack/analysis/csv/success_rates.csv", index=False)

    df_formatted = format_success_rates(df)
    df_formatted.to_csv("./attack/analysis/csv/success_rates_formatted.csv", index=False)

    plot_f1_scores(df_formatted)
    plot_f1_heatmaps(df_formatted, threshold_std=2, 
                 watermarker_order=["KGW", "SIR", "Adaptive"],
                 mutator_order=["Word", "EntropyWord", "Span", "Sentence", 
                                "Document", "Document1Step", "Document2Step"],
                 save_path="./attack/analysis/figs")

    # Execute the function with the given dataset
    plot_f1_lineplot(df_formatted, save_path_base="./attack/analysis/figs")

    # # Find optimal F1 threshold
    # best_threshold = evaluator.find_optimal_threshold(metric='f1')
    # print(f"Optimal F1 threshold for {name}: {best_threshold:.2f}")

    # # Find threshold at 10% FPR
    # fpr_threshold = evaluator.find_threshold_at_fpr(target_fpr=0.1)
    # print(f"FPR 0.1 threshold for {name}: {fpr_threshold:.2f}")

    # # Compare metrics at different thresholds
    # print(f"\nMetrics at fpr={fpr_threshold} threshold for {name}:")
    # print(evaluator.compute_metrics(fpr_threshold))

    # print(f"\nMetrics at optimal={best_threshold} threshold for {name}:")
    # print(evaluator.compute_metrics(best_threshold))    