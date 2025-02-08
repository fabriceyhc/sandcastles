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
            'ACC': safe_div(TP + TN, len(self.true_labels))
        }

def process_attack_traces(df):
    # Drop rows with NaN in 'watermark_score' to avoid issues with idxmin()
    df = df.dropna(subset=['watermark_score'])

    # Calculate metrics from filtered data
    df_sorted = df.sort_values(by=['group_id', 'step_num'])
    
    # Get original total steps before any filtering
    if 'original_steps' not in df.columns:
        total_steps = df_sorted.groupby('group_id')['step_num'].max()
        df_sorted['original_steps'] = df_sorted['group_id'].map(total_steps)
    
    result = df_sorted.groupby('group_id').agg(
        init_watermark_score=pd.NamedAgg(column='watermark_score', aggfunc='first'),
        min_watermark_score=pd.NamedAgg(column='watermark_score', aggfunc='min'),
        final_watermark_score=pd.NamedAgg(column='watermark_score', aggfunc='last'),
        total_attack_steps=pd.NamedAgg(column='original_steps', aggfunc='first'),
        min_score_step=pd.NamedAgg(
            column='watermark_score',
            aggfunc=lambda x: df_sorted.loc[x.idxmin(), 'step_num'] if not x.isna().all() else np.nan
        )
    ).reset_index()

    return result

def format_success_rates(df):
    # Pivot the dataframe to create separate columns for "fin" and "min" metrics
    df_pivot = df.pivot_table(
        index=["watermark_type", "mutator", "threshold", "score_mean", "score_std"],
        columns="score_type",
        values=["F1"],
    )

    # Flatten the MultiIndex in columns
    df_pivot.columns = [f"{metric}_{stype}" for metric, stype in df_pivot.columns]
    df_pivot.reset_index(inplace=True)

    return df_pivot

def plot_f1_scores(df, std_index=1, save_path="./attack/analysis/figs/f1_scores.png"):
    """
    Plots F1_fin and F1_min scores for different watermarking schemes and mutators using seaborn.
    
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
    
    # Plot F1_fin and F1_min for each watermark type and mutator
    for i, watermark_type in enumerate(watermark_types):
        subset = std_data[std_data['watermark_type'] == watermark_type]
        
        # Calculate offset x-positions for this watermark type
        x_offset = x_positions + (i - len(watermark_types) / 2) * offset
        
        # Plot F1_fin and F1_min as connected lines
        for j, row in subset.iterrows():
            ax.plot([x_offset[j], x_offset[j]], [row['F1_fin'], row['F1_min']], 
                    marker='o', markersize=8, label=f'{watermark_type} - {row["mutator"]}')
    
    # Set x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(mutators, rotation=45, ha='right', fontsize=12)
    
    # Add labels and title
    ax.set_xlabel('Mutator', fontsize=14)
    ax.set_ylabel('F1 Score', fontsize=14)
    ax.set_title(f'F1_fin and F1_min Scores for Different Watermarking Schemes and Mutators (Threshold Index: {std_index})', 
                 fontsize=16, pad=20)
    
    # Add a legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path) # saves to png
    
    # Show plot
    plt.show()

if __name__ == "__main__":

    # python -m attack.analysis.success_rate_calculator

    from attack.utils import load_all_csvs

    watermark_types = ["Adaptive", "KGW", "SIR"]
    mutators = ["DocumentMutator", "Document1StepMutator", "Document2StepMutator", 
            "SentenceMutator", "SpanMutator", "WordMutator", "EntropyWordMutator"]

    unwatermarked_mean_std = {
        "Adaptive": (49.425769812530945, 3.3846567364055837),
        "KGW": (-0.8277770841872096, 1.0529608950427352),
        "SIR": (0.0775412526135454, 0.06861609056292908),
    }

    results = []
    for watermark_type in watermark_types:
        score_mean, score_std = unwatermarked_mean_std[watermark_type]
        for mutator in mutators:
            print(f"\n[MAIN] {watermark_type} + {mutator} Results:")
            df = load_all_csvs("./attack/traces/annotated", watermark_type, mutator)
            
            if df.empty:
                print(f"[MAIN] No traces found for {watermark_type} + {mutator}")
                continue

            # Store original steps before filtering
            total_steps = df.groupby('group_id')['step_num'].max()
            df['original_steps'] = df['group_id'].map(total_steps)

            if "Adaptive" in watermark_type:
                df = df[~df['watermark_score'].between(-0.0001, 0.0001)]
            
            attack_metrics = process_attack_traces(df)
            
            print(attack_metrics[['group_id', 'init_watermark_score', 'min_watermark_score', 
                                'final_watermark_score', 'total_attack_steps',
                                'min_score_step']].head().to_string(index=False))

            min_scores = attack_metrics['min_watermark_score'].tolist()
            fin_scores = attack_metrics['final_watermark_score'].tolist()
            true_labels = [1 for _ in range(len(min_scores))]

            for name, scores in [('min', min_scores), ('fin', fin_scores)]:

                evaluator = WatermarkMetricsEvaluator(true_labels, scores)

                for i in range(4): # to explore 3 standard deviations from mean unwatermarked

                    threshold = score_mean + (i * score_std)
                    metrics = evaluator.compute_metrics(threshold)

                    results.append({
                        "watermark_type": watermark_type,
                        "mutator": mutator,
                        "score_type": name,
                        "score_mean": score_mean,
                        "score_std": score_std,
                        "threshold": threshold,
                        **metrics
                    })

    df = pd.DataFrame(results)
    df.to_csv("./attack/analysis/csv/success_rates.csv", index=False)

    df_formatted = format_success_rates(df)
    df_formatted.to_csv("./attack/analysis/csv/success_rates_formatted.csv", index=False)

    plot_f1_scores(df_formatted)

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