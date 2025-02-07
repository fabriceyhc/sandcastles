import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict

# Configure output directories
RESULTS_DIR = "./attack/analysis/"
os.makedirs(os.path.join(RESULTS_DIR, "csv"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "figs"), exist_ok=True)


def analyze_individual_traces(embedding_dict):
    """
    For each (wm_type, mutator) in the embedding_dict, compute the cosine 
    similarity between the first (step_num = -1) and final (step_num = max) 
    embeddings for each group_id. Then produce a pivot table with:
        - Rows    = unique prompts
        - Columns = mutators
        - Values  = average final similarity
    """
    all_results = []

    # Loop over all (wm_type, mutator) pairs in the embedding_dict
    for (wm_type, mutator), group_data in embedding_dict.items():
        df = group_data["data"]
        embeddings = group_data["embeddings"]

        # If there's no group_id column, skip
        if "group_id" not in df.columns:
            continue

        # Group by group_id so we can isolate each "trace"
        for group_id, trace_group in df.groupby("group_id"):
            # --- 1) Original embedding (step_num = -1) ---
            original_mask = (trace_group["step_num"] == -1)
            if not original_mask.any():
                continue  # no original embedding found
            original_index = trace_group.index[original_mask][0]
            original_embedding = embeddings[original_index].reshape(1, -1)

            # --- 2) Final embedding (step_num = max) ---
            max_step = trace_group["step_num"].max()
            final_mask = (trace_group["step_num"] == max_step)
            if not final_mask.any():
                continue  # no final embedding found
            final_index = trace_group.index[final_mask][0]
            final_embedding = embeddings[final_index].reshape(1, -1)

            # --- 3) Calculate cosine similarity ---
            similarity = cosine_similarity(original_embedding, final_embedding)[0, 0]

            # We assume `prompt` is in the trace_group DataFrame
            prompt_val = trace_group["prompt"].iloc[0]

            # Store result
            all_results.append({
                "prompt": prompt_val,
                "mutator": mutator,
                "wm_type": wm_type,  # optional, in case you want to keep track
                "group_id": group_id,
                "similarity": similarity,
            })

    # Convert the list of dicts into a DataFrame
    results_df = pd.DataFrame(all_results)

    # Group by (prompt, mutator) to get the average similarity across group_ids
    grouped = results_df.groupby(["prompt", "mutator"], as_index=False)["similarity"].mean()

    # Pivot to get a table of prompt vs. mutator
    pivot_table = grouped.pivot(index="prompt", columns="mutator", values="similarity")

    return pivot_table

def compute_prompt_retention(pivot_table, threshold=0.9):
    """
    Given a pivot table of average similarities (rows=prompt, columns=mutator),
    compute the percentage of prompts that remain above the given similarity
    threshold for each mutator column.
    """
    retention = {}
    for mutator in pivot_table.columns:
        # Drop NaN rows for this particular mutator
        col_data = pivot_table[mutator].dropna()
        total_prompts = len(col_data)
        if total_prompts == 0:
            # If no prompts for this mutator, define retention as 0.0%
            retention[mutator] = 0.0
        else:
            # Count how many are >= threshold
            above_threshold = (col_data >= threshold).sum()
            fraction_above = above_threshold / total_prompts
            retention[mutator] = fraction_above * 100.0

    # Convert to a DataFrame for easy reading
    retention_df = pd.DataFrame(
        list(retention.items()),
        columns=["mutator", f"% prompts >= {threshold:.2f}"]
    )
    return retention_df

def calculate_mutator_statistics(pivot_table):
    """Calculate comprehensive statistics for each mutator"""
    stats_df = pivot_table.describe().T.reset_index().rename(columns={'index': 'mutator'})
    
    # Calculate 95% confidence intervals
    stats_df['95% CI Lower'] = stats_df['mean'] - 1.96*stats_df['std']/np.sqrt(stats_df['count'])
    stats_df['95% CI Upper'] = stats_df['mean'] + 1.96*stats_df['std']/np.sqrt(stats_df['count'])
    
    # Reorder columns
    return stats_df[['mutator', 'count', 'mean', 'std', 
                    '95% CI Lower', '95% CI Upper',
                    'min', '25%', '50%', '75%', 'max']]

def plot_similarity_distributions(pivot_table):
    """Visualize similarity distributions with boxplots and histograms"""
    plt.figure(figsize=(14, 6))
    
    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pivot_table.melt(var_name='mutator', value_name='similarity'), 
                x='mutator', y='similarity')
    plt.axhline(0.9, color='red', linestyle='--', alpha=0.5)
    plt.title('Similarity Distribution by Mutator')
    plt.xticks(rotation=45)
    
    # Histogram
    plt.subplot(1, 2, 2)
    for mutator in pivot_table.columns:
        sns.kdeplot(pivot_table[mutator].dropna(), label=mutator, alpha=0.6)
    plt.axvline(0.9, color='red', linestyle='--', alpha=0.5)
    plt.title('Similarity Density Estimates')
    plt.xlabel('Cosine Similarity')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'figs', 'similarity_distributions.png'), dpi=300)
    plt.close()

def perform_statistical_analysis(pivot_table):
    """Run battery of statistical tests"""
    results = []
    
    # Compare each mutator to baseline (perfect similarity)
    for mutator in pivot_table.columns:
        data = pivot_table[mutator].dropna()
        if len(data) < 2:
            continue
            
        # One-sample t-test against μ=1.0
        ttest = stats.ttest_1samp(data, popmean=1.0)
        
        # Normality test
        normality = stats.shapiro(data)
        
        # Effect size
        cohen_d = (np.mean(data) - 1.0) / np.std(data, ddof=1)
        
        results.append({
            'mutator': mutator,
            'mean': np.mean(data),
            'std': np.std(data),
            't_stat': ttest.statistic,
            'p_value': ttest.pvalue,
            'cohen_d': cohen_d,
            'shapiro_p': normality.pvalue,
            'n': len(data)
        })
    
    return pd.DataFrame(results)

def analyze_retention_bins(pivot_table):
    """Categorize results into similarity bins"""
    bins = [0, 0.5, 0.7, 0.9, 1.0]
    labels = ['<50%', '50-70%', '70-90%', '≥90%']
    
    bin_results = []
    for mutator in pivot_table.columns:
        binned = pd.cut(pivot_table[mutator], bins=bins, labels=labels)
        counts = binned.value_counts(normalize=True).mul(100)
        bin_results.append(counts.rename(mutator))
    
    return pd.concat(bin_results, axis=1).fillna(0)

def analyze_stepwise_progression(embedding_dict):
    """Track similarity changes across mutation steps"""
    step_results = []
    
    for (wm_type, mutator), group_data in embedding_dict.items():
        df = group_data["data"]
        embeddings = group_data["embeddings"]

        # If there's no group_id column, skip
        if "group_id" not in df.columns:
            continue
        
        for group_id, trace_group in df.groupby("group_id"):
            steps = sorted(trace_group['step_num'].unique())
            if -1 not in steps or len(steps) < 2:
                continue
                
            original_embedding = embeddings[trace_group[trace_group['step_num'] == -1].index[0]]
            similarities = []
            
            for step in steps:
                if step == -1: 
                    similarities.append(1.0)
                    continue
                    
                step_embedding = embeddings[trace_group[trace_group['step_num'] == step].index[0]]
                similarities.append(cosine_similarity([original_embedding], [step_embedding])[0][0])
            
            step_results.append({
                'mutator': mutator,
                'steps': steps,
                'similarities': similarities
            })
    
    # Save raw progression data
    prog_df = pd.DataFrame(step_results)
    prog_df.to_csv(os.path.join(RESULTS_DIR, 'csv', 'stepwise_progression.csv'))
    
    # Calculate average trajectory
    avg_trajectory = []
    for mutator in prog_df['mutator'].unique():
        mutator_data = prog_df[prog_df['mutator'] == mutator]
        max_steps = max(len(t['steps']) for t in mutator_data.to_dict('records'))
        
        # Pad trajectories with NaNs to equal length
        padded = []
        for traj in mutator_data.to_dict('records'):
            pad_length = max_steps - len(traj['steps'])
            padded.append(traj['similarities'] + [np.nan]*pad_length)
            
        avg_trajectory.append({
            'mutator': mutator,
            'mean_similarity': np.nanmean(padded, axis=0),
            'steps': list(range(max_steps))
        })
    
    return pd.DataFrame(avg_trajectory)

def save_artifacts(artifact_dict):
    """Save multiple artifacts to appropriate locations"""
    for file_type, artifacts in artifact_dict.items():
        for filename, data in artifacts.items():
            full_path = os.path.join(RESULTS_DIR, file_type, filename)
            
            if file_type == 'csv':
                data.to_csv(full_path)
            elif file_type == 'figs':
                data.savefig(full_path, dpi=300, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    from attack.scripts.embed import load_embeddings_dict

    embedding_dict = load_embeddings_dict("./attack/traces/embeddings/")
    artifacts = {'csv': {}, 'figs': {}}

    # 1. Core analysis
    similarity_table = analyze_individual_traces(embedding_dict)
    artifacts['csv']['similarity_table.csv'] = similarity_table
    
    # 2. Retention analysis
    retention_df = compute_prompt_retention(similarity_table, 0.9)
    artifacts['csv']['retention_analysis.csv'] = retention_df
    
    # 3. Statistical analysis
    stats_df = calculate_mutator_statistics(similarity_table)
    statistical_tests = perform_statistical_analysis(similarity_table)
    artifacts['csv']['mutator_statistics.csv'] = stats_df
    artifacts['csv']['statistical_tests.csv'] = statistical_tests
    
    # 4. Distribution visualization
    plot_similarity_distributions(similarity_table)
    artifacts['figs']['similarity_distributions.png'] = plt.gcf()
    
    # 5. Retention bin analysis
    bin_df = analyze_retention_bins(similarity_table)
    artifacts['csv']['retention_bins.csv'] = bin_df
    
    # 6. Stepwise progression analysis
    progression_data = analyze_stepwise_progression(embedding_dict)
    artifacts['csv']['average_trajectories.csv'] = progression_data
    
    # Save all artifacts
    save_artifacts(artifacts)
    
    print("Analysis complete. Artifacts saved in:")
    print(f"- {RESULTS_DIR}/csv/")
    print(f"- {RESULTS_DIR}/figs/")