import os
import pandas as pd
import numpy as np
import h5py
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import linregress
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict


def analyze_prompt_trends(embedding_dict):
    """Analyze embedding trends per prompt with statistical testing"""
    results = defaultdict(list)
    
    for (wm_type, mutator), group_data in embedding_dict.items():
        df = group_data['data']
        embeddings = group_data['embeddings']
        
        # Group by prompt and analyze each separately
        for prompt, prompt_group in df.groupby('prompt'):
            # Get original embedding (step_num = -1)
            original_mask = (prompt_group['step_num'] == -1)
            original_embeddings = embeddings[prompt_group.index[original_mask]]
            
            if len(original_embeddings) == 0:
                continue
                
            # Get mutation steps (sorted, excluding original)
            steps = sorted(prompt_group[prompt_group['step_num'] >= 0]['step_num'].unique())
            if len(steps) < 2:
                continue
                
            # Calculate similarity trajectory
            similarities = []
            valid_steps = []
            
            for step in steps:
                step_mask = (prompt_group['step_num'] == step)
                step_embeddings = embeddings[prompt_group.index[step_mask]]
                
                if len(step_embeddings) == 0:
                    continue
                
                # Calculate average similarity to original
                sim_matrix = cosine_similarity(step_embeddings, original_embeddings)
                avg_sim = np.nanmean(sim_matrix)
                similarities.append(avg_sim)
                valid_steps.append(step)
            
            if len(similarities) < 2:
                continue
                
            # Calculate trend statistics
            slope, _, _, p_value, _ = linregress(valid_steps, similarities)
            
            # Classify trend
            if p_value > 0.05:
                trend = 'stable'
            else:
                trend = 'diverging' if slope < 0 else 'converging'
                
            results[(wm_type, mutator)].append(trend)
    
    return calculate_trend_percentages(results)

def calculate_trend_percentages(results):
    """Calculate percentage of diverging/converging/stable prompts"""
    summary = []
    
    for (wm_type, mutator), trends in results.items():
        total = len(trends)
        counts = {
            'diverging': sum(1 for t in trends if t == 'diverging'),
            'converging': sum(1 for t in trends if t == 'converging'),
            'stable': sum(1 for t in trends if t == 'stable')
        }
        percentages = {k: v/total*100 for k, v in counts.items()}
        
        summary.append({
            'Watermark': wm_type,
            'Mutator': mutator,
            '% Diverging': f"{percentages['diverging']:.1f}%",
            '% Converging': f"{percentages['converging']:.1f}%", 
            '% Stable': f"{percentages['stable']:.1f}%",
            'Total Prompts': total
        })
    
    return pd.DataFrame(summary)

def plot_trend_distribution(embedding_dict, save_path="./attack/analysis/figs/convergence.png"):
    """Visualize trend distribution across configurations"""
    plt.figure(figsize=(12, 6))
    
    for idx, ((wm_type, mutator), group_data) in enumerate(embedding_dict.items()):
        df = group_data['data']
        embeddings = group_data['embeddings']
        
        slopes = []
        p_values = []
        
        for prompt, prompt_group in df.groupby('prompt'):
            original_mask = (prompt_group['step_num'] == -1)
            original_embeddings = embeddings[prompt_group.index[original_mask]]
            
            if len(original_embeddings) == 0:
                continue
                
            steps = sorted(prompt_group[prompt_group['step_num'] >= 0]['step_num'].unique())
            if len(steps) < 2:
                continue
                
            similarities = []
            valid_steps = []
            
            for step in steps:
                step_mask = (prompt_group['step_num'] == step)
                step_embeddings = embeddings[prompt_group.index[step_mask]]
                
                if len(step_embeddings) == 0:
                    continue
                
                sim_matrix = cosine_similarity(step_embeddings, original_embeddings)
                avg_sim = np.nanmean(sim_matrix)
                similarities.append(avg_sim)
                valid_steps.append(step)
            
            if len(similarities) >= 2:
                slope, _, _, p_val, _ = linregress(valid_steps, similarities)
                slopes.append(slope)
                p_values.append(p_val)
        
        # Create jittered scatter plot
        jitter = np.random.normal(0, 0.1, len(slopes)) if slopes else []
        if slopes:  # Only plot if there's data
            plt.scatter(
                x=[idx + jit for jit in jitter],
                y=slopes,
                c=['red' if p < 0.05 else 'gray' for p in p_values],
                alpha=0.6,
                s=40,
                edgecolor='w',
                label=f"{wm_type} {mutator}"
            )
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Embedding Similarity Trends by Watermark-Mutator Combination")
    plt.ylabel("Slope (Negative = Diverging, Positive = Converging)")
    plt.xticks(range(len(embedding_dict)), 
               [f"{k[0]}\n{k[1]}" for k in embedding_dict.keys()],
               rotation=45,
               ha='right')
    plt.colorbar(label='Statistical Significance (p < 0.05)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    # python -m attack.analysis.convergence

    from attack.scripts.embed import load_embeddings_dict
    
    EMBEDDINGS_DIR = "./attack/traces/embeddings/"  # Update this path as needed

    # Load embeddings
    embedding_dict = load_embeddings_dict(EMBEDDINGS_DIR)

    # Analyze trends
    print("\nAnalyzing prompt-level trends...")
    trend_df = analyze_prompt_trends(embedding_dict)
    trend_df.to_csv("./attack/analysis/csv/convergence.csv", index=False)
    
    if not trend_df.empty:
        print("\nTrend Analysis Results:")
        print(trend_df.to_string(index=False))
    else:
        print("No valid trends found - check input data")
    
    # Generate visualization
    print("\nGenerating trend visualization...")
    plot_trend_distribution(embedding_dict)