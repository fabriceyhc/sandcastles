# RUN: CUDA_VISIBLE_DEVICES=0 python -m distinguisher.analyze

import pandas as pd
import torch
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz.distance import Levenshtein

import matplotlib
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

matplotlib.use('Agg')

def assign_unique_group_ids(df):
    df['new_group'] = (df['step_num'] == 0).astype(int)
    df['group_id'] = df['new_group'].cumsum()
    return df

def assign_prompt_ids(df, prompt_column="prompt"):
    df["prompt_id"] = pd.factorize(df[prompt_column])[0]
    return df

def assign_shape_ids(df, prompt_column="prompt_id", group_column="group_id"):
    df = df.sort_values(by=[prompt_column, group_column])
    df["shape_id"] = 0
    df["shape_id"] = (
        df.groupby(prompt_column)[group_column]
        .transform(lambda x: pd.factorize(x)[0])
    )
    return df

def compute_similarity_matrix(texts, metric, model_name='dunzhang/stella_en_400M_v5', cache_folder="/data2/.shared_models/"):
    """
    Compute pairwise similarity matrix using the specified metric.

    Args:
        texts (list): List of input texts.
        metric (str): Similarity metric ('levenshtein', 'tfidf', or 'sbert').
        model_name (str): Name of the SBERT model (used for 'sbert').
        cache_folder (str): Folder to cache the SBERT model.

    Returns:
        torch.Tensor or numpy.ndarray: Pairwise similarity matrix.
    """
    if metric == "levenshtein":
        import numpy as np
        num_texts = len(texts)
        similarity_matrix = np.zeros((num_texts, num_texts))
        for i in range(num_texts):
            for j in range(i + 1, num_texts):
                distance = Levenshtein.distance(texts[i], texts[j])
                similarity = 1 - distance / max(len(texts[i]), len(texts[j]))
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric
        return similarity_matrix

    elif metric == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix

    elif metric == "sbert":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(model_name, cache_folder=cache_folder, trust_remote_code=True).to(device)
        embeddings = model.encode(texts, convert_to_tensor=True, device=device)
        similarity_matrix = torch.nn.functional.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
        )
        return similarity_matrix

    else:
        raise ValueError("Invalid similarity metric. Choose from 'levenshtein', 'tfidf', or 'sbert'.")


# Graph creation function
def create_graph_from_similarity(similarity_matrix, threshold):
    """
    Create a graph from a precomputed similarity matrix.

    Args:
        similarity_matrix (torch.Tensor or numpy.ndarray): Pairwise similarity matrix.
        threshold (float): Threshold for adding edges.

    Returns:
        nx.Graph: Graph with nodes and edges based on similarity.
    """
    graph = nx.Graph()
    num_nodes = similarity_matrix.shape[0]
    graph.add_nodes_from(range(num_nodes))  # Add nodes for each document

    if isinstance(similarity_matrix, torch.Tensor):
        edges = [
            (i, j, float(similarity_matrix[i, j]))
            for i in range(num_nodes)
            for j in range(i + 1, num_nodes)
            if similarity_matrix[i, j] >= threshold
        ]
    else:  # NumPy array
        edges = [
            (i, j, similarity_matrix[i, j])
            for i in range(num_nodes)
            for j in range(i + 1, num_nodes)
            if similarity_matrix[i, j] >= threshold
        ]

    graph.add_weighted_edges_from(edges)
    return graph

def plot_network(graph,
                 prompt_ids=None,
                 shape_ids=None,
                 sample_size=1000,
                 layout='spring',
                 title="Graph Visualization",
                 output_file="graph_plot.png"):
    """
    Plot a graph, using colors for prompt_id and shapes for shape_id.

    Args:
        graph (nx.Graph): The input graph.
        prompt_ids (list): List of prompt IDs corresponding to the nodes (for colors).
        shape_ids (list): List of shape IDs corresponding to the nodes (for shapes).
        sample_size (int): Number of nodes to sample (set to None for full graph).
        layout (str): Layout algorithm ('spring', 'kamada_kawai', etc.).
        title (str): The title of the plot.
        output_file (str): Path to save the output plot.
    """
    import matplotlib.patches as mpatches

    # Sample nodes if graph is too large
    if sample_size and len(graph.nodes) > sample_size:
        sampled_nodes = list(graph.nodes)[:sample_size]
        sampled_graph = graph.subgraph(sampled_nodes)
        print(f"Sampled {len(sampled_graph.nodes)} nodes and {len(sampled_graph.edges)} edges.")
    else:
        sampled_graph = graph

    # Get positions for the layout
    if layout == 'spring':
        pos = nx.spring_layout(sampled_graph, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(sampled_graph)
    else:
        raise ValueError("Unsupported layout. Choose 'spring' or 'kamada_kawai'.")

    # Validate prompt_ids and shape_ids
    if prompt_ids is None or shape_ids is None:
        raise ValueError("Both prompt_ids and shape_ids must be provided.")

    # Factorize and validate the number of shapes and colors
    prompt_colors = pd.factorize(prompt_ids)[0]
    shape_shapes = pd.factorize(shape_ids)[0]

    num_colors = len(set(prompt_colors))
    num_shapes = len(set(shape_shapes))

    # Define available shapes
    shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    if num_shapes > len(shapes):
        raise ValueError(f"Too many unique shape_ids ({num_shapes}). Add more shapes or reduce groups.")

    # Define color map
    color_map = plt.cm.tab10

    # Plot each shape separately
    plt.figure(figsize=(12, 12))
    for shape_idx in range(num_shapes):
        nodes = [
            node
            for node, shape_val in zip(sampled_graph.nodes, shape_shapes)
            if shape_val == shape_idx
        ]
        node_colors = [prompt_colors[node] for node in nodes]
        nx.draw_networkx_nodes(
            sampled_graph,
            pos,
            nodelist=nodes,
            node_color=node_colors,
            cmap=color_map,
            node_shape=shapes[shape_idx],
            node_size=25,
            alpha=0.8
        )

    # Draw edges
    nx.draw_networkx_edges(sampled_graph, pos, alpha=0.3, edge_color="gray")

    # Create legends
    color_legend = [
        mpatches.Patch(color=color_map(i / num_colors), label=f"prompt_id={i}")
        for i in range(num_colors)
    ]
    shape_legend = [
        mpatches.Patch(color="gray", label=f"trial_{idx}={shapes[idx]}", alpha=0.6)
        for idx in range(num_shapes)
    ]

    plt.legend(handles=color_legend + shape_legend, loc='upper right', fontsize='small')
    plt.title(title)
    plt.savefig(output_file, format="png", dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    import glob

    traces = glob.glob("./distinguisher/attack_traces/*original=True_attack_results.csv")
    traces.sort()

    n_samples = 1000
    save_path = f"./distinguisher/graphs/analysis_log.csv"

    results = []
    for trace in traces:
        print(f"Processing {trace}...")
        df = assign_shape_ids(
            assign_prompt_ids(
                assign_unique_group_ids(pd.read_csv(trace))
            )
        )
        df["mutated_text"] = df["mutated_text"].fillna(df["current_text"])
        df['current_text'] = df["mutated_text"].shift(1)
        df["current_text"] = df["current_text"].fillna(df["mutated_text"])
        df = df.sample(n=n_samples)

        print(df.sort_values(["group_id", "prompt_id", "shape_id"]))

        mutator_type = trace.split("_")[3]

        for similarity_metric in ['levenshtein', 'sbert', 'tfidf']:
            print(f"Computing {similarity_metric} similarity for {mutator_type}...")
            similarity_matrix = compute_similarity_matrix(
                df['mutated_text'].tolist(),
                metric=similarity_metric
            )

            for threshold in [0.5, 0.75, 0.9, 0.95]:
                print(f"Creating graph for {mutator_type} with {similarity_metric} similarity @ threshold {threshold}...")
                graph = create_graph_from_similarity(similarity_matrix, threshold)

                plot_network(
                    graph,
                    prompt_ids=df["prompt_id"].tolist(),
                    shape_ids=df["shape_id"].tolist(),
                    sample_size=1000,
                    layout='spring',
                    title=f"{mutator_type}_{similarity_metric}_@_{threshold}",
                    output_file=f"./distinguisher/graphs/imgs/{mutator_type}_{similarity_metric}_@_{threshold}.png"
                )

                output_file = f"./distinguisher/graphs/graphml/{mutator_type}_{similarity_metric}_@_{threshold}.graphml"
                nx.write_graphml(graph, output_file)
                print(f"Graph saved to {output_file}. Use tools like Gephi to visualize it.")

                out = {
                    "trace": trace,
                    "mutator_type": mutator_type,
                    "similarity_metric": similarity_metric,
                    "threshold": threshold,
                    "num_nodes": len(graph.nodes),
                    "num_edges": len(graph.edges),
                    "num_groups": len(df["group_id"].unique()),
                    "graph_file": output_file,
                }
                results.append(out)
                print(f"Results: {out}")

                # Append new results to existing annotations and save to CSV
                results_df = pd.DataFrame(results)
                results_df.to_csv(save_path, index=False)


    # # RUN: CUDA_VISIBLE_DEVICES=3,4 python -m attack.annotate_traces

    # import os
    # import glob
    # import traceback
    # from extractors import FluencyMetric, GrammarMetric, QualityMetric, EditsMetric
        
    # # Initialize metric extractors
    # fluency = FluencyMetric()
    # grammar = GrammarMetric()
    # quality = QualityMetric()
    # edits   = EditsMetric()

    # traces = glob.glob("./distinguisher/attack_traces/*original=True_attack_results.csv")

    # for trace in traces:

    #     print(trace)

    #     o, w, m, s = os.path.basename(trace).split("_")[:4]
    #     s = int(s.replace("n-steps=", ""))
        
    #     df = assign_unique_group_ids(pd.read_csv(trace))
    #     df["mutated_text"] = df["mutated_text"].fillna(df["current_text"])
    #     df['current_text'] = df['mutated_text'].shift(1)
    #     df["current_text"] = df["current_text"].fillna(df["mutated_text"])

    #     # step_num,mutation_num,prompt,current_text,mutated_text,current_text_len,mutated_text_len,length_issue,quality_analysis,quality_preserved,watermark_detected,watermark_score,backtrack,total_time,mutator_time,oracle_time
    #     if "words_edited" not in df.columns:
    #         try:
    #             df = edits.evaluate_dataframe(df, current_text_column="current_text", mutated_text_column="mutated_text", new_column="words_edited")
    #         except:
    #             print(f"{'=' * 50} words_edited {'=' * 50}")
    #             print(traceback.format_exc())
    #     if "perplexity" not in df.columns:
    #         try:
    #             df = fluency.evaluate_dataframe(df, text_column="mutated_text", new_column="perplexity")
    #         except:
    #             print(f"{'=' * 50} perplexity {'=' * 50}")
    #             print(traceback.format_exc())
    #     if "grammar_errors" not in df.columns:    
    #         try:
    #             df = grammar.evaluate_dataframe(df, text_column="mutated_text", new_column="grammar_errors")
    #         except:
    #             print(f"{'=' * 50} grammar_errors {'=' * 50}")
    #             print(traceback.format_exc())
    #     if "internlm_quality" not in df.columns:
    #         try:
    #             df = quality.evaluate_dataframe(df, prompt_column="prompt", text_column="mutated_text", new_column="internlm_quality")
    #         except:
    #             print(f"{'=' * 50} internlm_quality {'=' * 50}")
    #             print(traceback.format_exc())

    #     print(df)
    #     print(trace)
    #     df.to_csv(trace, index=False)
