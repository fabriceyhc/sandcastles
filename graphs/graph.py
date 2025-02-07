import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Levenshtein import distance as levenshtein_distance
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import matplotlib.colors as mcolors
import umap

def calculate_similarity_matrix(df, similarity_method, model=None):
    """
    Calculate a similarity matrix for all pairs of texts in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the texts in 'current_text'.
        similarity_method (str): Similarity method ('semantic' or 'levenshtein').
        model (SentenceTransformer, optional): Pretrained model for semantic similarity.

    Returns:
        np.ndarray: A square similarity matrix.
    """
    texts = df['current_text'].tolist()
    n = len(texts)
    similarity_matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Calculating similarities", unit="pair"):
        for j in range(i + 1, n):
            similarity = calculate_similarity(similarity_method, texts[i], texts[j], model)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrix

    return similarity_matrix

def calculate_similarity(method, text1, text2, model=None):
    """
    Calculate similarity between two texts using the specified method.

    Parameters:
        method (str): Similarity method ('semantic' or 'levenshtein').
        text1 (str): First text.
        text2 (str): Second text.
        model (SentenceTransformer): Pretrained model for semantic similarity.

    Returns:
        float: Similarity score.
    """
    if method == 'semantic':
        if not model:
            raise ValueError("A SentenceTransformer model is required for semantic similarity.")
        embeddings = model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    elif method == 'levenshtein':
        max_len = max(len(text1), len(text2))
        return 1 - levenshtein_distance(text1, text2) / max_len

    else:
        raise ValueError("Unsupported similarity method. Use 'semantic' or 'levenshtein'.")

def assign_colors_and_markers(df):
    """
    Assign unique colors for each prompt and unique markers for each group_id within each prompt.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        dict: Mapping of node index to color and marker attributes.
    """
    # Add a prompt_id to identify unique prompts
    df['prompt_id'] = df['prompt'].astype('category').cat.codes

    # Get unique colors and markers
    unique_colors = list(mcolors.TABLEAU_COLORS.values())
    unique_markers = ['o', 's', '^', 'D', 'P', '*']  # Extend as needed

    color_map = {pid: unique_colors[i % len(unique_colors)] for i, pid in enumerate(df['prompt_id'].unique())}

    attributes = {}
    idx = 0

    for prompt_id, group in df.groupby('prompt_id'):  # Group by prompt_id
        marker_map = {gid: unique_markers[i % len(unique_markers)] for i, gid in enumerate(group['group_id'].unique())}

        for _, row in group.iterrows():
            attributes[idx] = {
                'color': color_map[prompt_id],
                'marker': marker_map[row['group_id']]
            }
            idx += 1

    return attributes

def create_similarity_graph(df, similarity_matrix, threshold):
    """
    Create a similarity graph from a similarity matrix.

    Parameters:
        df (pd.DataFrame): DataFrame with a column 'current_text'.
        similarity_matrix (np.ndarray): Precomputed similarity matrix.
        threshold (float): Minimum similarity score to create an edge.

    Returns:
        nx.Graph: Graph with texts as nodes and similarity edges.
    """
    # Ensure 'current_text' column exists
    if 'current_text' not in df.columns:
        raise ValueError("The DataFrame must contain a 'current_text' column.")

    texts = df['current_text'].tolist()

    # Create a graph
    G = nx.Graph()

    # Add nodes
    attributes = assign_colors_and_markers(df)
    for i, text in enumerate(texts):
        G.add_node(i, text=text, **attributes[i])

    # Add edges based on similarity matrix and threshold
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i, j] >= threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    return G

def plot_graph(G, output_path="graph_plot.png", title="Similarity Graph"):
    """
    Plot the similarity graph using Matplotlib for large-scale graphs.

    Parameters:
        G (nx.Graph): The graph to plot.
        output_path (str): Path to save the graph plot as a PNG image.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42)  # Use spring layout for visualization

    # Draw edges
    weights = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color=list(weights.values()), edge_cmap=plt.cm.Blues)

    # Draw nodes with colors and markers
    for node, attr in nx.get_node_attributes(G, 'color').items():
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_color=[attr],
            node_shape=G.nodes[node]['marker'],
            node_size=50,
            alpha=0.8
        )

    plt.title(title, fontsize=24)
    plt.axis('off')
    plt.savefig(output_path, format='png', dpi=300)
    plt.close()
    print(f"Graph plot saved to {output_path}")

def create_umap_visualization(df, embeddings, output_path="umap_plot.png", title="UMAP Visualization"):
    """
    Create a UMAP visualization for the texts using embeddings.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        embeddings (np.ndarray): Precomputed embeddings of the texts.
        output_path (str): Path to save the UMAP plot as a PNG image.
    """
    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(metric="cosine", random_state=42)
    embedding = reducer.fit_transform(embeddings)

    # Get colors and markers
    attributes = assign_colors_and_markers(df)
    colors = [attributes[i]['color'] for i in range(len(df))]
    markers = [attributes[i]['marker'] for i in range(len(df))]

    # Plot UMAP
    plt.figure(figsize=(12, 12))
    for marker in set(markers):
        indices = [i for i, m in enumerate(markers) if m == marker]
        plt.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            c=[colors[i] for i in indices],
            label=f"Marker: {marker}",
            alpha=0.8,
            s=50,
            edgecolors="k",
            marker=marker
        )

    plt.title(title, fontsize=16)
    plt.legend(fontsize=10, loc="best")
    plt.axis("off")
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()
    print(f"UMAP plot saved to {output_path}")

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0 python -m graphs.graph

    from attack.utils import load_all_csvs

    watermark_types = [
        "Adaptive",
        "KGW",
        "GPT4o_unwatermarked",
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

    for similarity_method in ["semantic"]:

        for watermark_type in watermark_types:
            
            for mutator in mutators:
                
                n = 10
                df = load_all_csvs("./attack/traces", watermark_type, mutator).iloc[::n]

                if df.empty:
                    print(f"[MAIN] No traces found for {watermark_type} + {mutator}")
                    continue

                # Load a pretrained model for semantic similarity (if required)
                model = None
                if similarity_method == "semantic":
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    embeddings = model.encode(df['current_text'].tolist())

                # Calculate similarity matrix once
                similarity_matrix = calculate_similarity_matrix(df, similarity_method, model)

                for threshold in [0.5, 0.75, 0.9]:

                    id_ = f"{watermark_type}_{mutator}_{similarity_method}_@_{threshold}"

                    graph_path = f"./graphs/saved/gexf/{id_}_similarity_graph.gexf"
                    plot_path = f"./graphs/saved/img/{id_}_graph_plot.png"
                    umap_path = f"./graphs/saved/img/{id_}_umap_plot.png"

                    # Check if the graph already exists
                    if os.path.exists(graph_path):
                        print(f"[MAIN] Graph already exists for {id_}. Loading...")
                        graph = nx.read_gexf(graph_path)
                    else:
                        print(f"[MAIN] Processing {id_}")
                        print(df.head(5))
                        print(df.info())

                        # Create the graph
                        graph = create_similarity_graph(df, similarity_matrix, threshold)

                        # Save the graph
                        nx.write_gexf(graph, graph_path)
                        print(f"Graph saved to {graph_path}")

                    # Plot the graph with a title
                    plot_title = f"Similarity Graph for {id_}"
                    plot_graph(graph, plot_path, title=plot_title)

                    if similarity_method == "semantic":
                        # Create UMAP visualization
                        plot_title = f"UMAP Plot for {id_}"
                        create_umap_visualization(df, embeddings, umap_path, title=plot_title)
