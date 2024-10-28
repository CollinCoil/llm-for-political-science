import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

def plot_similarity_heatmap(similarity_matrix, figsize=(10, 8), cmap='coolwarm', title='Semantic Similarity Heatmap'):
    """
    Create a heatmap from a semantic similarity matrix.

    Parameters:
    - matrix_file: numpy array, similarity matrix.
    - figsize: tuple, size of the figure.
    - cmap: str, color map to use for the heatmap.
    - title: str, title of the heatmap.
    """

    similarity_matrix_imprecise = similarity_matrix.astype("float16")

    # Create a heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix_imprecise, cmap=cmap, annot=False, fmt=".2f", cbar=True)

    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Sentences from Opinions', fontsize=12)
    plt.ylabel('Sentences from Amici', fontsize=12)

    # Show the plot
    plt.show()


def visualize_semantic_network(similarity_matrix, threshold=0.75, labels_1=None, labels_2=None, figsize=(12, 12)):
    """
    Visualize a semantic similarity network from a non-square similarity matrix.
    Removes isolated nodes (nodes with no connections).

    Parameters:
    - similarity_matrix: np.ndarray, the matrix containing semantic similarities (rows: dataset 1, columns: dataset 2).
    - threshold: float, the threshold for drawing links between sentences.
    - labels_1: list of str, optional, labels for the nodes from the first dataset (Amici).
    - labels_2: list of str, optional, labels for the nodes from the second dataset (Opinions).
    - figsize: tuple, size of the figure for visualization.
    """
    # Create a graph from the similarity matrix
    G = nx.Graph()

    # Get the number of sentences in each dataset
    num_sentences_1 = similarity_matrix.shape[0]
    num_sentences_2 = similarity_matrix.shape[1]

    # Add nodes for both datasets
    if labels_1 is None:
        labels_1 = [f'Amici Sentence {i}' for i in range(num_sentences_1)]
    if labels_2 is None:
        labels_2 = [f'Opinions Sentence {i}' for i in range(num_sentences_2)]

    # Efficiently find indices where the similarity exceeds the threshold
    rows, cols = np.where(similarity_matrix > threshold)

    # Create edges based on the indices
    edges = [(r, c + num_sentences_1) for r, c in zip(rows, cols)]

    # Add edges to the graph
    G.add_edges_from(edges)

    # Get the connected nodes
    connected_nodes = set(node for edge in edges for node in edge)
    
    # Split connected nodes into two groups
    connected_nodes_1 = {n for n in connected_nodes if n < num_sentences_1}
    connected_nodes_2 = {n for n in connected_nodes if n >= num_sentences_1}

    # Create filtered labels dictionary only for connected nodes
    labels = {}
    for n in connected_nodes_1:
        labels[n] = labels_1[n]
    for n in connected_nodes_2:
        labels[n] = labels_2[n - num_sentences_1]

    # Draw the network
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G)  # Positioning for visual clarity
    
    # Create weights based on the similarity scores
    weights = [similarity_matrix[r, c] for r, c in zip(rows, cols)]

    # Draw nodes with different colors for each dataset
    if connected_nodes_1:  # Only draw if there are connected nodes in dataset 1
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=list(connected_nodes_1),
                              node_color='skyblue',
                              node_size=300,
                              label='Amici')
    
    if connected_nodes_2:  # Only draw if there are connected nodes in dataset 2
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=list(connected_nodes_2),
                              node_color='lightgreen',
                              node_size=300,
                              label='Opinions')

    # Draw edges with weights
    nx.draw_networkx_edges(G, pos, width=np.array(weights) * 5, alpha=0.5)

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    # Add summary statistics to the title
    title = f'Semantic Similarity Network\n'
    title += f'Connected Nodes: {len(connected_nodes)} out of {num_sentences_1 + num_sentences_2}\n'
    title += f'(Amici: {len(connected_nodes_1)}/{num_sentences_1}, '
    title += f'Opinions: {len(connected_nodes_2)}/{num_sentences_2})'
    
    plt.title(title, fontsize=16)
    plt.legend()
    plt.axis('off')
    plt.show()
    
    # Return statistics about the network
    return {
        'total_nodes': num_sentences_1 + num_sentences_2,
        'connected_nodes': len(connected_nodes),
        'connected_amici': len(connected_nodes_1),
        'connected_opinions': len(connected_nodes_2),
        'total_edges': len(edges),
        'density': len(edges) / (len(connected_nodes) * (len(connected_nodes) - 1) / 2) if len(connected_nodes) > 1 else 0
    }