import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

def plot_similarity_heatmap(similarity_matrix, case_name, figsize=(10, 8), cmap='coolwarm'):
    """
    Create a heatmap from a semantic similarity matrix with a case name.

    Parameters:
    - similarity_matrix: numpy array, the similarity matrix.
    - case_name: str, name of the Supreme Court case.
    - figsize: tuple, size of the figure.
    - cmap: str, color map to use for the heatmap.
    """
    # Create a heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, cmap=cmap, annot=False, fmt=".2f", cbar=True)

    # Add labels and title
    plt.title(f"Semantic Similarity Heatmap for {case_name}", fontsize=16)
    plt.xlabel('Sentences from Amici', fontsize=12)
    plt.ylabel('Sentences from Opinions', fontsize=12)

    # Show the plot
    plt.show()

