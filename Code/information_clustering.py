import os
import pandas as pd
import torch
from transformers import BigBirdTokenizer, BigBirdModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import BisectingKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
import seaborn as sns

def create_document_embeddings(data_dirs, metadata_file, output_embeddings, output_labels):
    # Load metadata
    metadata = pd.read_csv(metadata_file)

    # Load tokenizer and model
    tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-large")
    model = BigBirdModel.from_pretrained("google/bigbird-roberta-large")

    # Function to process and embed a document
    def embed_document(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        # Tokenize and split into chunks of 4096 tokens
        tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=False, max_length=4096)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Get embeddings for each chunk
        embeddings = []
        token_counts = []

        for i in range(0, input_ids.size(1), 4096):
            chunk_input_ids = input_ids[:, i:i+4096]
            chunk_attention_mask = attention_mask[:, i:i+4096]

            with torch.no_grad():
                outputs = model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
                chunk_embedding = outputs.last_hidden_state.mean(dim=1)

            embeddings.append(chunk_embedding)
            token_counts.append(chunk_input_ids.size(1))  # Track token count for weighting

        # Weighted average of embeddings based on token count
        embeddings = torch.cat(embeddings, dim=0)  # Concatenate chunk embeddings
        token_counts = torch.tensor(token_counts, dtype=torch.float32)

        weighted_avg_embedding = torch.sum(embeddings * token_counts[:, None], dim=0) / torch.sum(token_counts)

        return weighted_avg_embedding

    # Prepare outputs
    embedding_list = []
    labels_list = []

    # Iterate through directories and documents
    for directory in data_dirs:
        dir_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
        for file_name in tqdm(dir_files, desc=f"Processing {directory}"):
            filepath = os.path.join(directory, file_name)
            embedding = embed_document(filepath)

            # Add embedding to list
            embedding_list.append(embedding)

            # Add metadata to labels
            meta_row = metadata[metadata["document"] == file_name]
            if meta_row.empty:
                print(f"Warning: No metadata match found for file {file_name} in directory {directory}")
                labels_list.append({"document": file_name, "directory": directory, "case": None, "case label": None})
            else:
                labels_list.append(meta_row.iloc[0].to_dict())

    # Convert embeddings and labels to final formats
    embedding_matrix = torch.stack(embedding_list)
    labels_df = pd.DataFrame(labels_list)

    # Save outputs
    torch.save(embedding_matrix, output_embeddings)
    labels_df.to_csv(output_labels, index=False)

    print(f"Embeddings saved to {output_embeddings}")
    print(f"Labels saved to {output_labels}")

def perform_clustering_and_visualize(embeddings_file, labels_file, n_clusters, output_plot):
    # Load embeddings and labels
    embeddings = torch.load(embeddings_file).numpy()
    labels_df = pd.read_csv(labels_file)

    # Perform k-means clustering
    kmeans = BisectingKMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # Add cluster assignments to labels
    labels_df['cluster'] = clusters

    # Calculate ARI
    ARI = adjusted_rand_score(labels_df["case label"], labels_df["cluster"])
    print(f"Clustering adjusted rand score: {ARI}")

    # Dimensionality reduction with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    tsne_results = tsne.fit_transform(embeddings)

    # Define case label names
    case_label_names = [
        "Burwell v. Hobby Lobby", 
        "California v. Texas", 
        "King v. Burwell", 
        "Little Sisters v. Pennsylvania", 
        "Maine Community Health v. US", 
        "NFIB v. Sebelius"
    ]

    # Map case labels to names
    labels_df['case label name'] = labels_df['case label'].map(lambda x: case_label_names[x])

    # Plot results
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        x=tsne_results[:, 0], 
        y=tsne_results[:, 1], 
        hue=labels_df['cluster'],        # Clustered case labels
        style=labels_df['case label name'], # True case labels with names
        palette='tab10', 
        s=100
    )

    # Update legend labels
    handles, labels = scatter.get_legend_handles_labels()

    # Modify legend titles
    new_labels = ["Clustered Case Label" if label == "cluster" else label for label in labels]
    new_labels = ["True Case Label" if label == "case label name" else label for label in new_labels]
    scatter.legend(handles, new_labels, title="Legend")

    # Final plot settings
    plt.title("Visualization of Amicus Curiae Brief Clusters")
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.show()

