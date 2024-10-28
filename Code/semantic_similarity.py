import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

def load_sentences_from_jsonl(file_path):
    """
    Load sentences from a JSON Lines file.
    Each line should contain a dictionary with a "text" field.
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentences.append(data["text"])
    return sentences

def calculate_semantic_similarity(jsonl_file_1, jsonl_file_2, model_name="sentence-transformers/all-mpnet-base-v2", output_file_location = "Data/semantic_similarity_matrix.npy"):
    """
    Calculate semantic similarity between sentences from two JSONL files.
    """
    # Load sentences from both JSONL files
    sentences_1 = load_sentences_from_jsonl(jsonl_file_1)
    sentences_2 = load_sentences_from_jsonl(jsonl_file_2)

    # Load the pre-trained Sentence Transformer model
    model = SentenceTransformer(model_name)

    # Create embeddings for the sentences from both sets
    embeddings_1 = model.encode(sentences_1, convert_to_tensor=True)
    embeddings_2 = model.encode(sentences_2, convert_to_tensor=True)

    # Calculate the semantic similarity matrix
    similarity_matrix = util.pytorch_cos_sim(embeddings_1, embeddings_2).cpu().numpy()

    # Save the semantic similarity matrix
    np.save(output_file_location, similarity_matrix)
    print(f"Saved the semantic similarity matrix to {output_file_location}")

    return similarity_matrix


