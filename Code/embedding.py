import json
import numpy as np
from sentence_transformers import SentenceTransformer

def create_embeddings_from_jsonl(jsonl_file_path, model_name='sentence-transformers/all-mpnet-base-v1', 
                                 embeddings_output_path='sentence_embeddings.npy', 
                                 ids_output_path='ids.npy', documents_output_path='documents.npy'):
    """
    Create embeddings for sentences stored in a JSONL file and save them as a NumPy array.

    Parameters:
    - jsonl_file_path (str): Path to the input JSONL file.
    - model_name (str): Name of the SentenceTransformer model to use for creating embeddings.
    - embeddings_output_path (str): Path to save the output embeddings as a .npy file.
    - ids_output_path (str): Path to save the IDs as a .npy file.
    - documents_output_path (str): Path to save the documents as a .npy file.
    
    Returns:
    - None
    """
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)

    # Initialize lists to store IDs, documents, and texts
    ids = []
    documents = []
    texts = []

    # Load sentences from the JSONL file
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            ids.append(entry['ID'])
            documents.append(entry['document'])
            texts.append(entry['text'])

    # Create embeddings for all sentences
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Store embeddings and additional data as NumPy arrays
    np.save(embeddings_output_path, embeddings)
    np.save(ids_output_path, np.array(ids))
    np.save(documents_output_path, np.array(documents))

    print(f"Embeddings created and saved to {embeddings_output_path}.")
    print(f"IDs saved to {ids_output_path}.")
    print(f"Documents saved to {documents_output_path}.")