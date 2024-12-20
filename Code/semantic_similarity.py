import json
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def load_sentences_from_jsonl_by_document(file_path, document_ids):
    """
    Load and filter sentences from a JSONL file based on specified document IDs.
    Each line should contain a dictionary with "document" and "text" fields.
    """
    sentences_by_doc = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc_id = data.get("document")
            if doc_id in document_ids:
                if doc_id not in sentences_by_doc:
                    sentences_by_doc[doc_id] = []
                sentences_by_doc[doc_id].append(data["text"])
    return sentences_by_doc

def load_metadata_cases(metadata_file):
    """
    Load metadata from CSV and organize documents by case.
    """
    metadata = pd.read_csv(metadata_file)
    cases = metadata.groupby("case")["document"].apply(list).to_dict()
    return cases

def calculate_semantic_similarity(opinion_jsonl_file, amici_jsonl_file, metadata_file, output_dir="Data/similarity_matrices", threshold=0.9, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Calculate the percentage of opinion text that is similar to amici briefs for each case.
    """
    # Load cases from metadata
    cases = load_metadata_cases(metadata_file)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = SentenceTransformer(model_name)
    
    results = {}
    for case, documents in cases.items():
        # Load sentences for current case from opinion and amici JSONL files
        opinion_sentences_by_doc = load_sentences_from_jsonl_by_document(opinion_jsonl_file, set(documents))
        amici_sentences_by_doc = load_sentences_from_jsonl_by_document(amici_jsonl_file, set(documents))
        
        # Gather all sentences from each category
        opinion_sentences = [sentence for doc_id, texts in opinion_sentences_by_doc.items() for sentence in texts]
        amici_sentences = [sentence for doc_id, texts in amici_sentences_by_doc.items() for sentence in texts]
        
        # Create embeddings
        opinion_embeddings = model.encode(opinion_sentences, convert_to_tensor=True)
        amici_embeddings = model.encode(amici_sentences, convert_to_tensor=True)
        
        # Calculate similarity matrix
        similarity_matrix = util.pytorch_cos_sim(opinion_embeddings, amici_embeddings).cpu().numpy()
        
        # Save the similarity matrix for the current case
        matrix_file = os.path.join(output_dir, f"{case}_similarity_matrix.npy")
        np.save(matrix_file, similarity_matrix)
        print(f"Saved similarity matrix for case '{case}' to {matrix_file}")
        
        # Determine maximum similarity for each opinion sentence and track the index of the most similar amicus sentence
        max_similarity_scores = similarity_matrix.max(axis=1)
        max_similarity_indices = similarity_matrix.argmax(axis=1)
        
        # Find sentences with similarity above threshold
        highly_similar_opinion_indices = np.where(max_similarity_scores >= threshold)[0]
        highly_similar_pairs = [
            {
                "opinion_sentence": opinion_sentences[i],
                "amicus_sentence": amici_sentences[max_similarity_indices[i]],
                "similarity_score": max_similarity_scores[i]
            }
            for i in highly_similar_opinion_indices
        ]
        
        # Calculate percent copied
        copied_text = " ".join(pair["opinion_sentence"] for pair in highly_similar_pairs)
        total_words = sum(len(sentence.split()) for sentence in opinion_sentences)
        copied_words = sum(len(sentence.split()) for sentence in copied_text.split())
        percent_copied = (copied_words / total_words) * 100 if total_words > 0 else 0
        
        # Store results for the case
        results[case] = {
            "total_words": total_words, 
            "percent_copied": percent_copied,
            "highly_similar_pairs": highly_similar_pairs,
            "similarity_matrix_file": matrix_file
        }
        print(f"Processed case {case}: {percent_copied:.2f}% of opinion text is similar to amici briefs")
    
    return results
