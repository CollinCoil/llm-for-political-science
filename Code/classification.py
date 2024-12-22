import os
import re
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from collections import defaultdict

from pyod.models.copod import COPOD
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.inne import INNE


# create_document_embeddings function remains the same as in your code
def create_document_embeddings(data_dirs, metadata_file, model_name, model_path):
    """Create embeddings for all documents using specified model"""
    print(f"\nProcessing with model: {model_name}")
    
    # Load metadata
    metadata = pd.read_csv(metadata_file)
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    
    # Get model-specific max length
    max_length = tokenizer.model_max_length
    if model_name in ["ModernBERT-base", "ModernBERT-large"]:
        max_length = 8196
    elif model_name in ["bart-base", "bart-large"]:
        max_length = 1024
    elif model_name in ["gte-small", "gte-base", "gte-large"]:
        max_length = 512
    print(f"Max length for {model_name}: {max_length}")
    
    embedding_list = []
    labels_list = []
    
    def embed_document(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Original embedding logic for other models
        tokens = tokenizer(text, return_tensors='pt', truncation=False, padding=False)
        input_ids = tokens["input_ids"][0]
        
        embeddings = []
        token_counts = []
        
        for i in range(0, input_ids.size(0), max_length):
            chunk_input_ids = input_ids[i:i + max_length]
            chunk_attention_mask = torch.ones_like(chunk_input_ids)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=chunk_input_ids.unsqueeze(0),
                    attention_mask=chunk_attention_mask.unsqueeze(0)
                )
                chunk_embedding = outputs.last_hidden_state.mean(dim=1)
            
            embeddings.append(chunk_embedding)
            token_counts.append(chunk_input_ids.size(0))
        
        embeddings = torch.cat(embeddings, dim=0)
        token_counts = torch.tensor(token_counts, dtype=torch.float32)
        weighted_avg_embedding = torch.sum(embeddings * token_counts[:, None], dim=0) / torch.sum(token_counts)
        
        return weighted_avg_embedding
    
    # Process all documents
    for directory in data_dirs:
        dir_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
        for file_name in tqdm(dir_files, desc=f"Processing {directory}"):
            filepath = os.path.join(directory, file_name)
            embedding = embed_document(filepath)
            embedding_list.append(embedding)
            
            meta_row = metadata[metadata["document"] == file_name]
            if meta_row.empty:
                print(f"Warning: No metadata match for {file_name}")
                labels_list.append({"document": file_name, "directory": directory})
            else:
                labels_list.append(meta_row.iloc[0].to_dict())
    
    # Convert to final format
    embedding_matrix = torch.stack(embedding_list)
    
    labels_df = pd.DataFrame(labels_list)
    
    return embedding_matrix.numpy(), labels_df

def initialize_models():
    """Initialize all classification and outlier detection models"""
    pyod_models = {
        "COPOD": COPOD(contamination=0.05),
        "LOF": LOF(n_neighbors=20, contamination=0.05),
        "OCSVM": OCSVM(kernel="rbf", gamma="auto", contamination=0.05),
        "IForest": IForest(contamination=0.05, max_samples=150, random_state=42),
        "DIF": DIF(contamination=0.05, max_samples=150),
        "INNE": INNE(contamination=0.05),
        "AutoEncoder": AutoEncoder(
            hidden_neuron_list=[250, 50, 25, 5],
            epoch_num=20,
            contamination=0.05,
            batch_size=64,
            dropout_rate=0,
            preprocessing=False,
            verbose=0
        )
    }
    
    sklearn_models = {
        "Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_samples=150, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(250, 50, 25, 5),
            max_iter=1000,
            random_state=42,
            batch_size=64
        )
    }
    
    return pyod_models, sklearn_models

def normalize_scores(scores):
    """
    Normalize and flip scores so that:
    1. Higher scores correspond to opinions (positive class)
    2. Scores are mapped to [0,1] range
    """
    # Flip scores so higher values indicate opinions
    scores = -scores
    
    # Min-max normalization to [0,1]
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.zeros_like(scores)
    normalized_scores = (scores - min_score) / (max_score - min_score)
    
    return normalized_scores

def evaluate_models(X, y, model_name):
    """Evaluate all models using 6-fold cross validation"""
    pyod_models, sklearn_models = initialize_models()
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    
    # Initialize results dictionary
    results = {
        'embedding_model': [],
        'model_type': [],
        'model_name': [],
    }
    # Add columns for each fold
    for i in range(6):
        results[f'roc_auc_{i+1}'] = []
    
    # Store results for calculating statistics later
    model_scores = defaultdict(list)
    
    # Process each model first, then each fold
    # Outlier Detection Models
    for name, model in pyod_models.items():
        fold_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"\nProcessing {name} fold {fold_idx + 1}/6")
            
            model.fit(X_train)
            # Get decision scores and normalize them
            y_scores = model.decision_function(X_test)
            y_scores_normalized = normalize_scores(y_scores)
            roc_auc = roc_auc_score(y_test, y_scores_normalized)
            fold_scores.append(roc_auc)
        
        # Store results for this model
        results['embedding_model'].append(model_name)
        results['model_type'].append('outlier')
        results['model_name'].append(name)
        for i, score in enumerate(fold_scores):
            results[f'roc_auc_{i+1}'].append(score)
        
    # Classification Models
    for name, model in sklearn_models.items():
        fold_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"\nProcessing {name} fold {fold_idx + 1}/6")
            
            model.fit(X_train, y_train)
            # Get probability scores for positive class
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                y_scores = model.decision_function(X_test)
                # Normalize decision scores to [0,1] range
                y_scores = normalize_scores(y_scores)
            
            roc_auc = roc_auc_score(y_test, y_scores)
            fold_scores.append(roc_auc)
        
        # Store results for this model
        results['embedding_model'].append(model_name)
        results['model_type'].append('classifier')
        results['model_name'].append(name)
        for i, score in enumerate(fold_scores):
            results[f'roc_auc_{i+1}'].append(score)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate mean and std ROC AUC
    fold_columns = [f'roc_auc_{i+1}' for i in range(6)]
    results_df['mean_roc_auc'] = results_df[fold_columns].mean(axis=1)
    results_df['std_roc_auc'] = results_df[fold_columns].std(axis=1)
    
    return results_df
