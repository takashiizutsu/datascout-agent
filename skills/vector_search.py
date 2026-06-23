"""
Skill: Vector Search (Semantic Similarity)

Generates embeddings for user queries and dataset metadata using the Gemini API.
Computes cosine similarity between the query embedding and the dataset embeddings.
Provides fallback mechanisms if the API key is missing or model endpoints fail.
"""

import os
import numpy as np
import google.generativeai as genai

# Models to attempt in order of preference
EMBEDDING_MODELS = [
    "models/text-embedding-004",      # Standard Vertex/Gemini
    "models/gemini-embedding-2",       # Current Gemini 2 Developer API
    "models/gemini-embedding-001"      # Legacy Gemini 1
]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Compute cosine similarity between two numeric vectors.
    """
    arr1 = np.array(v1, dtype=float)
    arr2 = np.array(v2, dtype=float)
    
    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def generate_embeddings(texts: list[str]) -> list[list[float]] | None:
    """
    Generate embeddings for a list of text strings using the Gemini API.
    Attempts models in EMBEDDING_MODELS list and handles failure gracefully.
    
    Returns:
        A list of embedding vectors (list of lists of floats), or None if generation failed.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[VectorSearch] Warning: GOOGLE_API_KEY not found in environment.")
        return None
        
    genai.configure(api_key=api_key)
    
    # Try models in order
    for model_name in EMBEDDING_MODELS:
        try:
            # Batch embedding call. 'content' accepts a list of strings.
            # task_type='retrieval_document' is recommended for corpus documents.
            result = genai.embed_content(
                model=model_name,
                content=texts,
                task_type="retrieval_document"
            )
            # Inspect the return format
            embeddings = result.get('embedding')
            if embeddings and isinstance(embeddings, list):
                # Ensure it's a list of vectors. For single item input,
                # the API might return a single vector or a list depending on input list type.
                # Since we pass a list of texts, it should return a list of vectors.
                # If it returned a single vector (e.g. if list was simplified by SDK), we wrap it.
                if len(texts) == 1 and not isinstance(embeddings[0], list):
                    return [embeddings]
                return embeddings
        except Exception as e:
            print(f"[VectorSearch] Info: Model '{model_name}' failed: {e}")
            
    print("[VectorSearch] Error: All Gemini embedding models failed to generate embeddings.")
    return None


def generate_query_embedding(query: str) -> list[float] | None:
    """
    Generate embedding for the search query.
    Uses task_type='retrieval_query' for query-side embeddings.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
        
    genai.configure(api_key=api_key)
    
    for model_name in EMBEDDING_MODELS:
        try:
            result = genai.embed_content(
                model=model_name,
                content=query,
                task_type="retrieval_query"
            )
            embedding = result.get('embedding')
            if embedding:
                return embedding
        except Exception as e:
            pass
            
    return None


def construct_searchable_text(ds: dict) -> str:
    """
    Convert a dataset metadata dictionary into a single concatenated text string
    describing the dataset for vector similarity matching.
    """
    parts = []
    
    # Always include Title
    title = ds.get("title", "")
    if title:
        parts.append(f"Title: {title}")
        
    # Include Subtitle if available
    subtitle = ds.get("subtitle", "")
    if subtitle:
        parts.append(f"Subtitle: {subtitle}")
        
    # Include description if available
    desc = ds.get("description", "")
    if desc:
        parts.append(f"Description: {desc}")
        
    # Include tags/keywords if available
    tags = ds.get("tags", [])
    if tags:
        # Check if list of strings or dicts
        tag_str_list = []
        for tag in tags:
            if isinstance(tag, str):
                tag_str_list.append(tag)
            elif isinstance(tag, dict):
                tag_str_list.append(tag.get("name", ""))
        tags_clean = [t for t in tag_str_list if t]
        if tags_clean:
            parts.append(f"Keywords: {', '.join(tags_clean)}")
            
    return ". ".join(parts)


def compute_vector_similarities(query: str, datasets: list[dict]) -> list[float] | None:
    """
    Computes cosine similarity between the query and all candidate datasets.
    
    Args:
        query: The user's natural language goal.
        datasets: List of dataset metadata dicts.
        
    Returns:
        A list of float similarity scores corresponding to each dataset in the input list,
        or None if embedding generation failed.
    """
    if not datasets:
        return []
        
    # 1. Generate query embedding
    query_emb = generate_query_embedding(query)
    if not query_emb:
        print("[VectorSearch] Warning: Failed to generate query embedding.")
        return None
        
    # 2. Build searchable texts for all datasets
    searchable_texts = [construct_searchable_text(ds) for ds in datasets]
    
    # 3. Batch generate dataset embeddings
    dataset_embs = generate_embeddings(searchable_texts)
    if not dataset_embs or len(dataset_embs) != len(datasets):
        print("[VectorSearch] Warning: Failed to generate dataset embeddings.")
        return None
        
    # 4. Compute cosine similarity for each dataset
    similarities = []
    for emb in dataset_embs:
        sim = cosine_similarity(query_emb, emb)
        similarities.append(sim)
        
    return similarities
