"""
Skill: Search Hugging Face Datasets

Wraps the huggingface_hub API to search for public datasets by query string
and return structured metadata mapped to our unified schema format.

Authentication:
    Reads HF_TOKEN from environment variables if present.
    If missing, runs public-only search gracefully.
"""

import os
from datetime import datetime
from huggingface_hub import HfApi

# Module-level API client (created once, reused across calls)
_api = None


def _get_api() -> HfApi:
    """Return an HfApi instance, creating it on first use."""
    global _api
    if _api is None:
        hf_token = os.getenv("HF_TOKEN")
        # Initialize client. Token is optional for public dataset search.
        _api = HfApi(token=hf_token)
    return _api


def search_huggingface_datasets(query: str, max_results: int = 20) -> list[dict]:
    """
    Search Hugging Face Hub for datasets matching a single query string.

    Args:
        query: A search query string (keywords).
        max_results: Maximum number of datasets to return.

    Returns:
        A list of dicts mapped to the unified schema.
    """
    try:
        api = _get_api()
        # list_datasets returns a generator of DatasetInfo objects
        results = api.list_datasets(search=query, limit=max_results)
        results_list = list(results)
    except Exception as e:
        print(f"[HuggingFaceSearch] Warning: Failed to query Hugging Face API: {e}")
        return []

    datasets = []
    for dataset in results_list:
        # 1. Clean and parse a readable Title from the Dataset ID
        # e.g., 'owner/diabetes-predict-db' -> 'Diabetes Predict Db'
        dataset_id = dataset.id
        name_part = dataset_id.split('/')[-1]
        title = name_part.replace('-', ' ').replace('_', ' ').title()

        # 2. Extract description (from dataset card or readme description)
        desc = getattr(dataset, "description", "") or ""
        
        # 3. Clean tags list
        tags = getattr(dataset, "tags", []) or []
        tags_list = [t for t in tags if isinstance(t, str)]

        # 4. Safely extract last updated date
        last_mod = getattr(dataset, "last_modified", None) or getattr(dataset, "created_at", None)
        last_updated_str = str(last_mod) if last_mod else "N/A"

        # 5. Extract downloads and likes
        downloads = getattr(dataset, "downloads", 0) or 0
        likes = getattr(dataset, "likes", 0) or 0

        datasets.append({
            "ref": dataset_id,
            "title": title,
            "subtitle": "",  # Hugging Face doesn't have a distinct subtitle attribute
            "description": desc,
            "tags": tags_list,
            "total_bytes": 0,  # Hugging Face doesn't have a simple total_bytes field in search
            "last_updated": last_updated_str,
            "download_count": downloads,
            "vote_count": likes,          # Map likes to Kaggle's vote_count equivalent
            "usability_rating": 0.70,     # Default rating so HF datasets aren't penalized
            "source": "huggingface",      # Explicit source identifier
        })

    return datasets


def search_huggingface_multi(keywords: list[str], max_per_keyword: int = 10) -> list[dict]:
    """
    Search Hugging Face with multiple keywords and return deduplicated results.

    Args:
        keywords: A list of search keyword strings.
        max_per_keyword: Max results to keep per keyword search.

    Returns:
        A deduplicated list of dataset metadata dicts.
    """
    seen_refs = set()
    all_datasets = []

    for keyword in keywords:
        results = search_huggingface_datasets(keyword, max_results=max_per_keyword)
        for ds in results:
            # Deduplicate by ref (ID)
            if ds["ref"] not in seen_refs:
                seen_refs.add(ds["ref"])
                all_datasets.append(ds)

    return all_datasets
