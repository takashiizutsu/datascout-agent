"""
Skill: Search Kaggle Datasets

Wraps the Kaggle API to search for datasets by query string and return
structured metadata for each result.

Authentication:
    The Kaggle API client reads credentials from environment variables:
        - KAGGLE_USERNAME  → mapped to config key "username"
        - KAGGLE_KEY       → mapped to config key "key"
    These must be loaded into the environment (via python-dotenv) before
    this module is called. The Kaggle client picks them up automatically.
"""

from kaggle.api.kaggle_api_extended import KaggleApi


# --- Module-level API client (created once, reused across calls) ---
# This avoids re-authenticating on every search call when we run
# multiple keyword searches in sequence.
_api = None


def _get_api() -> KaggleApi:
    """Return an authenticated KaggleApi instance, creating it on first use."""
    global _api
    if _api is None:
        _api = KaggleApi()
        _api.authenticate()
    return _api


def search_kaggle_datasets(query: str, max_results: int = 20) -> list[dict]:
    """
    Search Kaggle for datasets matching a single query string.

    Args:
        query: A search query string (keywords).
        max_results: Maximum number of datasets to return per query.

    Returns:
        A list of dicts, each containing Kaggle dataset metadata.
    """
    api = _get_api()

    # --- Search Kaggle ---
    # dataset_list() returns a list of ApiDataset objects from the kagglesdk.
    # Results are paginated (~20 per page); we take the first page and slice.
    results = api.dataset_list(search=query)

    # --- Convert ApiDataset objects to plain dicts ---
    # ApiDataset attributes are snake_case (the Kaggle CLI bridges this
    # from its internal camelCase field registry via camel_to_snake).
    datasets = []
    for dataset in results[:max_results]:
        datasets.append({
            "ref": dataset.ref,
            "title": dataset.title,
            "total_bytes": dataset.total_bytes,
            "last_updated": str(dataset.last_updated),
            "download_count": dataset.download_count,
            "vote_count": dataset.vote_count,
            "usability_rating": dataset.usability_rating,
        })

    return datasets


def search_kaggle_multi(keywords: list[str], max_per_keyword: int = 10) -> list[dict]:
    """
    Search Kaggle with multiple keywords and return deduplicated results.

    This is the main entry point used by the DatasetRetriever agent.
    It runs one Kaggle API search per keyword, then merges and deduplicates
    the results by dataset ref (e.g., "owner/dataset-name").

    Args:
        keywords: A list of search keyword strings.
        max_per_keyword: Max results to keep per keyword search.

    Returns:
        A deduplicated list of dataset metadata dicts.
    """
    seen_refs = set()
    all_datasets = []

    for keyword in keywords:
        results = search_kaggle_datasets(keyword, max_results=max_per_keyword)
        for ds in results:
            # --- Deduplicate by ref ---
            # The same dataset may appear in multiple keyword searches.
            if ds["ref"] not in seen_refs:
                seen_refs.add(ds["ref"])
                all_datasets.append(ds)

    return all_datasets
