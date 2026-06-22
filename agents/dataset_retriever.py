"""
Dataset Retriever Agent

Uses the Kaggle API to search for datasets matching the keywords
extracted by the IntentAnalyzer. Delegates to the search_kaggle_datasets skill.
"""

from skills.search_kaggle_datasets import search_kaggle_multi


class DatasetRetriever:
    """Retrieves candidate datasets from Kaggle using multi-keyword search."""

    def __init__(self):
        """Initialize the DatasetRetriever."""
        pass

    def retrieve(self, intent: dict, max_per_keyword: int = 10) -> list[dict]:
        """
        Search Kaggle for datasets matching the extracted intent.

        Runs one Kaggle API search per keyword and returns deduplicated results.

        Args:
            intent: Structured intent dict from IntentAnalyzer, containing
                    'keywords', 'domain', and 'goal_summary'.
            max_per_keyword: Maximum results to fetch per keyword search.

        Returns:
            A deduplicated list of dataset metadata dicts.
        """
        keywords = intent.get("keywords", [])
        if not keywords:
            print("[DatasetRetriever] Warning: no keywords to search.")
            return []

        print(f"[DatasetRetriever] Searching Kaggle with {len(keywords)} keyword(s): {keywords}")
        datasets = search_kaggle_multi(keywords, max_per_keyword=max_per_keyword)
        print(f"[DatasetRetriever] Found {len(datasets)} unique candidate datasets.")
        return datasets
