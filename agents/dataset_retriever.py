"""
Dataset Retriever Agent

Uses the Kaggle API to search for datasets matching the keywords
extracted by the IntentAnalyzer. Delegates to the search_kaggle_datasets skill.
"""

from skills.search_kaggle_datasets import search_kaggle_multi
from skills.search_huggingface_datasets import search_huggingface_multi


class DatasetRetriever:
    """Retrieves candidate datasets from Kaggle and Hugging Face using multi-keyword search."""

    def __init__(self):
        """Initialize the DatasetRetriever."""
        pass

    def retrieve(self, intent: dict, max_per_keyword: int = 10) -> list[dict]:
        """
        Search Kaggle and Hugging Face for datasets matching the extracted intent.

        Runs searches per keyword across both platforms, tagging their sources
        and merging the results.

        Args:
            intent: Structured intent dict from IntentAnalyzer, containing
                    'keywords', 'domain', and 'goal_summary'.
            max_per_keyword: Maximum results to fetch per keyword search.

        Returns:
            A merged list of dataset metadata dicts.
        """
        keywords = intent.get("keywords", [])
        if not keywords:
            print("[DatasetRetriever] Warning: no keywords to search.")
            return []

        # 1. Retrieve from Kaggle
        print(f"[DatasetRetriever] Searching Kaggle with {len(keywords)} keyword(s)...")
        try:
            kaggle_datasets = search_kaggle_multi(keywords, max_per_keyword=max_per_keyword)
            for ds in kaggle_datasets:
                ds["source"] = "kaggle"
            # Limit candidate list to stay under Gemini embedding rate limits
            kaggle_datasets = kaggle_datasets[:35]
            print(f"[DatasetRetriever] Kept top {len(kaggle_datasets)} unique datasets on Kaggle.")
        except Exception as e:
            print(f"[DatasetRetriever] Error: Kaggle search failed: {e}")
            kaggle_datasets = []

        # 2. Retrieve from Hugging Face
        print(f"[DatasetRetriever] Searching Hugging Face with {len(keywords)} keyword(s)...")
        try:
            hf_datasets = search_huggingface_multi(keywords, max_per_keyword=max_per_keyword)
            # source is already set to 'huggingface' inside search_huggingface_datasets
            # Limit candidate list to stay under Gemini embedding rate limits
            hf_datasets = hf_datasets[:35]
            print(f"[DatasetRetriever] Kept top {len(hf_datasets)} unique datasets on Hugging Face.")
        except Exception as e:
            print(f"[DatasetRetriever] Warning: Hugging Face search failed/unavailable: {e}")
            hf_datasets = []

        # 3. Merge results
        combined = kaggle_datasets + hf_datasets
        print(f"[DatasetRetriever] Merged total of {len(combined)} unique candidate datasets for evaluation.")
        
        return combined
