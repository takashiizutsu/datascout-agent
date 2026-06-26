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
        self.kaggle_retrieved = 0
        self.hf_retrieved = 0
        self.kaggle_deduplicated = 0
        self.hf_deduplicated = 0
        self.total_deduplicated = 0
        self.kaggle_ranked = 0
        self.hf_ranked = 0
        self.total_ranked = 0

    def retrieve(self, intent: dict, search_plan: dict = None, max_per_keyword: int = 10) -> list[dict]:
        """
        Search Kaggle and Hugging Face for datasets matching the intent or search plan.

        Runs searches per query/keyword across both platforms, tagging their sources,
        deduplicating, and merging the results.

        Args:
            intent: Structured intent dict from IntentAnalyzer.
            search_plan: Optional search plan from QueryPlannerAgent.
            max_per_keyword: Maximum results to fetch per query search.

        Returns:
            A merged and capped list of dataset metadata dicts.
        """
        if search_plan and search_plan.get("search_queries"):
            queries = search_plan["search_queries"]
            print(f"[DatasetRetriever] Using {len(queries)} search queries from the Gemini Search Plan.")
        else:
            queries = intent.get("keywords", [])
            print(f"[DatasetRetriever] Using {len(queries)} keywords from the Intent Analyzer (Fallback).")

        if not queries:
            print("[DatasetRetriever] Warning: no queries/keywords to search.")
            return []

        # 1. Retrieve from Kaggle
        print(f"[DatasetRetriever] Searching Kaggle with {len(queries)} query/queries...")
        try:
            kaggle_datasets = search_kaggle_multi(queries, max_per_keyword=max_per_keyword)
            for ds in kaggle_datasets:
                ds["source"] = "kaggle"
            print(f"[DatasetRetriever] Found {len(kaggle_datasets)} unique datasets on Kaggle.")
        except Exception as e:
            print(f"[DatasetRetriever] Error: Kaggle search failed: {e}")
            kaggle_datasets = []

        # 2. Retrieve from Hugging Face
        print(f"[DatasetRetriever] Searching Hugging Face with {len(queries)} query/queries...")
        try:
            hf_datasets = search_huggingface_multi(queries, max_per_keyword=max_per_keyword)
            # source is already set to 'huggingface' inside search_huggingface_datasets
            print(f"[DatasetRetriever] Found {len(hf_datasets)} unique datasets on Hugging Face.")
        except Exception as e:
            print(f"[DatasetRetriever] Warning: Hugging Face search failed/unavailable: {e}")
            hf_datasets = []

        # 3. Merge, Deduplicate, and Cap to 100
        self.kaggle_retrieved = len(kaggle_datasets)
        self.hf_retrieved = len(hf_datasets)

        seen_keys = set()
        deduplicated = []
        for ds in kaggle_datasets + hf_datasets:
            key = (ds.get("source"), ds.get("ref"))
            if key not in seen_keys:
                seen_keys.add(key)
                deduplicated.append(ds)

        self.kaggle_deduplicated = sum(1 for d in deduplicated if d.get("source") == "kaggle")
        self.hf_deduplicated = sum(1 for d in deduplicated if d.get("source") == "huggingface")
        self.total_deduplicated = len(deduplicated)

        # Cap the combined candidate pool to 100 total candidates
        combined = deduplicated[:100]
        self.kaggle_ranked = sum(1 for d in combined if d.get("source") == "kaggle")
        self.hf_ranked = sum(1 for d in combined if d.get("source") == "huggingface")
        self.total_ranked = len(combined)

        print(f"[DatasetRetriever] Merged and capped combined candidate pool to {len(combined)} unique datasets for evaluation.")

        return combined
