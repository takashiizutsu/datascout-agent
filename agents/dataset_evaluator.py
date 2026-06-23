"""
Dataset Evaluator Agent

Orchestrates the scoring and ranking of candidate datasets returned
from Kaggle API searches. Uses the score_dataset and generate_recommendation
skills to produce the final top-k recommendations.

Pipeline:
    1. Score each candidate dataset (relevance, popularity, freshness, usability).
    2. Sort by composite score descending.
    3. Take the top-k results.
    4. Generate a formatted recommendation report.
"""

from skills.score_dataset import score_dataset
from skills.generate_recommendation import generate_recommendation
from skills.vector_search import compute_vector_similarities


class DatasetEvaluator:
    """Evaluates and ranks Kaggle datasets, then generates recommendations."""

    def __init__(self):
        """Initialize the DatasetEvaluator. No model needed for rule-based MVP."""
        pass

    def evaluate(self, datasets: list[dict], intent: dict, top_k: int = 5) -> tuple[list[dict], str]:
        """
        Score, rank, and generate recommendations for candidate datasets.

        Args:
            datasets: List of dataset metadata dicts from the retriever.
            intent: The structured intent dict from IntentAnalyzer.
            top_k: Number of top datasets to return.

        Returns:
            A tuple of:
                - top_datasets (list[dict]): The top-k scored dataset dicts,
                  sorted by composite_score descending.
                - report (str): A formatted recommendation report string.
        """
        keywords = intent.get("keywords", [])
        query = intent.get("goal_summary", "")

        # --- Step 1: Precompute vector similarities using Gemini embeddings ---
        # Fallback to None if generation fails or GOOGLE_API_KEY is missing.
        similarities = None
        if query and datasets:
            similarities = compute_vector_similarities(query, datasets)

        # --- Step 2: Score every candidate dataset ---
        scored = []
        for idx, ds in enumerate(datasets):
            sim = similarities[idx] if similarities is not None else None
            scored_ds = score_dataset(ds, keywords, vector_similarity=sim, intent=intent)
            scored.append(scored_ds)

        # --- Step 3: Sort the final list by composite score (highest first) ---
        scored.sort(key=lambda d: d["composite_score"], reverse=True)

        # --- Step 4: Take top-k ---
        top = scored[:top_k]

        # --- Step 5: Generate the recommendation report ---
        report = generate_recommendation(top, intent)

        return top, report
