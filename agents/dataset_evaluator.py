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
from agents.evaluation_agent import EvaluationAgent


class DatasetEvaluator:
    """Evaluates and ranks Kaggle and Hugging Face datasets using rule-based scoring and Gemini."""

    def __init__(self):
        """Initialize the DatasetEvaluator and its sub-agents."""
        self.evaluation_agent = EvaluationAgent()

    def evaluate(self, datasets: list[dict], intent: dict, top_k: int = 10) -> tuple[list[dict], str, dict]:
        """
        Score, rank, independently evaluate, and generate recommendations.

        Args:
            datasets: List of dataset metadata dicts from the retriever.
            intent: The structured intent dict from IntentAnalyzer.
            top_k: Number of top datasets to return (default 10).

        Returns:
            A tuple of:
                - top_datasets (list[dict]): The top-k evaluated dataset dicts,
                  sorted by evaluation_score descending.
                - report (str): A formatted recommendation report string.
                - quality_assessment (dict): Overall search quality evaluation details.
        """
        keywords = intent.get("keywords", [])
        query = intent.get("goal_summary", "")

        # --- Step 1: Precompute vector similarities using Gemini embeddings ---
        similarities = None
        if query and datasets:
            similarities = compute_vector_similarities(query, datasets)

        # --- Step 2: Score every candidate dataset with rule-based formula ---
        scored = []
        for idx, ds in enumerate(datasets):
            sim = similarities[idx] if similarities is not None else None
            scored_ds = score_dataset(ds, keywords, vector_similarity=sim, intent=intent)
            scored.append(scored_ds)

        # --- Step 3: Sort by composite score and slice Top 20 candidates ---
        scored.sort(key=lambda d: d["composite_score"], reverse=True)
        top_20 = scored[:20]

        # --- Step 4: Run Gemini-based independent candidate evaluation ---
        print(f"[DatasetEvaluator] Evaluating Top {len(top_20)} candidates using Gemini...")
        evaluated_candidates = self.evaluation_agent.evaluate_candidates(top_20, intent)

        # --- Step 5: Run Gemini-based overall search quality evaluation on Top 20 ---
        print("[DatasetEvaluator] Evaluating overall search quality using Gemini...")
        quality_assessment = self.evaluation_agent.evaluate_search_quality(evaluated_candidates, intent)

        # --- Step 6: Take final top_k recommendations ---
        top = evaluated_candidates[:top_k]

        # --- Step 7: Generate the recommendation report ---
        report = generate_recommendation(top, intent, quality_assessment=quality_assessment)

        return top, report, quality_assessment
