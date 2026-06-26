"""
DataScout Agent — Main Entry Point (MVP)

A Kaggle API-powered AI agent that helps analysts discover and evaluate
public datasets based on a natural-language analysis goal.

End-to-end pipeline:
  1. User enters an analysis goal in plain English.
  2. IntentAnalyzer extracts search keywords (rule-based).
  3. DatasetRetriever searches Kaggle API with those keywords.
  4. DatasetEvaluator scores, ranks, and generates a top-5 recommendation.
  5. The recommendation report is printed to the console.
"""

import os
import sys

# --- Fix Windows console encoding ---
# The default Windows console encoding (cp1252) cannot display some Unicode
# characters that appear in Kaggle dataset titles. Force UTF-8 output.
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# --- Load environment variables from .env ---
# Must happen before any Kaggle imports, because the Kaggle API client
# reads KAGGLE_USERNAME and KAGGLE_KEY from the environment at init time.
load_dotenv()

from agents.intent_analyzer import IntentAnalyzer
from agents.dataset_retriever import DatasetRetriever
from agents.dataset_evaluator import DatasetEvaluator
from agents.query_planner_agent import QueryPlannerAgent


def main():
    """Run the full DataScout Agent pipeline."""
    print("=" * 64)
    print("  DataScout Agent — Kaggle Dataset Discovery (MVP)")
    print("=" * 64)
    print()

    # --- Verify Kaggle credentials ---
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        print("ERROR: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env")
        print("Copy .env.example to .env and fill in your Kaggle credentials.")
        return

    # --- Get the user's analysis goal ---
    goal = input("Describe your analysis goal: > ")
    if not goal.strip():
        print("No goal entered. Exiting.")
        return

    print()

    # ── Step 1: Intent Analysis ──────────────────────────────────
    print("[Step 1] Analyzing your goal...")
    analyzer = IntentAnalyzer()
    intent = analyzer.analyze(goal)
    print(f"  Keywords: {intent['keywords']}")
    print(f"  Domain:   {intent['domain'] or 'General'}")
    print()

    # ── Step 1.5: Query Planning ────────────────────────────────
    print("[Step 1.5] Generating Search Plan using Gemini...")
    planner = QueryPlannerAgent()
    search_plan = planner.generate_plan(goal)
    if search_plan:
        print(f"  Primary Topic: {search_plan.get('primary_topic', 'N/A')}")
        print(f"  Domain:        {search_plan.get('domain', 'N/A')}")
        print(f"  Queries:       {', '.join(search_plan.get('search_queries', []))}")
    else:
        print("  Failed to generate search plan. Falling back to keyword search.")
    print()

    # ── Step 2: Dataset Retrieval ────────────────────────────────
    print("[Step 2] Searching Kaggle and Hugging Face for datasets...")
    retriever = DatasetRetriever()
    candidates = retriever.retrieve(intent, search_plan=search_plan)
    if not candidates:
        print("  No datasets found on Kaggle/Hugging Face.")
        return
    print()

    # ── Step 3: Evaluation & Ranking ─────────────────────────────
    print("[Step 3] Evaluating and ranking datasets using Gemini Evaluation Agent...")
    evaluator = DatasetEvaluator()
    top_datasets, report, quality_assessment = evaluator.evaluate(candidates, intent, top_k=10, search_plan=search_plan)

    # --- Print Hugging Face Pipeline Diagnosis ---
    print("\n" + "="*54)
    print(" TEMPORARY DEBUG: Hugging Face Pipeline Diagnosis")
    print("="*54)
    print(f"{'Stage':<24} {'Kaggle':<8} {'Hugging Face':<14} {'Total':<5}")
    print(f"------------------------------------------------------")
    print(f"{'Retrieved':<24} {retriever.kaggle_retrieved:<8} {retriever.hf_retrieved:<14} {retriever.kaggle_retrieved + retriever.hf_retrieved:<5}")
    print(f"{'Merged':<24} {retriever.kaggle_retrieved:<8} {retriever.hf_retrieved:<14} {retriever.kaggle_retrieved + retriever.hf_retrieved:<5}")
    print(f"{'Deduplicated':<24} {retriever.kaggle_deduplicated:<8} {retriever.hf_deduplicated:<14} {retriever.total_deduplicated:<5}")
    print(f"{'Embedding Ranked':<24} {retriever.kaggle_ranked:<8} {retriever.hf_ranked:<14} {retriever.total_ranked:<5}")
    print(f"{'Gemini Evaluated':<24} {evaluator.kaggle_eval:<8} {evaluator.hf_eval:<14} {evaluator.total_eval:<5}")
    print(f"{'Final Top 10':<24} {evaluator.kaggle_top10:<8} {evaluator.hf_top10:<14} {evaluator.total_top10:<5}")
    print("="*54 + "\n")

    # ── Step 4: Display Recommendations ──────────────────────────
    print(report)


if __name__ == "__main__":
    main()
