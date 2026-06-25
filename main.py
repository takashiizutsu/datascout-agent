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

    # ── Step 2: Dataset Retrieval ────────────────────────────────
    print("[Step 2] Searching Kaggle for datasets...")
    retriever = DatasetRetriever()
    candidates = retriever.retrieve(intent)
    if not candidates:
        print("  No datasets found on Kaggle for those keywords.")
        return
    print()

    # ── Step 3: Evaluation & Ranking ─────────────────────────────
    print("[Step 3] Evaluating and ranking datasets using Gemini Evaluation Agent...")
    evaluator = DatasetEvaluator()
    top_datasets, report, quality_assessment = evaluator.evaluate(candidates, intent, top_k=10)

    # ── Step 4: Display Recommendations ──────────────────────────
    print(report)


if __name__ == "__main__":
    main()
