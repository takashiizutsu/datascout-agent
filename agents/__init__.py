"""
DataScout Agent — Agents Package

This package contains the core agents that power the DataScout pipeline:
  - IntentAnalyzer: Parses user queries into structured intent.
  - DatasetRetriever: Retrieves relevant datasets via semantic search.
  - DatasetEvaluator: Scores and ranks candidate datasets.
  - EvaluationAgent: Evaluates candidates and search quality using Gemini.
"""

from agents.intent_analyzer import IntentAnalyzer
from agents.dataset_retriever import DatasetRetriever
from agents.dataset_evaluator import DatasetEvaluator
from agents.evaluation_agent import EvaluationAgent

