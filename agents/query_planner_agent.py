"""
Query Planner Agent

Uses Gemini to analyze a user's natural-language goal and generate a structured Search Plan
containing short, targeted search queries for Kaggle and Hugging Face.
"""

import os
import json
import google.generativeai as genai

# Models to attempt in order of preference
GENERATIVE_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-flash-latest",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.0-flash-lite"
]


class QueryPlannerAgent:
    """Generates structured search plans from natural language goals using Gemini."""

    def __init__(self):
        """Initialize the QueryPlannerAgent and configure the Gemini API."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        else:
            print("[QueryPlannerAgent] Warning: GOOGLE_API_KEY not found in environment.")

    def generate_plan(self, goal: str) -> dict | None:
        """
        Generate a structured search plan from the user's natural-language goal.

        Args:
            goal: Natural-language research goal.

        Returns:
            A dict with:
                - primary_topic (str)
                - domain (str)
                - search_queries (list[str]): 5 to 8 concise queries
                - must_have_concepts (list[str])
                - optional_concepts (list[str])
                - excluded_terms (list[str])
            or None if generation failed.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[QueryPlannerAgent] Warning: No GOOGLE_API_KEY. Falling back to keyword search.")
            return None

        prompt = f"""You are an expert search planner.
User Goal: "{goal}"

Create a structured Search Plan to find relevant datasets on Kaggle and Hugging Face for this goal.
Your queries must be short, targeted, and concise (typically 2-4 words) to retrieve relevant datasets from standard keyword-based APIs.

Create 5 to 8 diverse and specific search queries. Do not use punctuation or complex operators. Focus on standard terms, report titles, synonyms, and variations of indicators.

Format the output strictly as a JSON object with the following keys:
- "primary_topic" (string, the core condition, entity, or domain)
- "domain" (string, the broad research area, e.g. "Finance", "Healthcare", "Education")
- "search_queries" (array of 5 to 8 strings, concise, targeted search queries)
- "must_have_concepts" (array of strings, key terms that are essential)
- "optional_concepts" (array of strings, useful but secondary terms)
- "excluded_terms" (array of strings, terms to exclude to reduce noise)
"""

        response_data = None
        for model_name in GENERATIVE_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                if response.text:
                    response_data = json.loads(response.text)
                    break
            except Exception as e:
                print(f"[QueryPlannerAgent] Info: Model '{model_name}' failed to generate search plan: {e}")

        if not response_data or "search_queries" not in response_data:
            print("[QueryPlannerAgent] Error: Failed to generate a valid search plan from Gemini.")
            return None

        return response_data
