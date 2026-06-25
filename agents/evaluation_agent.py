"""
Evaluation Agent

Uses Gemini to perform independent evaluations of candidate datasets
and evaluate overall search quality.
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


class EvaluationAgent:
    """Evaluates datasets and overall search quality using Gemini."""

    def __init__(self):
        """Initialize the EvaluationAgent and configure the Gemini API."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        else:
            print("[EvaluationAgent] Warning: GOOGLE_API_KEY not found in environment.")

    def evaluate_candidates(self, scored_candidates: list[dict], intent: dict) -> list[dict]:
        """
        Evaluate each candidate dataset using Gemini.
        
        Args:
            scored_candidates: List of dataset dicts (already scored by rules).
            intent: The structured intent dict from IntentAnalyzer.
            
        Returns:
            A list of datasets with evaluation_score, confidence, verdict, and evaluation_reason added,
            sorted by evaluation_score descending.
        """
        if not scored_candidates:
            return []

        goal = intent.get("goal_summary", "")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[EvaluationAgent] Warning: No API key. Falling back to rule-based composite scores.")
            return self._apply_fallbacks(scored_candidates)

        # Prepare the list of candidates for the prompt
        candidates_data = []
        for ds in scored_candidates:
            desc = ds.get("description") or ""
            # Truncate description to save token space
            desc_truncated = desc[:200] + "..." if len(desc) > 200 else desc
            
            candidates_data.append({
                "ref": ds.get("ref"),
                "title": ds.get("title"),
                "description": desc_truncated,
                "tags": ds.get("tags") or [],
                "source": ds.get("source"),
                "semantic_similarity": ds.get("vector_similarity"),
                "composite_score": ds.get("composite_score")
            })

        prompt = f"""You are an expert data science evaluator.
User Goal: "{goal}"

Evaluate the following list of candidate datasets retrieved for this goal.
Your job is to independently judge the usefulness and relevance of each dataset to the user's goal.

CRITICAL INSTRUCTIONS:
1. Treat the fields 'semantic_similarity' and 'composite_score' as reference information ONLY. Do NOT simply reproduce or match the existing ranking from 'composite_score'.
2. Use the dataset title, description, tags, and source repository as your primary evidence.
3. Judge how well the dataset actually matches the user's specific topics (e.g. medical conditions, locations, specific indicators) and whether the utility is high.
4. If a dataset matches generic terms (like "factors", "risk", "prediction") but is about an entirely different topic than what the user requested, penalize it heavily.

Candidates:
{json.dumps(candidates_data, indent=2)}

Format the output strictly as a JSON object containing a single key "evaluations" which is an array of objects.
Each object in the array must strictly have the following keys:
- "ref" (string, matching the input dataset's ref)
- "evaluation_score" (float, between 0.0 and 1.0, representing your independent assessment of the quality and relevance of the dataset to the user goal)
- "confidence" (float, between 0.0 and 1.0, representing your confidence in this evaluation)
- "verdict" (string, one of: "Excellent", "Good", "Partial", "Irrelevant")
- "evaluation_reason" (string, a clear 1-2 sentence explanation of your decision, noting the primary strengths or reasons for penalties)
"""

        # Call Gemini Generative Model with JSON response config
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
                print(f"[EvaluationAgent] Info: Model '{model_name}' failed to evaluate candidates: {e}")

        if not response_data or "evaluations" not in response_data:
            print("[EvaluationAgent] Error: Failed to get valid evaluations from Gemini. Using fallbacks.")
            return self._apply_fallbacks(scored_candidates)

        # Map evaluations back to datasets
        eval_map = {item["ref"]: item for item in response_data["evaluations"] if "ref" in item}
        
        evaluated_list = []
        for ds in scored_candidates:
            ref = ds.get("ref")
            ev = eval_map.get(ref)
            
            ds_copy = dict(ds)
            if ev:
                ds_copy["evaluation_score"] = float(ev.get("evaluation_score", ds.get("composite_score", 0.0)))
                ds_copy["confidence"] = float(ev.get("confidence", 0.5))
                ds_copy["verdict"] = str(ev.get("verdict", "Good"))
                ds_copy["evaluation_reason"] = str(ev.get("evaluation_reason", "Evaluated by Gemini."))
            else:
                # Fallback for individual missing items
                ds_copy["evaluation_score"] = ds.get("composite_score", 0.0)
                ds_copy["confidence"] = 0.50
                ds_copy["verdict"] = "Good"
                ds_copy["evaluation_reason"] = "Fallback scoring based on baseline rules."
                
            evaluated_list.append(ds_copy)

        # Re-sort candidates by evaluation_score descending
        evaluated_list.sort(key=lambda x: x["evaluation_score"], reverse=True)
        return evaluated_list

    def evaluate_search_quality(self, evaluated_candidates: list[dict], intent: dict) -> dict:
        """
        Evaluate the overall retrieval quality of the Top 20 candidate pool as a whole.
        
        Args:
            evaluated_candidates: The list of evaluated datasets (Top 20).
            intent: The structured intent dict.
            
        Returns:
            A dict with quality_score, verdict, and reasons.
        """
        fallback_quality = {
            "quality_score": 0.50,
            "verdict": "Medium Quality",
            "reasons": ["Fallback search quality assessment."]
        }

        if not evaluated_candidates:
            return fallback_quality

        goal = intent.get("goal_summary", "")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return fallback_quality

        # Keep candidate summary small
        candidates_summary = []
        for ds in evaluated_candidates[:20]:
            candidates_summary.append({
                "title": ds.get("title"),
                "source": ds.get("source"),
                "evaluation_score": ds.get("evaluation_score"),
                "verdict": ds.get("verdict"),
                "evaluation_reason": ds.get("evaluation_reason")
            })

        prompt = f"""You are an expert data science evaluator.
User Goal: "{goal}"

Evaluate the overall retrieval and search quality of the following Top 20 candidate datasets.
Consider:
1. Do the candidate datasets cover the user's primary topic (and secondary concepts/location)?
2. Is there a diverse representation of relevant sources?
3. Are the dataset descriptions, verdicts, and scores suggesting a high-quality match?
4. Are there critical gaps or unrelated noise dominating the search result?

Candidates list:
{json.dumps(candidates_summary, indent=2)}

Format the output strictly as a JSON object with the following keys:
- "quality_score" (float, between 0.0 and 1.0, representing the overall quality of the retrieved set)
- "verdict" (string, one of: "High Quality", "Medium Quality", "Low Quality")
- "reasons" (array of strings, listing 2-3 specific strengths or identified gaps/limitations of the search results)
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
                print(f"[EvaluationAgent] Info: Model '{model_name}' failed to evaluate search quality: {e}")

        if not response_data:
            return fallback_quality

        # Ensure types and keys are correct
        return {
            "quality_score": float(response_data.get("quality_score", 0.50)),
            "verdict": str(response_data.get("verdict", "Medium Quality")),
            "reasons": list(response_data.get("reasons") or ["Search evaluation completed."])
        }

    def _apply_fallbacks(self, candidates: list[dict]) -> list[dict]:
        """Apply fallback values when Gemini is unavailable."""
        fallback_list = []
        for ds in candidates:
            ds_copy = dict(ds)
            ds_copy["evaluation_score"] = ds.get("composite_score", 0.0)
            ds_copy["confidence"] = 0.50
            ds_copy["verdict"] = "Good"
            ds_copy["evaluation_reason"] = "Rule-based baseline scoring (Gemini Evaluation fallback)."
            fallback_list.append(ds_copy)
            
        fallback_list.sort(key=lambda x: x["evaluation_score"], reverse=True)
        return fallback_list
