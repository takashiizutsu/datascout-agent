"""
Intent Analyzer Agent

Takes a user's natural-language analysis goal and extracts search keywords
using rule-based NLP (no LLM required for the MVP).

Approach:
    1. Lowercase and tokenize the input.
    2. Remove common English stop words and filler phrases.
    3. Keep the remaining meaningful terms as keywords.
    4. Expand with a small synonym map for common research domains.

Example:
    Goal:   "I want to find datasets for dementia analysis."
    Output: {"keywords": ["dementia", "alzheimer", "cognitive decline"],
             "domain": "healthcare",
             "goal_summary": "I want to find datasets for dementia analysis."}
"""

import re


# Common English stop words + dataset-search filler words.
# These are stripped from the user's goal before extracting keywords.
STOP_WORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "a", "an", "the", "this", "that", "these", "those",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "about",
    "as", "into", "through", "during", "before", "after", "between",
    "and", "but", "or", "nor", "not", "no", "so", "if", "then",
    "want", "need", "like", "looking", "find", "search", "get", "look",
    "dataset", "datasets", "data", "analyze", "analysis", "study",
    "explore", "investigate", "research", "related", "regarding",
    "some", "any", "all", "each", "every", "most", "very", "really",
    "please", "help", "me", "us",
    # Generic business / research terms that add noise to Kaggle searches.
    "market", "industry", "business", "report", "trend", "trends",
    "sector", "overview", "insights", "statistics", "stats",
}

# A lightweight synonym/expansion map.
# When a keyword matches a key, the associated terms are added as extra
# search queries to broaden Kaggle coverage.
SYNONYM_MAP = {
    "dementia":       ["alzheimer", "cognitive decline"],
    "alzheimer":      ["dementia", "cognitive decline"],
    "cancer":         ["tumor", "oncology"],
    "diabetes":       ["blood sugar", "glucose"],
    "heart":          ["cardiac", "cardiovascular"],
    "housing":        ["real estate", "property prices"],
    "climate":        ["weather", "global warming", "temperature"],
    "covid":          ["coronavirus", "pandemic", "sars-cov-2"],
    "education":      ["student performance", "school"],
    "employment":     ["jobs", "unemployment", "labor market"],
    "crime":          ["criminal", "public safety"],
    "pollution":      ["air quality", "emissions"],
    "mental health":  ["depression", "anxiety", "psychological"],
}

# Simple domain detection: if any keyword matches a key, assign that domain.
DOMAIN_MAP = {
    "healthcare":     ["dementia", "alzheimer", "cancer", "diabetes", "heart",
                       "hospital", "patient", "medical", "disease", "health",
                       "drug", "clinical", "covid", "mental health"],
    "finance":        ["stock", "finance", "bank", "investment", "trading",
                       "credit", "loan", "market"],
    "environment":    ["climate", "weather", "pollution", "emission",
                       "temperature", "carbon", "ocean", "forest"],
    "education":      ["education", "student", "school", "university",
                       "learning", "exam", "grade"],
    "technology":     ["software", "programming", "ai", "machine learning",
                       "deep learning", "nlp", "computer vision"],
    "social":         ["crime", "employment", "population", "census",
                       "immigration", "poverty", "housing"],
}


class IntentAnalyzer:
    """Extracts structured search intent from a user's analysis goal (rule-based)."""

    def __init__(self):
        """Initialize the IntentAnalyzer. No model needed for rule-based MVP."""
        pass

    def analyze(self, goal: str) -> dict:
        """
        Analyze a user's analysis goal and return structured search intent.

        Args:
            goal: The user's natural-language analysis goal.

        Returns:
            A dict with:
                - keywords (list[str]): Search terms for the Kaggle API.
                - domain (str | None): The detected broad domain.
                - goal_summary (str): The original goal text.
                - primary_topic (str): The core entity/subject (e.g. 'diabetes').
                - secondary_concepts (list[str]): Concept qualifiers.
                - location (str | None): Geographic scope.
        """
        text_lower = goal.lower()

        # --- Step 1: Extract Location ---
        location = None
        locations_map = {
            "united states": "United States",
            "us": "United States",
            "usa": "United States",
            "united kingdom": "United Kingdom",
            "uk": "United Kingdom",
            "india": "India",
            "canada": "Canada",
            "germany": "Germany",
        }
        for loc_key, loc_name in locations_map.items():
            if re.search(r"\b" + re.escape(loc_key) + r"\b", text_lower):
                location = loc_name
                # Remove location terms to avoid treating location keywords as primary topic
                text_lower = re.sub(r"\b" + re.escape(loc_key) + r"\b", "", text_lower)
                break

        # --- Step 2: Tokenize the remaining text ---
        tokens = re.findall(r"[a-z]+", text_lower)

        # --- Step 3: Remove stop words ---
        keywords = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

        # --- Step 4: Deduplicate while preserving order ---
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        # --- Step 5: Expand with synonyms ---
        expanded = list(unique_keywords)
        for kw in unique_keywords:
            if kw in SYNONYM_MAP:
                for synonym in SYNONYM_MAP[kw]:
                    if synonym not in seen:
                        expanded.append(synonym)
                        seen.add(synonym)

        # --- Step 6: Identify Primary Topic & Secondary Concepts ---
        primary_topic = None
        # Try to match keywords against known SYNONYM_MAP keys (core medical/subject domains)
        for kw in unique_keywords:
            if kw in SYNONYM_MAP:
                primary_topic = kw
                break
        
        # If no known key matches, default to the first extracted keyword
        if not primary_topic and unique_keywords:
            primary_topic = unique_keywords[0]
            
        secondary_concepts = [kw for kw in unique_keywords if kw != primary_topic]

        # --- Step 7: Detect domain ---
        domain = self._detect_domain(expanded)

        return {
            "keywords": expanded,
            "domain": domain,
            "goal_summary": goal.strip(),
            "primary_topic": primary_topic or "general",
            "secondary_concepts": secondary_concepts,
            "location": location,
        }

    def _detect_domain(self, keywords: list[str]) -> str | None:
        """
        Detect the broad domain from keywords using the DOMAIN_MAP.

        Returns the first matching domain, or None if no match.
        """
        keyword_set = set(keywords)
        for domain, terms in DOMAIN_MAP.items():
            if keyword_set & set(terms):  # set intersection
                return domain
        return None
