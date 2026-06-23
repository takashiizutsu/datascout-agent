"""
Skill: Score Dataset

Computes a composite score for a single Kaggle dataset based on four criteria:
    - Relevance  (weight 0.40): How many search keywords appear in the title.
    - Popularity (weight 0.20): Normalized download count (log-scaled).
    - Freshness  (weight 0.15): How recently the dataset was updated.
    - Usability  (weight 0.25): Kaggle's own usability rating (0–1).

All sub-scores are normalized to the 0–1 range before weighting.

Penalty rule:
    Datasets with ZERO keyword matches in the title receive a 60% penalty
    to their composite score. This prevents popular-but-irrelevant datasets
    from ranking above relevant ones.
"""

import math
from datetime import datetime, timezone


# --- Scoring weights ---
# These should sum to 1.0.
WEIGHT_RELEVANCE  = 0.40
WEIGHT_POPULARITY = 0.20
WEIGHT_FRESHNESS  = 0.15
WEIGHT_USABILITY  = 0.25

# --- Vector search combination weights ---
WEIGHT_VECTOR = 0.60
WEIGHT_RULE_BASED = 0.40

# --- Penalty for zero relevance ---
# Multiplier applied to the composite score when no keywords match the title.
# 0.4 means "keep 40% of the score" → a 60% penalty.
ZERO_RELEVANCE_PENALTY = 0.4


def score_dataset(dataset: dict, keywords: list[str], vector_similarity: float | None = None, intent: dict = None) -> dict:
    """
    Compute a composite score for a Kaggle or Hugging Face dataset.

    Args:
        dataset: A dict of Kaggle dataset metadata (from search_kaggle_datasets).
        keywords: The list of search keywords from the IntentAnalyzer.
        vector_similarity: Optional pre-computed semantic similarity score (0-1).
        intent: Optional structured intent dict containing primary_topic, secondary_concepts, and location.

    Returns:
        A new dict that is a copy of the input dataset with added keys:
            - relevance_score (float): 0–1
            - popularity_score (float): 0–1
            - freshness_score (float): 0–1
            - usability_score (float): 0–1
            - vector_similarity (float | None): Cosine similarity
            - topic_relevance (float): Topic relevance match (0.0, 0.5, or 1.0)
            - location_match (float): Location relevance match (0.0 or 1.0)
            - generic_penalty_applied (bool): True if generic term penalty was applied
            - composite_score (float): Weighted combination (0–1), with penalty and boosts
    """
    relevance  = _score_relevance(dataset, keywords)
    popularity = _score_popularity(dataset)
    freshness  = _score_freshness(dataset)
    usability  = _score_usability(dataset)

    # --- Weighted composite ---
    rule_based_composite = (
        WEIGHT_RELEVANCE  * relevance
        + WEIGHT_POPULARITY * popularity
        + WEIGHT_FRESHNESS  * freshness
        + WEIGHT_USABILITY  * usability
    )

    # --- Zero-relevance penalty ---
    # If none of the user's keywords appear in the dataset title, the dataset
    # is likely not relevant to the goal. Penalize its composite score so that
    # popularity/usability alone can't push it into the top ranks.
    # Bypassed if vector similarity is high (>= 0.40) indicating a strong semantic match.
    apply_penalty = relevance == 0.0 and keywords
    if apply_penalty:
        if vector_similarity is None or vector_similarity < 0.40:
            rule_based_composite *= ZERO_RELEVANCE_PENALTY

    # --- Combine Rule-based and Semantic similarity scores ---
    if vector_similarity is not None:
        composite = (WEIGHT_VECTOR * vector_similarity) + (WEIGHT_RULE_BASED * rule_based_composite)
    else:
        composite = rule_based_composite

    # ──────────────────────────────────────────────────────────────────
    # Intent-Aware Scoring (Topic Boost, Location Boost, Generic Penalty)
    # ──────────────────────────────────────────────────────────────────
    import re
    
    title_lower = (dataset.get("title") or "").lower()
    subtitle_lower = (dataset.get("subtitle") or "").lower()
    desc_lower = (dataset.get("description") or "").lower()
    
    topic_relevance = 0.0
    primary_match = False
    location_match = 0.0
    generic_penalty_applied = False
    
    # 1. Topic Boost
    primary_topic = intent.get("primary_topic") if intent else None
    if primary_topic and primary_topic != "general":
        primary_topic = primary_topic.lower()
        
        # Local copy of synonyms mapping for robust, self-contained matching
        LOCAL_SYNONYMS = {
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
        
        syns = LOCAL_SYNONYMS.get(primary_topic, [])
        primary_terms = [primary_topic] + syns
        
        # Direct Match (Title or Subtitle contains the primary topic or its synonyms)
        if any(term in title_lower for term in primary_terms) or any(term in subtitle_lower for term in primary_terms):
            topic_relevance = 1.0
            primary_match = True
            composite += 0.15
        # Indirect Match (Description contains the primary topic or its synonyms)
        elif any(term in desc_lower for term in primary_terms) or (dataset.get("tags") and any(term in tag.lower() for tag in dataset["tags"] for term in primary_terms)):
            topic_relevance = 0.5
            primary_match = True
            composite += 0.05
            
    # 2. Location Boost
    location_term = intent.get("location") if intent else None
    if location_term:
        loc_lower = location_term.lower()
        loc_terms = [loc_lower]
        if loc_lower == "united states":
            loc_terms.extend(["us", "usa", "u.s."])
        elif loc_lower == "united kingdom":
            loc_terms.extend(["uk", "gb", "u.k."])
            
        # Check if location appears in title, description, or tags
        matches_loc = (
            any(term in title_lower for term in loc_terms) 
            or any(term in subtitle_lower for term in loc_terms) 
            or any(term in desc_lower for term in loc_terms)
            or (dataset.get("tags") and any(term in tag.lower() for tag in dataset["tags"] for term in loc_terms))
        )
        if matches_loc:
            location_match = 1.0
            composite += 0.05

    # 3. Generic Term Penalty
    # List of generic concepts that yield false positives when they are not bound to the primary topic
    GENERIC_TERMS = {
        "factors", "prediction", "predict", "risk", "analysis", "report", 
        "market", "states", "united", "trend", "trends", "overview", "insights"
    }
    
    # Extract words from title
    title_words = set(re.findall(r"[a-z]+", title_lower))
    has_generic = bool(title_words & GENERIC_TERMS)
    
    # If the title matches generic search words but is NOT related to the primary topic,
    # apply a heavy penalty (65% penalty) to prevent it from ranking highly.
    if has_generic and not primary_match and primary_topic and primary_topic != "general":
        generic_penalty_applied = True
        composite *= 0.35
        
    # Cap final composite score at 1.0 and floor at 0.0
    composite = max(0.0, min(1.0, composite))

    # Return a copy of the dataset augmented with scores.
    scored = dict(dataset)
    scored["relevance_score"]  = round(relevance, 4)
    scored["popularity_score"] = round(popularity, 4)
    scored["freshness_score"]  = round(freshness, 4)
    scored["usability_score"]  = round(usability, 4)
    scored["vector_similarity"] = round(vector_similarity, 4) if vector_similarity is not None else None
    scored["topic_relevance"]  = topic_relevance
    scored["location_match"]  = location_match
    scored["generic_penalty_applied"] = generic_penalty_applied
    scored["composite_score"]  = round(composite, 4)

    # --- Add URL for display based on source repository ---
    if dataset.get("source") == "kaggle":
        scored["url"] = f"https://www.kaggle.com/datasets/{dataset['ref']}"
    else:
        scored["url"] = f"https://huggingface.co/datasets/{dataset['ref']}"

    return scored


# ──────────────────────────────────────────────────────────────────────
# Sub-score functions
# ──────────────────────────────────────────────────────────────────────

def _score_relevance(dataset: dict, keywords: list[str]) -> float:
    """
    Relevance score: fraction of keywords that appear in the dataset title.

    A simple keyword-match heuristic. Case-insensitive.
    Returns 0.0 if there are no keywords, 1.0 if all keywords match.
    """
    if not keywords:
        return 0.0

    title_lower = (dataset.get("title") or "").lower()
    matches = sum(1 for kw in keywords if kw.lower() in title_lower)
    return matches / len(keywords)


def _score_popularity(dataset: dict) -> float:
    """
    Popularity score: log-scaled download count, capped at 1.0.

    Uses log10(downloads + 1) normalized so that:
        - 0 downloads   → 0.0
        - 100 downloads → ~0.4
        - 10,000        → ~0.8
        - 100,000+      → 1.0

    The denominator (5.0 = log10(100_000)) sets the "max" reference.
    """
    downloads = dataset.get("download_count") or 0
    if downloads <= 0:
        return 0.0
    return min(math.log10(downloads + 1) / 5.0, 1.0)


def _score_freshness(dataset: dict) -> float:
    """
    Freshness score: how recently the dataset was updated.

    Uses an exponential decay with a half-life of ~365 days:
        - Updated today       → 1.0
        - Updated 1 year ago  → ~0.5
        - Updated 3 years ago → ~0.12
        - Updated 5+ years    → ~0.03
    """
    last_updated_str = dataset.get("last_updated")
    if not last_updated_str:
        return 0.0

    try:
        # Parse the timestamp string from Kaggle (e.g., "2024-01-14 17:10:25.240000").
        last_updated = datetime.fromisoformat(last_updated_str)
        # Make timezone-aware if naive.
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_old = max((now - last_updated).days, 0)
    except (ValueError, TypeError):
        return 0.0

    # Exponential decay: score = 2^(-days / 365)
    return 2 ** (-days_old / 365)


def _score_usability(dataset: dict) -> float:
    """
    Usability score: directly uses Kaggle's usability_rating (already 0–1).

    Returns 0.0 if the rating is missing.
    """
    rating = dataset.get("usability_rating")
    if rating is None:
        return 0.0
    return float(rating)
