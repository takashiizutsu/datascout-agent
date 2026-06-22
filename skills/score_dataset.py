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

# --- Penalty for zero relevance ---
# Multiplier applied to the composite score when no keywords match the title.
# 0.4 means "keep 40% of the score" → a 60% penalty.
ZERO_RELEVANCE_PENALTY = 0.4


def score_dataset(dataset: dict, keywords: list[str]) -> dict:
    """
    Compute a composite score for a Kaggle dataset.

    Args:
        dataset: A dict of Kaggle dataset metadata (from search_kaggle_datasets).
        keywords: The list of search keywords from the IntentAnalyzer.

    Returns:
        A new dict that is a copy of the input dataset with added keys:
            - relevance_score (float): 0–1
            - popularity_score (float): 0–1
            - freshness_score (float): 0–1
            - usability_score (float): 0–1
            - composite_score (float): Weighted combination (0–1), with penalty
    """
    relevance  = _score_relevance(dataset, keywords)
    popularity = _score_popularity(dataset)
    freshness  = _score_freshness(dataset)
    usability  = _score_usability(dataset)

    # --- Weighted composite ---
    composite = (
        WEIGHT_RELEVANCE  * relevance
        + WEIGHT_POPULARITY * popularity
        + WEIGHT_FRESHNESS  * freshness
        + WEIGHT_USABILITY  * usability
    )

    # --- Zero-relevance penalty ---
    # If none of the user's keywords appear in the dataset title, the dataset
    # is likely not relevant to the goal. Penalize its composite score so that
    # popularity/usability alone can't push it into the top ranks.
    if relevance == 0.0 and keywords:
        composite *= ZERO_RELEVANCE_PENALTY

    # Return a copy of the dataset augmented with scores.
    scored = dict(dataset)
    scored["relevance_score"]  = round(relevance, 4)
    scored["popularity_score"] = round(popularity, 4)
    scored["freshness_score"]  = round(freshness, 4)
    scored["usability_score"]  = round(usability, 4)
    scored["composite_score"]  = round(composite, 4)

    # --- Add Kaggle URL for display ---
    scored["url"] = f"https://www.kaggle.com/datasets/{dataset['ref']}"

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
