"""
Skill: Generate Recommendation

Produces a formatted, human-readable recommendation report for the top-ranked
datasets. No LLM needed — uses rule-based template formatting for the MVP.

Edge cases handled:
    - No datasets found → helpful message with keyword broadening suggestion.
    - Fewer than 5 results → shows whatever is available.
    - Weak relevance across all results → displays a warning banner.
"""


def generate_recommendation(scored_datasets: list[dict], intent: dict) -> str:
    """
    Generate a formatted recommendation report for the top scored datasets.

    Args:
        scored_datasets: A list of dataset dicts with composite_score and
                         sub-scores already computed (sorted best-first).
        intent: The structured intent dict from IntentAnalyzer.

    Returns:
        A formatted multi-line string containing the full recommendation report.
    """
    keywords = intent.get("keywords", [])
    goal = intent.get("goal_summary", "N/A")
    domain = intent.get("domain") or "General"

    # --- Edge case: no results ---
    if not scored_datasets:
        return _no_results_message(keywords)

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  DataScout Agent — Top {len(scored_datasets)} Recommendations")
    lines.append(f"  Goal: \"{goal}\"")
    lines.append(f"  Domain: {domain}")
    lines.append(f"  Keywords: {', '.join(keywords)}")
    if intent:
        lines.append(f"  Primary Topic: {intent.get('primary_topic', 'N/A')}")
        lines.append(f"  Secondary Concepts: {', '.join(intent.get('secondary_concepts', []))}")
        if intent.get('location'):
            lines.append(f"  Location: {intent.get('location')}")
    lines.append("=" * 70)

    # --- Edge case: weak relevance warning ---
    # If the best dataset has a relevance score below 0.3, warn the user.
    best_relevance = max(ds["relevance_score"] for ds in scored_datasets) if scored_datasets else 0.0
    if best_relevance < 0.3:
        lines.append("")
        lines.append("  ⚠  WEAK RELEVANCE WARNING")
        lines.append("  None of the top results strongly match your keywords.")
        lines.append("  Consider rephrasing your goal or using more specific terms.")
        lines.append("  " + "-" * 66)

    for rank, ds in enumerate(scored_datasets, start=1):
        lines.append("")
        lines.append(f"  #{rank}  {ds['title']}")
        lines.append("  " + "-" * 66)

        # Build composite sub-scores representation
        score_parts = [f"relevance={ds['relevance_score']:.2f}"]
        if ds.get("vector_similarity") is not None:
            score_parts.append(f"semantic={ds['vector_similarity']:.2f}")
        score_parts.extend([
            f"popularity={ds['popularity_score']:.2f}",
            f"freshness={ds['freshness_score']:.2f}",
            f"usability={ds['usability_score']:.2f}"
        ])

        # Add intent-aware boosts/penalties to score parts
        intent_parts = []
        if ds.get("topic_relevance", 0.0) > 0.0:
            intent_parts.append(f"topic_boost={'+0.15' if ds['topic_relevance'] == 1.0 else '+0.05'}")
        if ds.get("location_match", 0.0) > 0.0:
            intent_parts.append("location_boost=+0.05")
        if ds.get("generic_penalty_applied"):
            intent_parts.append("generic_penalty=65%_reduction")
        if intent_parts:
            score_parts.append(f"intent: {', '.join(intent_parts)}")

        lines.append(f"  Score:      {ds['composite_score']:.2f}  ({', '.join(score_parts)})")
        lines.append(f"  Downloads:  {ds['download_count']:,}")
        lines.append(f"  Usability:  {ds['usability_rating']}")
        lines.append(f"  Updated:    {ds['last_updated']}")
        lines.append(f"  URL:        {ds['url']}")
        source_name = "Kaggle" if ds.get("source") == "kaggle" else "Hugging Face"
        lines.append(f"  Source:     {source_name}")

        # --- Reason ---
        reason = _build_reason(ds, keywords)
        lines.append(f"  Reason:     {reason}")

        # --- Limitation ---
        limitation = _build_limitation(ds)
        lines.append(f"  Limitation: {limitation}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    return "\n".join(lines)


def _no_results_message(keywords: list[str]) -> str:
    """
    Return a helpful message when no datasets are found.
    Suggests broader keywords the user could try.
    """
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  No relevant Kaggle datasets were found for this query.")
    lines.append("=" * 70)
    lines.append("")
    if keywords:
        lines.append(f"  Keywords searched: {', '.join(keywords)}")
    lines.append("")
    lines.append("  Suggestions:")
    lines.append("    - Try broader or more general keywords.")
    lines.append("    - Remove overly specific terms.")
    lines.append("    - Use common names or abbreviations.")
    if keywords:
        # Suggest using individual keywords as separate searches.
        single = keywords[0] if keywords else "your topic"
        lines.append(f'    - Example: try just "{single}" instead of the full phrase.')
    lines.append("")
    return "\n".join(lines)


def _build_reason(ds: dict, keywords: list[str]) -> str:
    """
    Build a short reason string explaining why this dataset was recommended.
    Uses the sub-scores to pick the strongest factors.
    """
    parts = []

    # Relevance & Semantic similarity
    v_sim = ds.get("vector_similarity")
    if v_sim is not None:
        if v_sim >= 0.70:
            parts.append(f"strong semantic match ({v_sim:.2f})")
        elif v_sim >= 0.50:
            parts.append(f"moderate semantic match ({v_sim:.2f})")

    # Relevance
    if ds["relevance_score"] >= 0.5:
        title_lower = ds["title"].lower()
        matched = [kw for kw in keywords if kw.lower() in title_lower]
        parts.append(f"Title matches {len(matched)}/{len(keywords)} keywords")
    elif ds["relevance_score"] > 0:
        parts.append("Partial keyword match in title")

    # Popularity
    if ds["popularity_score"] >= 0.7:
        parts.append(f"very popular ({ds['download_count']:,} downloads)")
    elif ds["popularity_score"] >= 0.4:
        parts.append(f"moderately popular ({ds['download_count']:,} downloads)")

    # Usability
    if ds["usability_score"] >= 0.9:
        parts.append("excellent usability rating")
    elif ds["usability_score"] >= 0.7:
        parts.append("good usability rating")

    # Freshness
    if ds["freshness_score"] >= 0.8:
        parts.append("recently updated")

    if not parts:
        return "Matched search query."

    # Capitalize first part, join with semicolons.
    parts[0] = parts[0][0].upper() + parts[0][1:]
    return "; ".join(parts) + "."


def _build_limitation(ds: dict) -> str:
    """
    Build a short limitation string noting potential concerns.
    """
    issues = []

    # Relevance concern
    if ds["relevance_score"] == 0.0:
        v_sim = ds.get("vector_similarity")
        if v_sim is None or v_sim < 0.40:
            issues.append("no keyword match in title — verify relevance manually")
        else:
            issues.append(f"no keyword match in title (semantic similarity is {v_sim:.2f})")

    # Freshness concern
    if ds["freshness_score"] < 0.3:
        issues.append("dataset may be outdated (not recently updated)")
    elif ds["freshness_score"] < 0.5:
        issues.append("dataset has not been updated in over a year")

    # Size concern
    total_bytes = ds.get("total_bytes") or 0
    if total_bytes > 1_000_000_000:  # > 1 GB
        size_gb = total_bytes / 1_000_000_000
        issues.append(f"large dataset ({size_gb:.1f} GB) — may be slow to download")
    elif total_bytes < 10_000:  # < 10 KB
        issues.append("very small dataset — may lack sufficient data")

    # Usability concern
    if ds["usability_score"] < 0.5:
        issues.append("low usability rating on Kaggle")

    # Popularity concern
    if ds["popularity_score"] < 0.2:
        issues.append("low community engagement")

    if not issues:
        return "No major concerns."

    issues[0] = issues[0][0].upper() + issues[0][1:]
    return "; ".join(issues) + "."
