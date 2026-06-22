# DataScout Agent

**A Kaggle API-powered AI agent that helps analysts discover and evaluate public datasets.**

> 🎓 Kaggle Capstone Submission

---

## Purpose

DataScout Agent is an intelligent assistant that helps data analysts go from a high-level analysis goal (e.g., *"I want to analyze dementia in the US"*) to a shortlist of the best publicly available datasets on Kaggle. It uses the Kaggle API for real-time dataset discovery and an LLM-powered evaluation pipeline to rank results.

## Target Behavior (MVP)

1. **User inputs an analysis goal** — a plain-English description of what they want to study.
2. **Keyword extraction** — the agent uses an LLM to extract effective search keywords from the goal.
3. **Kaggle dataset search** — the agent queries the Kaggle API to find matching datasets.
4. **Multi-criteria evaluation** — each dataset is scored on:
   - **Relevance** — how well does the dataset match the stated goal?
   - **Usability** — Kaggle's usability rating, file formats, documentation quality.
   - **Freshness** — when was the dataset last updated?
   - **Popularity** — download count, vote count, community engagement.
5. **Top-5 recommendation** — the agent returns the 5 best datasets with clear reasons and noted limitations.

## Architecture

```
User Goal (natural language)
    │
    ▼
┌──────────────────┐
│  Intent Analyzer  │   Extracts search keywords from the goal
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  Dataset Retriever    │   Searches Kaggle API with extracted keywords
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Dataset Evaluator    │   Scores & ranks results, generates recommendations
└────────┬─────────────┘
         │
         ▼
   Top-5 Recommendations
   (with reasons & limitations)
```

### Agents (`agents/`)

| Agent | Responsibility |
|---|---|
| `intent_analyzer.py` | Uses LLM to extract search keywords from a user's analysis goal |
| `dataset_retriever.py` | Queries the Kaggle API to find datasets matching the keywords |
| `dataset_evaluator.py` | Scores datasets on relevance, usability, freshness, and popularity |

### Skills (`skills/`)

| Skill | Responsibility |
|---|---|
| `search_kaggle_datasets.py` | Wraps the Kaggle API to search and fetch dataset metadata |
| `score_dataset.py` | Computes a composite score for a dataset across evaluation criteria |
| `generate_recommendation.py` | Produces a natural-language recommendation with reasons and limitations |

### Data (Optional / Future)

| File | Description |
|---|---|
| `dataset_catalog.csv` | Optional curated catalog for offline fallback or domain-specific boosting |

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini (via `google-generativeai`) |
| Dataset Source | Kaggle API (via `kaggle` Python package) |
| Language | Python 3.10+ |

## Setup

1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
4. Ensure your Kaggle API credentials are configured:
   - Place your `kaggle.json` at `~/.kaggle/kaggle.json`, **or**
   - Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your `.env` file.
5. Run the agent:
   ```bash
   python main.py
   ```

## Planned Features (Post-MVP)

- [ ] Conversational follow-up and query refinement
- [ ] Local dataset catalog for offline / hybrid search
- [ ] Dataset preview and column-level analysis
- [ ] Exportable recommendation reports (Markdown / PDF)
- [ ] Multi-source search (HuggingFace, data.gov, etc.)

## License

This project is for educational purposes as part of a Kaggle Capstone submission.
