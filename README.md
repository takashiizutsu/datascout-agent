# DataScout Agent

**A Kaggle API-powered AI agent system that helps analysts discover, evaluate, and rank public datasets for any analytical goal.**

> 🎓 Kaggle Capstone Submission

---

## 📌 Problem Statement

Data analysts often spend hours searching for high-quality, relevant datasets for new projects. Searching public repositories like Kaggle manually involves navigating mismatched keywords, filtering out low-usability uploads, and guessing dataset quality based on popularity alone. This manual vetting process slows down the initiation of analytical projects and often leads to sub-optimal dataset selection.

## 💡 Solution

**DataScout Agent** automates the dataset discovery and vetting lifecycle. Given a natural language analysis goal, the system:
1. Translates the high-level goal into an effective keyword search strategy.
2. Queries the Kaggle API to extract candidate datasets.
3. Automatically evaluates and scores each candidate against a multi-dimensional metric framework.
4. Recommends the top 5 most relevant, high-quality, and up-to-date datasets, detailing why they were chosen and highlighting any potential data limitations.

---

## 🏗️ Multi-Agent Architecture

DataScout Agent follows a modular agent-and-skill structure where specialized agents collaborate to analyze, retrieve, score, and format recommendations.

```
       [ User Analysis Goal ]
                 │
                 ▼
     ┌──────────────────────┐
     │   Intent Analyzer    │  <-- Strips noise, expands synonyms
     └───────────┬──────────┘
                 │ (Search Keywords & Domain)
                 ▼
     ┌──────────────────────┐
     │  Dataset Retriever   │  <-- Queries Kaggle API with deduping
     └───────────┬──────────┘
                 │ (Candidate Datasets Metadata)
                 ▼
     ┌──────────────────────┐
     │  Dataset Evaluator   │  <-- Computes composite quality scores
     └───────────┬──────────┘
                 │ (Scored & Ranked Candidates)
                 ▼
     ┌──────────────────────┐
     │  Recommendation Gen. │  <-- Builds reasons, warnings, and limitations
     └───────────┬──────────┘
                 │
                 ▼
     [ Recommended Datasets ]
```

### 1. Intent Analyzer (`agents/intent_analyzer.py`)
* Analyzes natural language inputs using rule-based tokenization.
* Strips generic noise and filler words (e.g., *market*, *analysis*, *dataset*, *report*).
* Expands query keywords with target domain synonyms (e.g., *dementia* triggers *alzheimer* and *cognitive decline*).
* Detects the general subject domain (e.g., *healthcare*, *finance*, *climate*).

### 2. Dataset Retriever (`agents/dataset_retriever.py`)
* Takes search keywords and orchestrates concurrent queries to the Kaggle API.
* Merges results and deduplicates datasets that match multiple keywords.

### 3. Dataset Evaluator (`agents/dataset_evaluator.py` & `skills/score_dataset.py`)
Scores each candidate dataset out of `1.0` using a weighted formula:
* **Relevance (40%):** Matches search keywords inside the title. Applies a **60% penalty** if there are zero keyword matches to suppress popular but irrelevant results.
* **Popularity (20%):** Normalized, log-scaled download count.
* **Freshness (15%):** Exponential time-decay based on the last update timestamp (365-day half-life).
* **Usability (25%):** Direct inclusion of Kaggle's usability score.

### 4. Recommendation Generator (`skills/generate_recommendation.py`)
* Ranks candidate datasets by their composite score.
* Generates a user-friendly report highlighting:
  * **Reason for Selection:** Identifies the strongest quality score factors (e.g., "highly popular", "recently updated").
  * **Limitation Warnings:** Flags risks such as small dataset sizes, lack of recent updates, low usability, or missing direct keyword matches in the title.
  * **Weak Relevance Warnings:** Displays a warning banner if no dataset scores highly.
  * **Broadening Suggestions:** Suggests alternative search terms if zero results are found.

---

## 🖥️ Streamlit UI

The project features a clean, responsive web interface built with **Streamlit** ([app.py](file:///c:/Users/0701i/OneDrive/Documents/datascout-agent/app.py)):

* **Interactive Search:** Input goals and execute the discovery agent in real time.
* **Visual Progress:** Displays a step-by-step processing status of the analysis, retrieval, and evaluation steps.
* **Polished Metric Cards:** Visualizes dataset performance metrics (Composite Score, Downloads, Usability, Last Updated) at a glance.
* **Kaggle API Status Indicator:** Sidebar status monitor showing if environment credentials are set up correctly.

---

## 📂 Project Structure

```
datascout-agent/
│
├── agents/
│   ├── dataset_evaluator.py     # Orchestrates scoring and ranking candidates
│   ├── dataset_retriever.py     # Searches Kaggle API and dedups results
│   └── intent_analyzer.py       # Extracts keywords and expands synonyms
│
├── skills/
│   ├── generate_recommendation.py # Formats top recommendation reports
│   ├── score_dataset.py         # Multi-criteria scoring algorithms
│   └── search_kaggle_datasets.py # Interacts with Kaggle API SDK
│
├── app.py                      # Streamlit web application
├── main.py                     # CLI entrypoint
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment variables file
└── .gitignore                  # Git ignore rules (includes .env)
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd datascout-agent
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory (based on `.env.example`):
   ```env
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```
   *(Note: You can retrieve your API key from the "Settings" tab of your Kaggle profile page by clicking "Create New Token".)*

---

## 🚀 Usage

### Streamlit Web Interface
Run the following command to start the web app:
```bash
streamlit run app.py
```
The app will open automatically in your browser (typically at `http://localhost:8501`).

### Command Line Interface (CLI)
You can also run the agent directly in the console:
```bash
python main.py
```

#### Usage Example (CLI):
```
Describe your analysis goal: > I want to analyze the US dementia market.

[Step 1] Analyzing your goal...
  Keywords: ['dementia', 'alzheimer', 'cognitive decline']
  Domain:   healthcare

[Step 2] Searching Kaggle for datasets...
[DatasetRetriever] Searching Kaggle with 3 keyword(s): ['dementia', 'alzheimer', 'cognitive decline']
[DatasetRetriever] Found 37 unique candidate datasets.

[Step 3] Evaluating and ranking datasets...

======================================================================
  DataScout Agent — Top 5 Recommendations
  Goal: "I want to analyze the US dementia market."
  Domain: healthcare
  Keywords: dementia, alzheimer, cognitive decline
======================================================================

  #1  Alzheimer's Disease Dataset
  ------------------------------------------------------------------
  Score:      0.60  (relevance=0.33, popularity=0.71, freshness=0.08, usability=1.00)
  Downloads:  35,241
  Usability:  10.0
  Updated:    2023-12-05 06:17:10
  URL:        https://www.kaggle.com/datasets/rabeesh/alzheimers-disease-dataset
  Reason:     Partial keyword match in title; very popular (35,241 downloads); excellent usability rating.
  Limitation: Dataset has not been updated in over a year.
```

---

## 🔮 Future Improvements

- [ ] **LLM Intent Expansion**: Transition from rule-based keyword mapping to LLM-powered context understanding using Google Gemini.
- [ ] **Dynamic Weight Adjustments**: Allow users to adjust evaluation weights (e.g., prioritize freshness over popularity) directly in the UI.
- [ ] **Dataset Previewing**: Download sample rows of the top datasets to display schemas and summary statistics directly in the Streamlit UI.
- [ ] **Multi-Source Crawling**: Expand discovery to Hugging Face, OpenML, and data.gov.
- [ ] **Offline Knowledge Base**: Enable fallback searching on a local dataset index.

---

## 📄 License

This project is open-source and intended for educational and Capstone evaluation purposes.
