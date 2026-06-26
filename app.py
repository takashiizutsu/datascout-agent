"""
DataScout Agent — Streamlit Web Interface

A clean and simple web application for searching, evaluating, and ranking Kaggle datasets
based on a natural language analysis goal.

To run this application, run this command in your terminal:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Ensure environment variables (.env) are loaded BEFORE importing agents
# because the Kaggle client initializes and reads credentials from environment at import time.
load_dotenv()

# Import the core agent and skill logic from the project
from agents.intent_analyzer import IntentAnalyzer
from agents.query_planner_agent import QueryPlannerAgent
from agents.evaluation_agent import EvaluationAgent
from skills.search_kaggle_datasets import search_kaggle_multi
from skills.search_huggingface_datasets import search_huggingface_multi
from skills.score_dataset import score_dataset
from skills.generate_recommendation import generate_recommendation, _build_reason, _build_limitation
from skills.vector_search import compute_vector_similarities

def update_progress_dashboard(placeholder, current_step_idx):
    steps = [
        ("🧠 Intent Analysis", "Analyzing search intent & extracting keywords"),
        ("📋 Query Planning", "Generating structured search plan using Gemini"),
        ("📡 Searching Kaggle", "Retrieving datasets from Kaggle API"),
        ("🤗 Searching Hugging Face", "Retrieving datasets from Hugging Face Hub"),
        ("🔀 Merging Results", "Merging and deduplicating candidate datasets"),
        ("📊 Embedding Ranking", "Pre-filtering and embedding semantic similarity checking"),
        ("🤖 Gemini Evaluation", "Independently evaluating candidates using Gemini"),
        ("🎯 Final Recommendations", "Finalizing recommendations report")
    ]
    with placeholder.container(border=True):
        st.markdown("#### ⚡ Execution Pipeline Progress")
        # Layout steps in a compact vertical list
        for idx, (step_name, detail) in enumerate(steps, start=1):
            if idx < current_step_idx:
                st.markdown(f"✅ **{step_name}** — *{detail}*")
            elif idx == current_step_idx:
                st.markdown(f"🌀 **{step_name}** — *{detail}...*")
            else:
                st.markdown(f"⚪ *{step_name}*")

def update_progress(status_text, progress_bar, progress_val, text_content):
    status_text.markdown(text_content)
    progress_bar.progress(progress_val)

# Configure the Streamlit page layout and theme
st.set_page_config(
    page_title="DataScout Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Component ---
with st.sidebar:
    st.title("🛡️ DataScout Setup")
    st.markdown("---")
    st.markdown("### Kaggle API Status")
    
    # Safely check for Kaggle credentials without exposing the actual values
    username_exists = bool(os.getenv("KAGGLE_USERNAME"))
    key_exists = bool(os.getenv("KAGGLE_KEY"))
    
    if username_exists and key_exists:
        st.success("Kaggle API Configured")
    else:
        st.error("Kaggle API Credentials Missing")
        st.warning("Please ensure KAGGLE_USERNAME and KAGGLE_KEY are set in your `.env` file.")
        
    st.markdown("---")
    st.markdown("### Hugging Face API Status")
    hf_token_exists = bool(os.getenv("HF_TOKEN"))
    if hf_token_exists:
        st.success("HF Hub Token Configured")
    else:
        st.info("HF Token Missing (Public Search Mode)")
        
    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. **Intent Analyzer** extracts search keywords from your goal.\n"
        "2. **Dataset Retriever** queries Kaggle and Hugging Face using those keywords.\n"
        "3. **Dataset Evaluator** scores and ranks candidate datasets using relevance, popularity, freshness, and usability.\n"
        "4. **Recommendations** present the top 5 candidates with detailed reasons and limitations."
    )

# --- Main Page Layout ---
st.title("🔍 DataScout Agent")
st.markdown("### *AI-Powered Dataset Discovery & Evaluation (Kaggle & Hugging Face)*")
st.write(
    "Describe what you want to analyze in plain English. The agent will analyze your intent, "
    "retrieve candidate datasets from Kaggle, and evaluate them based on multiple quality criteria."
)

# Text Input for the user's analysis goal
goal_input = st.text_input(
    label="Enter your analysis goal:",
    placeholder="e.g., I want to analyze the US dementia market.",
    help="Type what you want to study. Keep it clear and specific (e.g., 'dementia trends', 'climate change', 'housing prices')."
)

# Column wrapper for the Search button to keep the UI clean
btn_col1, btn_col2 = st.columns([1, 5])
with btn_col1:
    search_clicked = st.button("Search Datasets", type="primary", use_container_width=True)

# --- Search Execution Pipeline ---
# --- Search Execution Pipeline ---
if search_clicked:
    if not (username_exists and key_exists):
        st.error("Kaggle credentials missing. Please set KAGGLE_USERNAME and KAGGLE_KEY in your .env file.")
    elif not goal_input.strip():
        st.warning("Please enter an analysis goal before searching.")
    else:
        # Create progress container immediately visible to the user
        progress_container = st.empty()
        with progress_container.container(border=True):
            st.markdown("### ⚡ Discovery Pipeline Progress")
            status_text = st.empty()
            progress_bar = st.progress(0.0)
        
        try:
            # Step 1: Extract Intent & Keywords
            update_progress(status_text, progress_bar, 0.0, "#### 🧠 Step 1/7: Analyzing search intent & extracting keywords...")
            analyzer = IntentAnalyzer()
            intent = analyzer.analyze(goal_input)
            
            # Step 2: Generate Search Plan using Gemini
            update_progress(status_text, progress_bar, 0.15, "#### 📋 Step 2/7: Generating structured search plan using Gemini...")
            planner = QueryPlannerAgent()
            search_plan = planner.generate_plan(goal_input)
            
            # Form queries list
            if search_plan and search_plan.get("search_queries"):
                queries = search_plan["search_queries"]
            else:
                queries = intent.get("keywords", [])
                
            if not queries:
                progress_container.empty()
                st.error("No relevant keywords or search queries could be generated.")
            else:
                # Step 3: Searching Kaggle & Hugging Face
                update_progress(status_text, progress_bar, 0.30, "#### 📡 Step 3/7: Searching Kaggle API...")
                try:
                    kaggle_datasets = search_kaggle_multi(queries, max_per_keyword=10)
                    for ds in kaggle_datasets:
                        ds["source"] = "kaggle"
                except Exception as e:
                    print(f"Kaggle search failed: {e}")
                    kaggle_datasets = []
                
                update_progress(status_text, progress_bar, 0.40, "#### 🤗 Step 3/7: Searching Hugging Face Hub...")
                try:
                    hf_datasets = search_huggingface_multi(queries, max_per_keyword=10)
                except Exception as e:
                    print(f"HF search failed: {e}")
                    hf_datasets = []
                
                # Step 4: Merging Results
                update_progress(status_text, progress_bar, 0.50, "#### 🔀 Step 4/7: Merging and deduplicating candidate datasets...")
                seen_keys = set()
                candidates = []
                for ds in kaggle_datasets + hf_datasets:
                    key = (ds.get("source"), ds.get("ref"))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        candidates.append(ds)
                # Cap candidates to 100
                candidates = candidates[:100]
                
                if not candidates:
                    progress_container.empty()
                    st.error("No relevant Kaggle/Hugging Face datasets were found for this query.")
                    st.markdown("### Suggestions:")
                    st.markdown("- Try broader or more general keywords.")
                    st.markdown("- Remove overly specific terms.")
                    st.markdown("- Use common names or abbreviations.")
                    if intent.get('keywords'):
                        st.markdown(f'- Example: try just **"{intent["keywords"][0]}"** instead of the full phrase.')
                else:
                    # Step 5: Embedding Ranking
                    update_progress(status_text, progress_bar, 0.65, "#### 📊 Step 5/7: Scoring candidates & calculating semantic similarities...")
                    keywords = intent.get("keywords", [])
                    query = intent.get("goal_summary", "")
                    
                    similarities = None
                    if query and candidates:
                        try:
                            similarities = compute_vector_similarities(query, candidates)
                        except Exception as e:
                            print(f"Embedding computation failed: {e}")
                            
                    scored = []
                    for idx, ds in enumerate(candidates):
                        sim = similarities[idx] if similarities is not None else None
                        scored_ds = score_dataset(ds, keywords, vector_similarity=sim, intent=intent)
                        scored.append(scored_ds)
                        
                    scored.sort(key=lambda d: d["composite_score"], reverse=True)
                    top_20 = scored[:20]
                    
                    # Step 6: Gemini Evaluation
                    update_progress(status_text, progress_bar, 0.80, "#### 🤖 Step 6/7: Evaluating candidates & overall search quality with Gemini...")
                    evaluation_agent = EvaluationAgent()
                    evaluated_candidates = evaluation_agent.evaluate_candidates(top_20, intent)
                    quality_assessment = evaluation_agent.evaluate_search_quality(evaluated_candidates, intent)
                    
                    # Step 7: Final Recommendations
                    update_progress(status_text, progress_bar, 0.95, "#### 🎯 Step 7/7: Formatting final recommendations report...")
                    top_datasets = evaluated_candidates[:10]
                    report = generate_recommendation(top_datasets, intent, quality_assessment=quality_assessment, search_plan=search_plan)
                    
                    # Calculate diagnosis counts
                    kaggle_retrieved = len(kaggle_datasets)
                    hf_retrieved = len(hf_datasets)
                    total_retrieved = kaggle_retrieved + hf_retrieved
                    
                    kaggle_merged = kaggle_retrieved
                    hf_merged = hf_retrieved
                    total_merged = total_retrieved
                    
                    seen_keys = set()
                    deduplicated_list = []
                    for ds in kaggle_datasets + hf_datasets:
                        key = (ds.get("source"), ds.get("ref"))
                        if key not in seen_keys:
                            seen_keys.add(key)
                            deduplicated_list.append(ds)
                            
                    kaggle_deduplicated = sum(1 for d in deduplicated_list if d.get("source") == "kaggle")
                    hf_deduplicated = sum(1 for d in deduplicated_list if d.get("source") == "huggingface")
                    total_deduplicated = len(deduplicated_list)
                    
                    kaggle_ranked = sum(1 for d in candidates if d.get("source") == "kaggle")
                    hf_ranked = sum(1 for d in candidates if d.get("source") == "huggingface")
                    total_ranked = len(candidates)
                    
                    kaggle_eval = sum(1 for d in top_20 if d.get("source") == "kaggle")
                    hf_eval = sum(1 for d in top_20 if d.get("source") == "huggingface")
                    total_eval = len(top_20)
                    
                    kaggle_top10 = sum(1 for d in top_datasets if d.get("source") == "kaggle")
                    hf_top10 = sum(1 for d in top_datasets if d.get("source") == "huggingface")
                    total_top10 = len(top_datasets)

                    # Print to console
                    print("\n" + "="*54)
                    print(" TEMPORARY DEBUG: Hugging Face Pipeline Diagnosis")
                    print("="*54)
                    print(f"{'Stage':<24} {'Kaggle':<8} {'Hugging Face':<14} {'Total':<5}")
                    print(f"------------------------------------------------------")
                    print(f"{'Retrieved':<24} {kaggle_retrieved:<8} {hf_retrieved:<14} {total_retrieved:<5}")
                    print(f"{'Merged':<24} {kaggle_merged:<8} {hf_merged:<14} {total_merged:<5}")
                    print(f"{'Deduplicated':<24} {kaggle_deduplicated:<8} {hf_deduplicated:<14} {total_deduplicated:<5}")
                    print(f"{'Embedding Ranked':<24} {kaggle_ranked:<8} {hf_ranked:<14} {total_ranked:<5}")
                    print(f"{'Gemini Evaluated':<24} {kaggle_eval:<8} {hf_eval:<14} {total_eval:<5}")
                    print(f"{'Final Top 10':<24} {kaggle_top10:<8} {hf_top10:<14} {total_top10:<5}")
                    print("="*54 + "\n")

                    # Update progress container to show complete
                    with progress_container.container(border=True):
                        st.markdown("### ⚡ Discovery Pipeline Progress")
                        st.success("🎉 Discovery and evaluation complete!")
                        st.progress(1.0)
                    
                    # Render diagnosis table in UI
                    with st.expander("🔍 Hugging Face Pipeline Diagnosis (Debug Table)", expanded=True):
                        st.markdown("### 🔍 Temporary Pipeline Diagnosis Counts")
                        st.write("This table shows the counts of Kaggle and Hugging Face datasets at each step of the pipeline.")
                        debug_df = {
                            "Stage": ["Retrieved", "Merged", "Deduplicated", "Embedding Ranked (Capped)", "Gemini Evaluated (Top 20)", "Final Top 10"],
                            "Kaggle": [kaggle_retrieved, kaggle_merged, kaggle_deduplicated, kaggle_ranked, kaggle_eval, kaggle_top10],
                            "Hugging Face": [hf_retrieved, hf_merged, hf_deduplicated, hf_ranked, hf_eval, hf_top10],
                            "Total": [total_retrieved, total_merged, total_deduplicated, total_ranked, total_eval, total_top10]
                        }
                        st.table(debug_df)

                    # --- Render Results ---
                    
                    # --- Display Intent and Search Plan side-by-side ---
                    if search_plan:
                        col_intent, col_plan = st.columns(2)
                        with col_intent:
                            with st.container(border=True):
                                st.markdown("### 🎯 Extracted Intent Analysis")
                                st.markdown(f"**Primary Topic:** `{intent.get('primary_topic', 'N/A')}`")
                                st.markdown(f"**Location Filter:** `{intent.get('location', 'Global')}`")
                                st.markdown(f"**Secondary Concepts:** `{', '.join(intent.get('secondary_concepts', [])) or 'None'}`")
                        with col_plan:
                            with st.container(border=True):
                                st.markdown("### 📋 Generated Search Plan")
                                sp1, sp2 = st.columns([1, 1])
                                with sp1:
                                    st.markdown(f"**Topic:** `{search_plan.get('primary_topic', 'N/A')}`")
                                    st.markdown(f"**Domain:** `{search_plan.get('domain', 'N/A')}`")
                                    st.markdown(f"**Must-have:** `{', '.join(search_plan.get('must_have_concepts', []))}`")
                                    st.markdown(f"**Optional:** `{', '.join(search_plan.get('optional_concepts', []))}`")
                                    if search_plan.get('excluded_terms'):
                                        st.markdown(f"**Excluded:** `{', '.join(search_plan.get('excluded_terms', []))}`")
                                with sp2:
                                    st.markdown("**Search Queries:**")
                                    for query_str in search_plan.get("search_queries", []):
                                        st.markdown(f"- `{query_str}`")
                    else:
                        with st.container(border=True):
                            st.markdown("### 🎯 Extracted Intent Analysis")
                            ic1, ic2, ic3 = st.columns(3)
                            ic1.markdown(f"**Primary Topic:** `{intent.get('primary_topic', 'N/A')}`")
                            ic2.markdown(f"**Location Filter:** `{intent.get('location', 'Global')}`")
                            ic3.markdown(f"**Secondary Concepts:** `{', '.join(intent.get('secondary_concepts', [])) or 'None'}`")
                    
                    # --- Display Search Quality Assessment ---
                    if quality_assessment:
                        verdict = quality_assessment.get('verdict', 'N/A')
                        if verdict.lower() in ["high", "excellent", "good"]:
                            verdict_badge = f"🟢 **{verdict}**"
                            card_title = "🏆 Overall Search Quality Assessment (High Relevance)"
                        elif verdict.lower() in ["medium", "moderate", "fair"]:
                            verdict_badge = f"🟡 **{verdict}**"
                            card_title = "⚠️ Overall Search Quality Assessment (Moderate Relevance)"
                        else:
                            verdict_badge = f"🔴 **{verdict}**"
                            card_title = "🚨 Overall Search Quality Assessment (Low Relevance)"
                            
                        with st.container(border=True):
                            st.markdown(f"### {card_title}")
                            qc1, qc2 = st.columns([1, 3])
                            with qc1:
                                st.metric("Quality Score", f"{quality_assessment.get('quality_score', 0.0):.2f}")
                                st.markdown(f"**Verdict:** {verdict_badge}")
                            with qc2:
                                st.markdown("**Key Observations:**")
                                for reason_bullet in quality_assessment.get("reasons", []):
                                    st.markdown(f"- {reason_bullet}")
                    
                    # --- Semantic search info status ---
                    semantic_active = any(ds.get("vector_similarity") is not None for ds in top_datasets)
                    if semantic_active:
                        st.info("🧠 **Semantic Search Enabled:** Leveraging Gemini embeddings to rank datasets by semantic meaning in addition to keywords.")
                    else:
                        st.warning("⚠️ **Fallback Rule-Based Search Active:** Gemini embeddings are unavailable. Using keyword match and metadata metrics only.")
                        
                    # Check for weak relevance warning
                    best_relevance = max(ds["relevance_score"] for ds in top_datasets) if top_datasets else 0.0
                    if best_relevance < 0.3:
                        st.warning(
                            "⚠️ **Weak Relevance Warning:** None of the top results strongly match your keywords. "
                            "Consider rephrasing your goal or using more specific terms."
                        )
                        
                    st.success(f"Successfully evaluated {len(candidates)} candidate datasets. Here are the top recommendations:")
                    
                    # --- Render Top recommendations in cards ---
                    for rank, ds in enumerate(top_datasets, start=1):
                        with st.container(border=True):
                            # Header Row
                            header_col1, header_col2 = st.columns([3, 1])
                            with header_col1:
                                st.markdown(f"### **#{rank} {ds['title']}**")
                            with header_col2:
                                if ds.get("source") == "kaggle":
                                    st.markdown("🔹 **Kaggle Dataset**")
                                else:
                                    st.markdown("🤗 **Hugging Face Dataset**")
                                st.link_button("🌐 Open Dataset", ds['url'], use_container_width=True)
                                
                            # Metrics Grid
                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.metric("Evaluation Score", f"{ds.get('evaluation_score', 0.0):.2f}")
                            m2.metric("Confidence Score", f"{ds.get('confidence', 0.0):.2f}")
                            m3.metric("Verdict", ds.get('verdict', 'N/A'))
                            m4.metric("Downloads", f"{ds.get('download_count', 0):,}")
                            
                            usability = ds.get('usability_rating')
                            sem_sim = ds.get('vector_similarity')
                            if usability is not None and usability != 'N/A' and usability != 0.0:
                                m5.metric("Usability", f"{usability or 'N/A'}")
                            elif sem_sim is not None and sem_sim != 'N/A':
                                m5.metric("Semantic Sim", f"{sem_sim:.2f}" if isinstance(sem_sim, float) else f"{sem_sim}")
                            else:
                                m5.metric("Usability / Sim", "N/A")
                                
                            # Reasons and Details
                            st.markdown(f"**📋 Evaluation Reason:** {ds.get('evaluation_reason', 'N/A')}")
                            
                            # Baseline reasons and limitations using helper functions
                            reason_text = _build_reason(ds, intent['keywords'])
                            limitation_text = _build_limitation(ds)
                            
                            reasons_col, lims_col = st.columns(2)
                            with reasons_col:
                                st.markdown(f"**👍 Why Recommended (Baseline):**\n- {reason_text}")
                            with lims_col:
                                st.markdown(f"**⚠️ Limitations:**\n- {limitation_text}")
                                
                            # Intent-aware adjustments
                            adjustments = []
                            if ds.get("topic_relevance", 0.0) == 1.0:
                                adjustments.append("🟢 **Primary Topic Boost (+0.15)** (Title/Subtitle match)")
                            elif ds.get("topic_relevance", 0.0) == 0.5:
                                adjustments.append("🟡 **Primary Topic Boost (+0.05)** (Description/Tag match)")
                            if ds.get("location_match", 0.0) > 0.0:
                                adjustments.append("🌐 **Location Match Boost (+0.05)**")
                            if ds.get("generic_penalty_applied"):
                                adjustments.append("🔴 **Generic Keyword Penalty (-65% composite score)**")
                                
                            if adjustments:
                                st.markdown("**Adjustments applied:** " + " | ".join(adjustments))
                                
                            # Footer caption
                            st.caption(
                                f"Dataset Reference: `{ds['ref']}` | "
                                f"Composite Score: {ds['composite_score']:.2f} | "
                                f"Semantic Similarity: {ds.get('vector_similarity', 'N/A')} | "
                                f"Last Updated: {ds.get('last_updated', 'N/A')}"
                            )
                            
                    # Expose the full CLI format report in an expander for debugging/reference
                    with st.expander("View Full Text Report"):
                        st.text(report)
                        
        except Exception as e:
            progress_container.empty()
            st.error(f"An unexpected error occurred during execution: {e}")
