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
from agents.dataset_retriever import DatasetRetriever
from agents.dataset_evaluator import DatasetEvaluator
from skills.generate_recommendation import _build_reason, _build_limitation

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
if search_clicked:
    if not (username_exists and key_exists):
        st.error("Kaggle credentials missing. Please set KAGGLE_USERNAME and KAGGLE_KEY in your .env file.")
    elif not goal_input.strip():
        st.warning("Please enter an analysis goal before searching.")
    else:
        # Use st.status to display the steps of the background process clearly
        with st.status("DataScout Agent is working...", expanded=True) as status:
            try:
                # Step 1: Extract Intent & Keywords
                status.write("🧠 Analyzing search intent...")
                analyzer = IntentAnalyzer()
                intent = analyzer.analyze(goal_input)
                
                # Show keywords in status window
                status.write(f"• **Extracted keywords:** {', '.join(intent['keywords'])}")
                status.write(f"• **Detected domain:** {intent['domain'] or 'General'}")
                
                # Step 2: Retrieve Candidates from Kaggle
                status.write("📡 Retrieving datasets from Kaggle API...")
                retriever = DatasetRetriever()
                candidates = retriever.retrieve(intent)
                
                if not candidates:
                    status.update(label="No datasets found.", state="complete")
                    st.error("No relevant Kaggle/Hugging Face datasets were found for this query.")
                    
                    st.markdown("### Suggestions:")
                    st.markdown("- Try broader or more general keywords.")
                    st.markdown("- Remove overly specific terms.")
                    st.markdown("- Use common names or abbreviations.")
                    if intent['keywords']:
                        st.markdown(f'- Example: try just **"{intent["keywords"][0]}"** instead of the full phrase.')
                else:
                    # Step 3: Evaluate and Rank Candidates
                    status.write("📊 Scoring and ranking candidate datasets...")
                    evaluator = DatasetEvaluator()
                    top_datasets, report = evaluator.evaluate(candidates, intent, top_k=5)
                    
                    # Update status indicator to finished
                    status.update(label="Discovery and evaluation complete!", state="complete", expanded=False)
                    
                    # --- Display Extracted Intent ---
                    st.markdown("### 🎯 Extracted Intent Analysis")
                    ic1, ic2, ic3 = st.columns(3)
                    ic1.markdown(f"**Primary Topic:** `{intent.get('primary_topic', 'N/A')}`")
                    ic2.markdown(f"**Location Filter:** `{intent.get('location', 'Global')}`")
                    ic3.markdown(f"**Secondary Concepts:** `{', '.join(intent.get('secondary_concepts', [])) or 'None'}`")
                    st.markdown("---")

                    # --- Display Results ---
                    semantic_active = any(ds.get("vector_similarity") is not None for ds in top_datasets)
                    if semantic_active:
                        st.info("🧠 **Semantic Search Enabled:** Leveraging Gemini embeddings to rank datasets by semantic meaning in addition to keywords.")
                    else:
                        st.warning("⚠️ **Fallback Rule-Based Search Active:** Gemini embeddings are unavailable. Using keyword match and metadata metrics only.")

                    st.success(f"Successfully evaluated {len(candidates)} candidate datasets. Here are the top recommendations:")
                    
                    # Check for weak relevance warning across all retrieved datasets
                    best_relevance = max(ds["relevance_score"] for ds in top_datasets) if top_datasets else 0.0
                    if best_relevance < 0.3:
                        st.warning(
                            "⚠️ **Weak Relevance Warning:** None of the top results strongly match your keywords. "
                            "Consider rephrasing your goal or using more specific terms."
                        )
                    
                    # Loop over and render top recommendations
                    for rank, ds in enumerate(top_datasets, start=1):
                        with st.container():
                            # Clickable title linking to the dataset page
                            source_badge = "📊 **Kaggle**" if ds.get("source") == "kaggle" else "🤗 **Hugging Face**"
                            st.markdown(f"#### **#{rank} [{ds['title']}]({ds['url']})** &nbsp;&nbsp;&nbsp;&nbsp; {source_badge}")
                            
                            # Grid of key dataset metrics (5 columns for semantic similarity)
                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.metric("Composite Score", f"{ds['composite_score']:.2f}")
                            
                            # Display semantic similarity or a helper text if fallback was active
                            v_sim = ds.get("vector_similarity")
                            if v_sim is not None:
                                m2.metric("Semantic Similarity", f"{v_sim:.2f}")
                            else:
                                m2.metric("Semantic Similarity", "N/A", help="embeddings fell back to rule-based")
                                
                            m3.metric("Downloads", f"{ds['download_count']:,}")
                            m4.metric("Usability Rating", f"{ds['usability_rating'] or 'N/A'}")
                            
                            # Safely extract the date part from the timestamp
                            raw_date = ds.get('last_updated', '')
                            date_str = raw_date.split()[0] if raw_date and raw_date.split() else "N/A"
                            m5.metric("Last Updated", date_str)
                            
                            # Show intent-aware adjustments
                            adjustments = []
                            if ds.get("topic_relevance", 0.0) == 1.0:
                                adjustments.append("🟢 **Primary Topic Boost (+0.15)** (Title/Subtitle match)")
                            elif ds.get("topic_relevance", 0.0) == 0.5:
                                adjustments.append("🟡 **Primary Topic Boost (+0.05)** (Description/Tag match)")
                            if ds.get("location_match", 0.0) > 0.0:
                                adjustments.append("🌐 **Location Match Boost (+0.05)**")
                            if ds.get("generic_penalty_applied"):
                                adjustments.append("🔴 **Generic Keyword Penalty (-65% composite score)** (Matches generic keywords without primary topic)")
                            
                            if adjustments:
                                st.write(" | ".join(adjustments))

                            # Generate explanations using the helper functions
                            reason = _build_reason(ds, intent['keywords'])
                            limitation = _build_limitation(ds)
                            
                            st.write(f"**👍 Reason:** {reason}")
                            st.write(f"**⚠️ Limitation:** {limitation}")
                            st.caption(f"Dataset Reference: `{ds['ref']}`")
                            st.markdown("---")
                    
                    # Expose the full CLI format report in an expander for debugging/reference
                    with st.expander("View Full Text Report"):
                        st.text(report)
                        
            except Exception as e:
                status.update(label="An error occurred.", state="error")
                st.error(f"An unexpected error occurred during execution: {e}")
