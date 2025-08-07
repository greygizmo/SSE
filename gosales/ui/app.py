import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from gosales.utils.paths import OUTPUTS_DIR
except ImportError:
    # Fallback if gosales module isn't found
    OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

st.set_page_config(layout="wide")

st.title("GoSales - ICP & Whitespace Engine")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "ETL & Coverage",
    "Features & Explainability",
    "Scores & Whitespace",
])

if page == "Overview":
    st.markdown("""
    This portal documents each step of the pipeline:
    - ETL: raw CSVs → star schema; industry enrichment joins
    - Feature Engineering: time-aware features, industry dummies, interactions
    - Modeling: time-split training; AUC; SHAP explanations
    - Outputs: scores, whitespace, coverage reports
    """)

elif page == "ETL & Coverage":
    st.subheader("Industry Coverage")
    cols = st.columns(3)
    try:
        summary = pd.read_csv(OUTPUTS_DIR / 'industry_coverage_summary.csv')
        total = int(summary.loc[summary['metric']=='total_customers','value'])
        with_ind = int(summary.loc[summary['metric']=='with_industry','value'])
        pct = float(summary.loc[summary['metric']=='coverage_pct','value'])
        cols[0].metric("Total Customers", f"{total:,}")
        cols[1].metric("With Industry", f"{with_ind:,}")
        cols[2].metric("Coverage %", f"{pct:.2f}%")
    except Exception:
        st.warning("Coverage summary not available")

    c1, c2 = st.columns(2)
    try:
        top_ind = pd.read_csv(OUTPUTS_DIR / 'industry_top50.csv')
        c1.subheader("Top Industries")
        c1.dataframe(top_ind, use_container_width=True)
    except Exception:
        c1.warning("Top industries not available")
    try:
        top_sub = pd.read_csv(OUTPUTS_DIR / 'sub_industry_top50.csv')
        c2.subheader("Top Sub-Industries")
        c2.dataframe(top_sub, use_container_width=True)
    except Exception:
        c2.warning("Top sub-industries not available")

    try:
        fuzz = pd.read_csv(OUTPUTS_DIR / 'industry_fuzzy_matches.csv')
        st.subheader("Fuzzy Matches (Audit)")
        st.dataframe(fuzz.head(500), use_container_width=True)
    except Exception:
        st.info("No fuzzy matches captured or file not present")

elif page == "Features & Explainability":
    st.subheader("SHAP Values (per-customer)")
    shap_file = OUTPUTS_DIR / 'shap_values_solidworks.csv'
    if shap_file.exists():
        shap_df = pd.read_csv(shap_file)
        st.write("Sample of SHAP values (top 100 rows):")
        st.dataframe(shap_df.head(100), use_container_width=True)
        st.download_button("Download SHAP CSV", data=shap_df.to_csv(index=False), file_name='shap_values.csv')
    else:
        st.warning("SHAP file not found. Train the model to generate explainability outputs.")

elif page == "Scores & Whitespace":
    st.subheader("ICP Scores")
    try:
        icp_scores = pd.read_csv(OUTPUTS_DIR / "icp_scores.csv")
        st.dataframe(icp_scores, use_container_width=True)
    except FileNotFoundError:
        st.warning("ICP scores not available.")

    st.subheader("Whitespace Opportunities")
    try:
        whitespace = pd.read_csv(OUTPUTS_DIR / "whitespace.csv")
        st.dataframe(whitespace, use_container_width=True)
    except FileNotFoundError:
        st.warning("Whitespace opportunities not available.")
