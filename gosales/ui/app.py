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
        total = int(summary.loc[summary['metric']=='total_customers','value'].iloc[0])
        with_ind = int(summary.loc[summary['metric']=='with_industry','value'].iloc[0])
        pct = float(summary.loc[summary['metric']=='coverage_pct','value'].iloc[0])
        cols[0].metric("Total Customers", f"{total:,}")
        cols[1].metric("With Industry", f"{with_ind:,}")
        cols[2].metric("Coverage %", f"{pct:.2f}%")
        st.caption("Shows overall coverage of industry metadata after enrichment.")
    except Exception:
        st.warning("Coverage summary not available")

    c1, c2 = st.columns(2)
    try:
        top_ind = pd.read_csv(OUTPUTS_DIR / 'industry_top50.csv')
        c1.subheader("Top Industries")
        c1.caption("Count of customers by industry (top 50). Use to validate distribution and join quality.")
        c1.dataframe(top_ind, use_container_width=True)
    except Exception:
        c1.warning("Top industries not available")
    try:
        top_sub = pd.read_csv(OUTPUTS_DIR / 'sub_industry_top50.csv')
        c2.subheader("Top Sub-Industries")
        c2.caption("Count of customers by sub-industry (top 50). Highlights granularity and long tail.")
        c2.dataframe(top_sub, use_container_width=True)
    except Exception:
        c2.warning("Top sub-industries not available")

    try:
        fuzz = pd.read_csv(OUTPUTS_DIR / 'industry_fuzzy_matches.csv')
        st.subheader("Fuzzy Matches (Audit)")
        st.caption("High-confidence fuzzy matches (score ≥ 95). Spot-check to ensure correctness of enrichment.")
        st.dataframe(fuzz.head(500), use_container_width=True)
    except Exception:
        st.info("No fuzzy matches captured or file not present")

elif page == "Features & Explainability":
    st.subheader("SHAP Values (per-customer)")
    with st.expander("How to interpret SHAP"):
        st.markdown(
            "- Positive SHAP: pushes a customer's ICP score up (more likely to buy).\n"
            "- Negative SHAP: pushes the score down (less likely).\n"
            "- Magnitude: size of impact on the model's decision for that customer.\n"
            "- Units: model margin; compare relatively across features for a customer."
        )
    shap_file = OUTPUTS_DIR / 'shap_values_solidworks.csv'
    if shap_file.exists():
        shap_df = pd.read_csv(shap_file)
        st.write("Sample of SHAP values (top 100 rows):")
        st.dataframe(shap_df.head(100), use_container_width=True)
        st.download_button("Download SHAP CSV", data=shap_df.to_csv(index=False), file_name='shap_values.csv')
        # Feature glossary
        st.subheader("Feature Glossary")
        def _describe_feature(name: str) -> str:
            base = {
                'total_transactions_all_time': 'Total number of historical transactions.',
                'transactions_last_2y': 'Transactions during 2023–2024.',
                'total_gp_all_time': 'Total gross profit across all time.',
                'total_gp_last_2y': 'Gross profit during 2023–2024.',
                'avg_transaction_gp': 'Average gross profit per transaction.',
                'services_transaction_count': 'Number of Services division transactions.',
                'simulation_transaction_count': 'Number of Simulation division transactions.',
                'hardware_transaction_count': 'Number of Hardware division transactions.',
                'total_services_gp': 'Gross profit from Services transactions.',
                'total_training_gp': 'Gross profit from Training SKU.',
                'gp_2024': 'Gross profit in year 2024.',
                'gp_2023': 'Gross profit in year 2023.',
                'product_diversity_score': 'Count of distinct divisions purchased from.',
                'growth_ratio_24_over_23': 'Growth proxy = gp_2024 / (gp_2023 + 1).',
            }
            if name in base:
                return base[name]
            if name.startswith('is_sub_'):
                return 'Sub-industry indicator (1 if customer belongs to this sub-industry).'
            if name.startswith('is_'):
                return 'Industry indicator (1 if customer belongs to this industry).'
            if name.endswith('_x_services'):
                return 'Interaction with normalized Services GP (industry × services engagement).'
            if name.endswith('_x_avg_gp'):
                return 'Interaction with normalized average transaction GP (industry × spend intensity).'
            if name.endswith('_x_diversity'):
                return 'Interaction with normalized product diversity (industry × breadth of engagement).'
            if name.endswith('_x_growth'):
                return 'Interaction with growth ratio (industry × growth dynamics).'
            return 'Model feature (behavioral or categorical).'

        feature_cols = [c for c in shap_df.columns if c != 'customer_id']
        glossary = pd.DataFrame({'feature': feature_cols, 'description': [_describe_feature(c) for c in feature_cols]}).sort_values('feature')
        st.dataframe(glossary, use_container_width=True, height=300)
        # Model card
        st.subheader("Model Card")
        try:
            mc = pd.read_csv(OUTPUTS_DIR / 'model_card_solidworks.csv')
            st.dataframe(mc, use_container_width=True)
        except Exception:
            st.info("Model card not available yet.")
    else:
        st.warning("SHAP file not found. Train the model to generate explainability outputs.")

elif page == "Scores & Whitespace":
    st.subheader("ICP Scores")
    st.caption("Predicted likelihood that a customer will purchase in the Solidworks division within the prediction window.")
    try:
        icp_scores = pd.read_csv(OUTPUTS_DIR / "icp_scores.csv")
        st.dataframe(icp_scores, use_container_width=True)
    except FileNotFoundError:
        st.warning("ICP scores not available.")

    st.subheader("Whitespace Opportunities")
    st.caption("Products not yet purchased by a customer where model and behavioral signals suggest potential demand.")
    try:
        whitespace = pd.read_csv(OUTPUTS_DIR / "whitespace.csv")
        st.dataframe(whitespace, use_container_width=True)
    except FileNotFoundError:
        st.warning("Whitespace opportunities not available.")
