import json
from pathlib import Path

import pandas as pd
import streamlit as st

from gosales.utils.paths import OUTPUTS_DIR
from gosales.ui.utils import discover_validation_runs, compute_validation_badges, load_thresholds, load_alerts


st.set_page_config(page_title="GoSales Engine", layout="wide")
st.title("GoSales Engine – Artifact Explorer")

# Simple cache helpers
@st.cache_data(show_spinner=False)
def _read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8')
    except Exception:
        return ""

# Sidebar controls
with st.sidebar:
    colr1, colr2 = st.columns([3,1])
    colr1.write(":memo: Navigation")
    if colr2.button("Refresh", help="Clear cached artifacts and reload"):
        st.cache_data.clear()
    tab = st.radio("Page", ["Metrics", "Explainability", "Whitespace", "Validation", "Runs"], index=0)

def list_validation_runs():
    base = OUTPUTS_DIR / 'validation'
    if not base.exists():
        return []
    rows = []
    for div_dir in base.iterdir():
        if not div_dir.is_dir():
            continue
        for cut_dir in div_dir.iterdir():
            if cut_dir.is_dir():
                rows.append((div_dir.name, cut_dir.name, cut_dir))
    return rows

if tab == "Metrics":
    st.header("Metrics & Training Artifacts")
    # Simple viewer for Phase 3 metrics
    metrics_files = sorted(OUTPUTS_DIR.glob("metrics_*.json"))
    if not metrics_files:
        st.info("No training metrics found")
    else:
        sel = st.selectbox("Pick metrics file", metrics_files)
        st.code((sel.read_text(encoding='utf-8')))  

elif tab == "Explainability":
    st.header("Explainability (Phase 3)")
    shap_files = sorted(OUTPUTS_DIR.glob("shap_global_*.csv"))
    if shap_files:
        sel = st.selectbox("SHAP global file", shap_files)
        st.dataframe(pd.read_csv(sel).head(50))
    coef_files = sorted(OUTPUTS_DIR.glob("coef_*.csv"))
    if coef_files:
        sel2 = st.selectbox("LR coefficients file", coef_files)
        st.dataframe(pd.read_csv(sel2))
    if not shap_files and not coef_files:
        st.info("No explainability artifacts found.")

elif tab == "Whitespace":
    st.header("Whitespace (Phase 4)")
    ws_files = sorted(OUTPUTS_DIR.glob("whitespace_*.csv"))
    if ws_files:
        sel = st.selectbox("Whitespace ranked CSV", ws_files)
        st.dataframe(pd.read_csv(sel).head(100))
    else:
        st.info("No whitespace files found.")

elif tab == "Validation":
    st.header("Forward Validation (Phase 5)")
    runs = discover_validation_runs()
    if not runs:
        st.info("No validation runs found.")
    else:
        labels = [f"{div} @ {cut}" for div, cut, _ in runs]
        sel = st.selectbox("Pick run", options=list(range(len(runs))), format_func=lambda i: labels[i])
        _, _, path = runs[sel]
        thr = load_thresholds()
        # Badges
        st.subheader("Quality Badges")
        badges = compute_validation_badges(path, thresholds=thr)
        b1, b2, b3 = st.columns(3)
        def _badge(col, title, item):
            status = item.get('status', 'unknown')
            value = item.get('value', None)
            threshold = item.get('threshold', None)
            color = '#60c460' if status == 'ok' else ('#e06666' if status == 'alert' else '#bdbdbd')
            body = f"{value:.3f}" if isinstance(value, (int, float)) else "—"
            thr_txt = f"<span style='font-size:12px;color:#666;'>thr {threshold:.3f}</span>" if isinstance(threshold, (int, float)) else ""
            col.markdown(f"""
                <div style='border-left:6px solid {color}; padding:8px; border-radius:4px; background:#f7f7f7;'>
                    <div style='font-weight:600;'>{title}</div>
                    <div style='font-size:20px'>{body}</div>
                    {thr_txt}
                </div>
            """, unsafe_allow_html=True)
        _badge(b1, 'Calibration MAE', badges['cal_mae'])
        _badge(b2, 'PSI(EV vs GP)', badges['psi_ev_vs_gp'])
        _badge(b3, 'KS(train vs holdout)', badges['ks_phat_train_holdout'])

        # Alerts
        alerts = load_alerts(path)
        if alerts:
            with st.expander("Alerts"):
                for a in alerts:
                    st.warning(f"{a.get('type')}: value={a.get('value')} threshold={a.get('threshold')}")
        col1, col2 = st.columns(2)
        # Metrics
        metrics_path = path / 'metrics.json'
        if metrics_path.exists():
            st.subheader("Metrics")
            st.code(metrics_path.read_text(encoding='utf-8'))
        # Drift
        drift_path = path / 'drift.json'
        if drift_path.exists():
            st.subheader("Drift")
            st.code(drift_path.read_text(encoding='utf-8'))
        # Scenarios
        scen_path = path / 'topk_scenarios_sorted.csv'
        if scen_path.exists():
            st.subheader("Scenarios (sorted)")
            st.dataframe(pd.read_csv(scen_path))
        # Segment performance
        seg_path = path / 'segment_performance.csv'
        if seg_path.exists():
            st.subheader("Segment performance")
            st.dataframe(pd.read_csv(seg_path))
        # Downloads
        st.subheader("Downloads")
        for fname in ["validation_frame.parquet","gains.csv","calibration.csv","topk_scenarios.csv","topk_scenarios_sorted.csv","segment_performance.csv","metrics.json","drift.json"]:
            fpath = path / fname
            if fpath.exists():
                st.download_button(label=f"Download {fname}", data=fpath.read_bytes(), file_name=fname)

elif tab == "Runs":
    st.header("Runs (Registry)")
    reg_path = OUTPUTS_DIR / 'runs' / 'runs.jsonl'
    if not reg_path.exists():
        st.info("No runs registry found at outputs/runs/runs.jsonl")
    else:
        rows = _read_jsonl(reg_path)
        if not rows:
            st.info("Runs registry is empty.")
        else:
            df = pd.DataFrame(rows)
            df = df.sort_values('run_id', ascending=False)
            st.dataframe(df, use_container_width=True, height=300)
            idx = st.number_input("Select row index", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
            sel = df.iloc[int(idx)]
            st.subheader(f"Run {sel.get('run_id','?')} — {sel.get('phase','?')} [{sel.get('status','?')}]")
            run_dir = Path(sel.get('artifacts_path', OUTPUTS_DIR / 'runs' / str(sel.get('run_id',''))))
            man = run_dir / 'manifest.json'
            cfg = run_dir / 'config_resolved.yaml'
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Manifest (planned/emitted artifacts)")
                if man.exists():
                    st.code(_read_text(man))
                else:
                    st.info("manifest.json not found")
            with c2:
                st.caption("Resolved Config Snapshot")
                if cfg.exists():
                    st.code(_read_text(cfg))
                else:
                    st.info("config_resolved.yaml not found")
import streamlit as st
import pandas as pd
import json
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

st.set_page_config(
    layout="wide",
    page_title="GoEngineer ICP & Whitespace",
    page_icon=str((Path(__file__).parent.parent / 'docs' / 'GoEngineer Bug Option 1.png').resolve())
)

with st.container():
    cols = st.columns([1,6])
    try:
        logo_path = Path(__file__).parent.parent / 'docs' / 'GoEngineer-full-logo-horizontal-black.png'
        cols[0].image(str(logo_path.resolve()), use_container_width=True)
    except Exception:
        pass
    cols[1].markdown("<h2 style='margin-top:0;'>ICP & Whitespace Engine</h2>", unsafe_allow_html=True)

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "ETL & Coverage",
    "Features & Explainability",
    "Modeling & Validation",
    "Scores & Whitespace",
])

if page == "Overview":
    st.markdown("""
    This portal documents each step of the GoSales ICP & Whitespace pipeline with full transparency.
    
    1) ETL (Extract → Transform → Load)
    - Ingests raw wide `Sales_Log.csv` and converts to a tidy `fact_transactions` table (one SKU per row), plus `dim_customer` with industry enrichment.
    - Hybrid enrichment strategy: exact name, numeric prefix, then (optional) fuzzy matching audit.
    
    2) Leakage-safe Feature Engineering
    - A `cutoff_date` strictly separates historical feature data (≤ cutoff) from future labels (> cutoff).
    - Feature families: RFM, windowed aggregates (3/6/12/24m), temporal dynamics (slopes/volatility), cadence (tenure & gaps), seasonality, division mix, SKU micro-signals, branch/rep shares, basket lift, industry plus interactions.
    
    3) Modeling
    - Logistic Regression baseline and LightGBM for accuracy. Class imbalance handled; calibration curves exported.
    
    4) Validation (Future Data / Holdout)
    - 2025 YTD holdout with labels derived directly from `Division == 'Solidworks'` (Jan–Jun 2025).
    - Gains table and summary metrics saved.
    
    5) Scoring & Whitespace
    - Per-customer ICP probability and opportunities for divisions not yet purchased.
    """)
    with st.expander("Algorithm explainer (data-leakage guardrails)"):
        st.markdown("""
        - Features only use transactions on or before the `cutoff_date`.
        - Labels are defined by purchases occurring strictly after the cutoff and within the future window.
        - We do not include any future-derived features (e.g., post-cutoff totals or explicit targets) to avoid leakage.
        """)

elif page == "ETL & Coverage":
    st.subheader("Industry Coverage")
    st.caption("Brand palette applied across charts to align with GoEngineer style guide.")
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
        import plotly.express as px
        fig = px.bar(top_ind, x='industry', y='count', color_discrete_sequence=['#BAD532'])
        fig.update_layout(xaxis_title='', yaxis_title='Customers', height=420, margin=dict(l=10,r=10,b=10,t=10))
        c1.plotly_chart(fig, use_container_width=True)
    except Exception:
        c1.warning("Top industries not available")
    try:
        top_sub = pd.read_csv(OUTPUTS_DIR / 'sub_industry_top50.csv')
        c2.subheader("Top Sub-Industries")
        c2.caption("Count of customers by sub-industry (top 50). Highlights granularity and long tail.")
        import plotly.express as px
        fig2 = px.bar(top_sub, x='industry_sub', y='count', color_discrete_sequence=['#336D91'])
        fig2.update_layout(xaxis_title='', yaxis_title='Customers', height=420, margin=dict(l=10,r=10,b=10,t=10))
        c2.plotly_chart(fig2, use_container_width=True)
    except Exception:
        c2.warning("Top sub-industries not available")

    try:
        fuzz = pd.read_csv(OUTPUTS_DIR / 'industry_fuzzy_matches.csv')
        st.subheader("Fuzzy Matches (Audit)")
        st.caption("High-confidence fuzzy matches (score ≥ 95). Spot-check to ensure correctness of enrichment.")
        st.dataframe(fuzz.head(500), use_container_width=True)
    except Exception:
        st.info("No fuzzy matches captured or file not present")

    st.subheader("Data Contracts & Row Counts")
    c3, c4 = st.columns(2)
    try:
        rc = pd.read_csv(OUTPUTS_DIR / 'contracts' / 'row_counts.csv')
        c3.caption("Table row counts after ETL")
        c3.dataframe(rc, use_container_width=True)
    except Exception:
        c3.info("Row counts not available")
    try:
        viol = pd.read_csv(OUTPUTS_DIR / 'contracts' / 'violations.csv')
        c4.caption("Contract violations (missing columns, nulls in PKs, duplicates)")
        c4.dataframe(viol.head(200), use_container_width=True)
    except Exception:
        c4.info("No violations file found (or empty)")

    st.subheader("Label Audit (prevalence & window)")
    try:
        la = pd.read_csv(OUTPUTS_DIR / 'labels_summary.csv')
        st.dataframe(la, use_container_width=True)
        st.caption("Window start/end, total customers, positives and prevalence for the target division.")
    except Exception:
        st.info("Label summary not available yet.")

elif page == "Features & Explainability":
    st.subheader("SHAP Values (per-customer)")
    with st.expander("How to interpret SHAP"):
        st.markdown(
            """
            - SHAP explains a single prediction by attributing impact to each feature.\
              For binary classification, the model computes a score on the log-odds scale.\
              The sum of all SHAP values + a baseline equals the model's raw score for that customer.\
            - **Sign**: Positive SHAP increases the ICP likelihood; negative decreases it.\
            - **Magnitude**: Absolute value is the strength of the effect for that customer.\
            - **Categorical flags** (e.g., `is_industry...`, `is_sub_...`): 1 means the customer is in that group.\
              A positive value indicates membership in that group is associated with higher conversion likelihood,\
              conditioned on the other features.\
            - **Interactions** (e.g., `_x_services`, `_x_avg_gp`, `_x_diversity`, `_x_growth`): quantify how the base\
              effect of an industry/sub-industry changes as engagement or growth increases. Positive means the combo\
              is especially favorable.\
            - SHAP values are additive explanations; compare features for the same customer rather than across very\
              different customers with very different profiles.\
            """
        )
    shap_file = OUTPUTS_DIR / 'shap_values_solidworks.csv'
    if shap_file.exists():
        shap_df = pd.read_csv(shap_file)
        st.write("Sample of SHAP values (top 100 rows):")
        st.dataframe(shap_df.head(100), use_container_width=True)
        st.download_button("Download SHAP CSV", data=shap_df.to_csv(index=False), file_name='shap_values.csv')

        # Merge customer name and ICP score; then show Top 100 by score
        try:
            icp_df = pd.read_csv(OUTPUTS_DIR / 'icp_scores.csv', usecols=['customer_id','customer_name','icp_score'])
            # Align dtypes to ensure the join works
            if 'customer_id' in shap_df.columns:
                shap_df['customer_id'] = pd.to_numeric(shap_df['customer_id'], errors='coerce')
            icp_df['customer_id'] = pd.to_numeric(icp_df['customer_id'], errors='coerce')
            merged = shap_df.merge(icp_df, on='customer_id', how='left')
            # Reorder columns to surface name and score first
            leading_cols = [c for c in ['customer_name','icp_score','customer_id'] if c in merged.columns]
            remaining_cols = [c for c in merged.columns if c not in leading_cols]
            merged = merged[leading_cols + remaining_cols]
            top_n = merged.sort_values('icp_score', ascending=False, na_position='last').head(100)
            st.subheader("Top 100 Accounts by ICP Score (with SHAP detail)")
            st.caption("Sorted by predicted probability; includes customer name and SHAP contributions.")
            st.dataframe(top_n, use_container_width=True, height=500)
            st.download_button("Download Top 100 (with SHAP)", data=top_n.to_csv(index=False), file_name='top100_shap.csv')
        except Exception as e:
            st.warning(f"Could not merge SHAP with scores for top-100 view: {e}")
        # Feature glossary
        st.subheader("Feature Library & Glossary")
        # Feature catalog
        try:
            fc = pd.read_csv(OUTPUTS_DIR / 'feature_catalog_solidworks.csv')
            with st.expander("Feature catalog (names, dtypes, coverage)"):
                st.dataframe(fc, use_container_width=True, height=300)
                st.download_button("Download feature catalog", data=fc.to_csv(index=False), file_name='feature_catalog_solidworks.csv')
        except Exception:
            st.info("Feature catalog not found. Run the pipeline to emit it.")

        st.markdown("""
        Families implemented:
        - Core RFM and diversity (product & SKU)
        - Windowed aggregates (3/6/12/24 months): transactions, GP, average GP/tx
        - Temporal dynamics (12 months): monthly GP/TX slope & volatility
        - Cadence: tenure_days, interpurchase intervals, last_gap_days
        - Seasonality: quarter shares over 24 months
        - Division mix (12 months): per-division totals, GP shares, days_since_last_{division}
        - SKU micro-signals (12 months): GP, qty, GP-per-unit for key SKUs
        - Branch/Rep shares: top-branches and top-reps share of transactions (feature period)
        - Basket lift: weighted SKU presence-affinity vs Solidworks baseline
        - Industry join and selected interactions with engagement and growth
        """)
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

elif page == "Modeling & Validation":
    st.subheader("Training Metrics & Calibration")
    # Model card
    try:
        mc = pd.read_csv(OUTPUTS_DIR / 'model_card_solidworks.csv')
        st.caption("Model selection and AUC on the internal train/test split.")
        st.dataframe(mc, use_container_width=True)
    except Exception:
        st.info("Model card not available.")

    # Calibration curve
    try:
        calib = pd.read_csv(OUTPUTS_DIR / 'calibration_solidworks.csv')
        st.caption("Probability calibration: mean predicted vs fraction positives in quantile bins (train split).")
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=calib['bin'], y=calib['mean_predicted'], mode='lines+markers', name='Mean predicted', line=dict(color='#336D91')))
        fig.add_trace(go.Scatter(x=calib['bin'], y=calib['fraction_positives'], mode='lines+markers', name='Fraction positives', line=dict(color='#BAD532')))
        fig.update_layout(height=380, margin=dict(l=10,r=10,b=10,t=10), xaxis_title='Bin', yaxis_title='Rate')
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download calibration CSV", data=calib.to_csv(index=False), file_name='calibration_solidworks.csv')
    except Exception:
        st.info("Calibration CSV not found.")

    st.subheader("Holdout Validation (2025 YTD)")
    cols = st.columns(2)
    try:
        with open(OUTPUTS_DIR / 'validation_metrics_2025.json', 'r') as f:
            vm = json.load(f)
        cols[0].metric("AUC", f"{vm.get('auc_score', 0):.4f}")
        cols[0].metric("Precision", f"{vm.get('precision', 0):.4f}")
        cols[0].metric("Recall", f"{vm.get('recall', 0):.4f}")
        cols[0].metric("F1", f"{vm.get('f1_score', 0):.4f}")
        cols[1].metric("Total Customers", f"{vm.get('total_customers', 0):,}")
        cols[1].metric("Actual Buyers", f"{vm.get('actual_buyers', 0):,}")
        cols[1].metric("Conversion Rate", f"{vm.get('conversion_rate', 0):.4f}")
    except Exception:
        st.info("Validation metrics not found.")

    try:
        gains = pd.read_csv(OUTPUTS_DIR / 'validation_gains_2025.csv', header=[0,1])
        st.caption("Decile analysis (higher deciles = higher predicted probability)")
        # Flatten multiindex columns if present
        gains.columns = ['_'.join([str(c) for c in col if c]) for col in gains.columns.values]
        import plotly.express as px
        if 'bought_in_division_mean' in gains.columns:
            figg = px.bar(gains, x=gains.index+1, y='bought_in_division_mean', labels={'x':'Decile','bought_in_division_mean':'Conversion'}, color_discrete_sequence=['#BAD532'])
            figg.update_layout(height=420, margin=dict(l=10,r=10,b=10,t=10))
            st.plotly_chart(figg, use_container_width=True)
        st.dataframe(gains, use_container_width=True)
        st.download_button("Download gains CSV", data=gains.to_csv(index=False), file_name='validation_gains_2025.csv')
    except Exception:
        st.info("Gains table not found.")

elif page == "Scores & Whitespace":
    st.subheader("ICP Scores")
    st.caption("Predicted likelihood that a customer will purchase in the Solidworks division within the prediction window.")
    try:
        icp_scores = pd.read_csv(OUTPUTS_DIR / "icp_scores.csv")
        # Simple filters
        min_score = st.slider("Min ICP score", 0.0, 1.0, 0.0, 0.01)
        search = st.text_input("Search customer name contains", "")
        df = icp_scores.copy()
        df = df[df['icp_score'] >= min_score]
        if search:
            df = df[df['customer_name'].astype(str).str.contains(search, case=False, na=False)]
        st.dataframe(df, use_container_width=True, height=500)
        st.download_button("Download filtered scores", data=df.to_csv(index=False), file_name='icp_scores_filtered.csv')
    except FileNotFoundError:
        st.warning("ICP scores not available.")

    st.subheader("Whitespace Opportunities")
    st.caption("Products not yet purchased by a customer where model and behavioral signals suggest potential demand.")
    try:
        whitespace = pd.read_csv(OUTPUTS_DIR / "whitespace.csv")
        st.dataframe(whitespace, use_container_width=True, height=500)
        st.download_button("Download whitespace", data=whitespace.to_csv(index=False), file_name='whitespace.csv')
    except FileNotFoundError:
        st.warning("Whitespace opportunities not available.")

