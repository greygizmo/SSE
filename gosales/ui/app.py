import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from gosales.utils.paths import OUTPUTS_DIR, MODELS_DIR
from gosales.ui.utils import discover_validation_runs, compute_validation_badges, load_thresholds, load_alerts, compute_default_validation_index


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

@st.cache_data(show_spinner=False)
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _format_ts(path: Path) -> str:
    """Return the last modified timestamp for a path."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"

def _discover_divisions() -> list[str]:
    divs: set[str] = set()
    try:
        for p in MODELS_DIR.glob("*_model"):
            if p.is_dir():
                divs.add(p.name.replace("_model", "").strip())
    except Exception:
        pass
    # Fallback: infer from metrics_*.json
    try:
        for p in OUTPUTS_DIR.glob("metrics_*.json"):
            name = p.stem.replace("metrics_", "").strip()
            if name:
                divs.add(name)
    except Exception:
        pass
    return sorted(divs, key=lambda s: s.lower())

def _discover_whitespace_cutoffs() -> list[str]:
    cutoffs: set[str] = set()
    for p in OUTPUTS_DIR.glob("whitespace_*.csv"):
        stem = p.stem
        # expected: whitespace_<cutoff>
        parts = stem.split("_")
        if len(parts) >= 2:
            cutoffs.add(parts[-1])
    return sorted(cutoffs, reverse=True)

# Sidebar controls
with st.sidebar:
    colr1, colr2 = st.columns([3,1])
    colr1.write(":memo: Navigation")
    if colr2.button("Refresh", help="Clear cached artifacts and reload"):
        st.cache_data.clear()
    tab = st.radio("Page", ["Overview", "Metrics", "Explainability", "Whitespace", "Validation", "Runs"], index=0)
    # Global divisions and default whitespace cutoff
    st.session_state.setdefault('divisions', _discover_divisions())
    # Preselect most recent whitespace cutoff
    try:
        wc = _discover_whitespace_cutoffs()
        if wc:
            st.session_state['latest_whitespace_cutoff'] = wc[0]
    except Exception:
        pass
    # Cache thresholds in session
    if 'thresholds' not in st.session_state:
        try:
            st.session_state['thresholds'] = load_thresholds()
        except Exception:
            st.session_state['thresholds'] = {}

    # Default preferred validation run: Solidworks @ 2024-06-30 if present
    if 'preferred_validation' not in st.session_state:
        try:
            runs = discover_validation_runs()
            for div, cut, _ in runs:
                if div.lower() == 'solidworks' and cut == '2024-06-30':
                    st.session_state['preferred_validation'] = {
                        'division': 'Solidworks',
                        'cutoff': '2024-06-30',
                    }
                    break
        except Exception:
            pass

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

if tab == "Overview":
    st.header("Overview")
    st.write("High-level summary and ETL/data-quality snapshots if available.")
    with st.expander("Tips"):
        st.markdown("- Use this page to sanity-check ETL coverage and contracts before training.")
        st.markdown("- If counts look off, re-run ETL or inspect `contracts/violations.csv` for failures.")
    # Industry coverage
    cov = OUTPUTS_DIR / 'industry_coverage_summary.csv'
    if cov.exists():
        try:
            df = _read_csv(cov)
            total = int(df.loc[df['metric']=='total_customers','value'].iloc[0]) if not df.empty else None
            with_ind = int(df.loc[df['metric']=='with_industry','value'].iloc[0]) if not df.empty else None
            pct = float(df.loc[df['metric']=='coverage_pct','value'].iloc[0]) if not df.empty else None
            c1, c2, c3 = st.columns(3)
            if total is not None:
                c1.metric("Total Customers", f"{total:,}")
            if with_ind is not None:
                c2.metric("With Industry", f"{with_ind:,}")
            if pct is not None:
                c3.metric("Coverage %", f"{pct:.2f}%")
            st.caption(f"Last updated: {_format_ts(cov)}")
        except Exception:
            st.info("Coverage summary could not be parsed.")
    # Contracts
    st.subheader("Data Contracts")
    cc1, cc2 = st.columns(2)
    rc = OUTPUTS_DIR / 'contracts' / 'row_counts.csv'
    if rc.exists():
        cc1.caption(f"Row counts — {_format_ts(rc)}")
        cc1.dataframe(_read_csv(rc), use_container_width=True)
    else:
        cc1.info("Row counts not available")
    viol = OUTPUTS_DIR / 'contracts' / 'violations.csv'
    if viol.exists():
        cc2.caption(f"Violations — {_format_ts(viol)}")
        cc2.dataframe(_read_csv(viol), use_container_width=True)
    else:
        cc2.info("Violations file not available or empty")

elif tab == "Metrics":
    st.header("Metrics & Training Artifacts")
    divisions = _discover_divisions()
    if not divisions:
        st.info("No divisions discovered (expected models/*_model or metrics_*.json)")
    else:
        div = st.selectbox("Division", divisions, help="Choose a division to view model artifacts")
        with st.expander("How to read these metrics", expanded=False):
            st.markdown("- AUC/PR-AUC: overall ranking quality (higher is better). Brier: probability accuracy (lower is better).")
            st.markdown("- Gains: average conversion by decile (1=top 10% by score). Expect decreasing pattern.")
            st.markdown("- Thresholds: score cut lines for top‑K%. Use to set capacity policies in ops.")
        # Model card
        mc_path = OUTPUTS_DIR / f"model_card_{div.lower()}.json"
        if mc_path.exists():
            st.subheader("Model Card")
            st.caption(f"Last updated: {_format_ts(mc_path)}")
            st.code(_read_text(mc_path))
        # Metrics JSON
        mt_path = OUTPUTS_DIR / f"metrics_{div.lower()}.json"
        if mt_path.exists():
            st.subheader("Training Metrics (JSON)")
            st.caption(f"Last updated: {_format_ts(mt_path)}")
            st.code(_read_text(mt_path))
        # Calibration
        cal_path = OUTPUTS_DIR / f"calibration_{div.lower()}.csv"
        if cal_path.exists():
            st.subheader("Calibration (train split)")
            st.caption(f"Last updated: {_format_ts(cal_path)}")
            st.caption("Mean predicted vs fraction positives by bins; close tracking indicates good calibration.")
            cal = _read_csv(cal_path)
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                if 'bin' in cal.columns:
                    x = cal['bin']
                else:
                    x = list(range(1, len(cal)+1))
                fig.add_trace(go.Scatter(x=x, y=cal['mean_predicted'], mode='lines+markers', name='Mean predicted'))
                fig.add_trace(go.Scatter(x=x, y=cal['fraction_positives'], mode='lines+markers', name='Fraction positives'))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(cal)
            st.download_button("Download calibration CSV", data=cal.to_csv(index=False), file_name=cal_path.name)
        # Gains
        g_path = OUTPUTS_DIR / f"gains_{div.lower()}.csv"
        if g_path.exists():
            st.subheader("Gains (train split)")
            st.caption(f"Last updated: {_format_ts(g_path)}")
            st.caption("Average conversion by decile (1=top 10% by score).")
            gains = _read_csv(g_path)
            try:
                import plotly.express as px
                ycol = 'bought_in_division_mean' if 'bought_in_division_mean' in gains.columns else gains.columns[1] if len(gains.columns)>1 else None
                x = gains['decile'] if 'decile' in gains.columns else list(range(1, len(gains)+1))
                if ycol:
                    figg = px.bar(gains, x=x, y=ycol)
                    st.plotly_chart(figg, use_container_width=True)
            except Exception:
                pass
            st.dataframe(gains, use_container_width=True)
            st.download_button("Download gains CSV", data=gains.to_csv(index=False), file_name=g_path.name)
        # Thresholds
        th_path = OUTPUTS_DIR / f"thresholds_{div.lower()}.csv"
        if th_path.exists():
            st.subheader("Top-K Thresholds")
            st.caption(f"Last updated: {_format_ts(th_path)}")
            st.caption("Score thresholds to select top‑K% of customers; use with capacity planning.")
            thr = _read_csv(th_path)
            st.dataframe(thr, use_container_width=True)
            st.download_button("Download thresholds CSV", data=thr.to_csv(index=False), file_name=th_path.name)

elif tab == "Explainability":
    st.header("Explainability (Phase 3)")
    divisions = _discover_divisions()
    if not divisions:
        st.info("No divisions discovered")
    else:
        div = st.selectbox("Division", divisions)
        sg = OUTPUTS_DIR / f"shap_global_{div.lower()}.csv"
        ss = OUTPUTS_DIR / f"shap_sample_{div.lower()}.csv"
        cf = OUTPUTS_DIR / f"coef_{div.lower()}.csv"
        # Feature catalog and stats (show latest by cutoff if multiple)
        cat_candidates = sorted(OUTPUTS_DIR.glob(f"feature_catalog_{div.lower()}_*.csv"), reverse=True)
        stats_candidates = sorted(OUTPUTS_DIR.glob(f"feature_stats_{div.lower()}_*.json"), reverse=True)
        if cat_candidates:
            st.subheader("Feature Catalog")
            cat = _read_csv(cat_candidates[0])
            st.caption("Columns: name (feature id), dtype (pandas dtype), coverage (non-null share). Use to assess feature availability.")
            st.caption(f"Last updated: {_format_ts(cat_candidates[0])}")
            st.dataframe(cat, use_container_width=True, height=320)
            st.download_button("Download feature catalog", data=cat.to_csv(index=False), file_name=cat_candidates[0].name)
        if stats_candidates:
            st.subheader("Feature Stats")
            st.caption("Includes per-column coverage; optional winsor caps for gp_sum features; checksum ensures determinism of the feature parquet.")
            st.caption(f"Last updated: {_format_ts(stats_candidates[0])}")
            st.code(_read_text(stats_candidates[0]))
        if sg.exists():
            with st.expander("SHAP Global — what it means", expanded=True):
                st.markdown("- Mean absolute SHAP reflects average feature influence magnitude across customers. Higher = more impact on predictions.")
                st.markdown("- Use this to identify globally important features; pair with coefficients for direction (if LR).")
            st.caption(f"Last updated: {_format_ts(sg)}")
            sg_df = _read_csv(sg)
            st.dataframe(sg_df, use_container_width=True, height=320)
            # Optional bar chart if aggregated column present
            try:
                import plotly.express as px
                if 'feature' in sg_df.columns and 'mean_abs_shap' in sg_df.columns:
                    topn = sg_df.sort_values('mean_abs_shap', ascending=False).head(20)
                    fig = px.bar(topn, x='feature', y='mean_abs_shap')
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
        if ss.exists():
            with st.expander("SHAP Sample — how to read", expanded=False):
                st.markdown("- Row = customer; columns = per-feature SHAP values.")
                st.markdown("- Sign: positive raises probability; negative lowers. Compare features within the same customer.")
                st.markdown("- Magnitude: larger absolute value = stronger effect for that customer.")
            st.caption(f"Last updated: {_format_ts(ss)}")
            ss_df = _read_csv(ss).head(200)
            st.dataframe(ss_df, use_container_width=True, height=320)
            st.download_button("Download SHAP sample", data=ss_df.to_csv(index=False), file_name=ss.name)
        if cf.exists():
            with st.expander("Logistic Regression Coefficients — interpretation", expanded=False):
                st.markdown("- Positive coefficient increases log-odds; negative decreases. Magnitude depends on feature scaling.")
                st.markdown("- Combine with SHAP for instance-level interpretation.")
            st.caption(f"Last updated: {_format_ts(cf)}")
            cf_df = _read_csv(cf)
            st.dataframe(cf_df, use_container_width=True, height=320)
            st.download_button("Download coefficients", data=cf_df.to_csv(index=False), file_name=cf.name)
        if not any(p.exists() for p in [sg, ss, cf]):
            st.info("No explainability artifacts found for this division.")

elif tab == "Whitespace":
    st.header("Whitespace (Phase 4)")
    cutoffs = _discover_whitespace_cutoffs()
    if not cutoffs:
        st.info("No whitespace files found.")
    else:
        # Use latest cutoff as default
        latest = st.session_state.get('latest_whitespace_cutoff')
        default_idx = cutoffs.index(latest) if latest in cutoffs else 0
        sel_cut = st.selectbox("Cutoff", cutoffs, index=default_idx, help="Choose ranking outputs by cutoff date (latest auto-selected)")
        ws = OUTPUTS_DIR / f"whitespace_{sel_cut}.csv"
        if ws.exists():
            st.caption(f"Last updated: {_format_ts(ws)}")
            # Filters
            df = _read_csv(ws)
            if not df.empty:
                # Simple filters on key columns when present
                cols = st.multiselect("Columns to show", df.columns.tolist(), default=df.columns.tolist()[:12], help="Tip: reduce visible columns to focus on key signals")
                if cols:
                    st.dataframe(df[cols].head(200), use_container_width=True)
                else:
                    st.dataframe(df.head(200), use_container_width=True)
                st.download_button("Download whitespace", data=df.to_csv(index=False), file_name=ws.name)
        # Explanations
        ex = OUTPUTS_DIR / f"whitespace_explanations_{sel_cut}.csv"
        if ex.exists():
            st.subheader("Explanations")
            st.caption(f"Last updated: {_format_ts(ex)}")
            st.caption("Short reasons combining key drivers (probability, affinity, EV).")
            st.dataframe(_read_csv(ex).head(200), use_container_width=True)
        # Metrics
        wm = OUTPUTS_DIR / f"whitespace_metrics_{sel_cut}.json"
        if wm.exists():
            st.subheader("Whitespace Metrics")
            st.caption(f"Last updated: {_format_ts(wm)}")
            st.caption("Capture@K, division shares, stability vs prior run, coverage, and weights.")
            st.code(_read_text(wm))
        # Thresholds
        wthr = OUTPUTS_DIR / f"thresholds_whitespace_{sel_cut}.csv"
        if wthr.exists():
            st.subheader("Capacity Thresholds")
            st.caption(f"Last updated: {_format_ts(wthr)}")
            st.caption("Top‑percent / per‑rep / hybrid thresholds for list sizing & diversification.")
            st.dataframe(_read_csv(wthr), use_container_width=True)
        # Logs preview
        wlog = OUTPUTS_DIR / f"whitespace_log_{sel_cut}.jsonl"
        if wlog.exists():
            st.subheader("Log Preview")
            st.caption(f"Last updated: {_format_ts(wlog)}")
            st.caption("First 50 structured log rows; use for quick audit and guardrails.")
            lines = _read_jsonl(wlog)
            st.code(json.dumps(lines[:50], indent=2))
        # Market-basket rules (division-specific; match this cutoff)
        mb_files = list(OUTPUTS_DIR.glob(f"mb_rules_*_{sel_cut}.csv"))
        if mb_files:
            st.subheader("Market-Basket Rules")
            st.caption("SKU-level co‑occurrence rules; Lift > 1 indicates positive association with the target division.")
            sel_mb = st.selectbox("Select rules file", mb_files, format_func=lambda p: p.name)
            st.caption(f"Last updated: {_format_ts(sel_mb)}")
            mb = _read_csv(sel_mb)
            st.dataframe(mb.head(300), use_container_width=True)
            st.download_button("Download rules CSV", data=mb.to_csv(index=False), file_name=sel_mb.name)

elif tab == "Validation":
    st.header("Forward Validation (Phase 5)")
    runs = discover_validation_runs()
    if not runs:
        st.info("No validation runs found.")
    else:
        labels = [f"{div} @ {cut}" for div, cut, _ in runs]
        # Prefer selection from session state if provided by Runs page
        default_index = compute_default_validation_index(runs, st.session_state.get('preferred_validation'))
        sel = st.selectbox("Pick run", options=list(range(len(runs))), index=default_index, format_func=lambda i: labels[i])
        _, _, path = runs[sel]
        thr = st.session_state.get('thresholds', load_thresholds())
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
        with st.expander("What these badges mean"):
            st.markdown("- Calibration MAE: average absolute gap between predicted probability and observed rate (lower is better).")
            st.markdown("- PSI(EV vs GP): value-weighted distribution drift between expected value proxy and realized GP over deciles (lower is better).")
            st.markdown("- KS(train vs holdout): max CDF gap between train and holdout score distributions (lower is better).")

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
            st.caption(f"Last updated: {_format_ts(metrics_path)}")
            st.code(metrics_path.read_text(encoding='utf-8'))
        # Drift
        drift_path = path / 'drift.json'
        if drift_path.exists():
            st.subheader("Drift")
            st.caption(f"Last updated: {_format_ts(drift_path)}")
            st.code(drift_path.read_text(encoding='utf-8'))
        # Calibration (holdout)
        cal_path = path / 'calibration.csv'
        if cal_path.exists():
            st.subheader("Calibration (holdout)")
            st.caption(f"Last updated: {_format_ts(cal_path)}")
            st.caption("Probability calibration on holdout; closer lines indicate better calibration.")
            cal = _read_csv(cal_path)
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                x = cal['bin'] if 'bin' in cal.columns else list(range(1, len(cal)+1))
                fig.add_trace(go.Scatter(x=x, y=cal['mean_predicted'], mode='lines+markers', name='Mean predicted'))
                fig.add_trace(go.Scatter(x=x, y=cal['fraction_positives'], mode='lines+markers', name='Fraction positives'))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(cal)
        # Gains (holdout)
        g2_path = path / 'gains.csv'
        if g2_path.exists():
            st.subheader("Gains (holdout)")
            st.caption(f"Last updated: {_format_ts(g2_path)}")
            st.caption("Average conversion by decile in holdout data.")
            gains2 = _read_csv(g2_path)
            try:
                import plotly.express as px
                ycol = 'fraction_positives' if 'fraction_positives' in gains2.columns else gains2.columns[1] if len(gains2.columns)>1 else None
                x = gains2['decile'] if 'decile' in gains2.columns else list(range(1, len(gains2)+1))
                if ycol:
                    figh = px.bar(gains2, x=x, y=ycol)
                    st.plotly_chart(figh, use_container_width=True)
            except Exception:
                pass
            st.dataframe(gains2, use_container_width=True)
        # Scenarios
        scen_path = path / 'topk_scenarios_sorted.csv'
        if scen_path.exists():
            st.subheader("Scenarios (sorted)")
            st.caption(f"Last updated: {_format_ts(scen_path)}")
            st.dataframe(pd.read_csv(scen_path))
        # Segment performance
        seg_path = path / 'segment_performance.csv'
        if seg_path.exists():
            st.subheader("Segment performance")
            st.caption(f"Last updated: {_format_ts(seg_path)}")
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
            st.caption("Each entry is a pipeline run with start/finish, phase, status, and artifact path.")
            st.caption(f"Last updated: {_format_ts(reg_path)}")
            # Flag dry-run entries
            if 'status' in df.columns:
                df['note'] = df['status'].apply(lambda s: 'dry-run (no compute)' if str(s).lower()=='dry-run' else '')
            st.dataframe(df, use_container_width=True, height=300)
            idx = st.number_input("Select row index", min_value=0, max_value=max(0, len(df)-1), value=0, step=1, help="Pick a run to view manifest/config and deep-link to Validation (if applicable)")
            sel = df.iloc[int(idx)]
            st.subheader(f"Run {sel.get('run_id','?')} — {sel.get('phase','?')} [{sel.get('status','?')}]")
            run_dir = Path(sel.get('artifacts_path', OUTPUTS_DIR / 'runs' / str(sel.get('run_id',''))))
            man = run_dir / 'manifest.json'
            cfg = run_dir / 'config_resolved.yaml'
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Manifest (planned/emitted artifacts)")
                if man.exists():
                    st.caption(f"Last updated: {_format_ts(man)}")
                    st.code(_read_text(man))
                    st.download_button("Download manifest.json", data=man.read_bytes(), file_name='manifest.json')
                else:
                    st.info("manifest.json not found")
            with c2:
                st.caption("Resolved Config Snapshot")
                if cfg.exists():
                    st.caption(f"Last updated: {_format_ts(cfg)}")
                    st.code(_read_text(cfg))
                    st.download_button("Download config_resolved.yaml", data=cfg.read_bytes(), file_name='config_resolved.yaml')
                else:
                    st.info("config_resolved.yaml not found")
            # Quick link to Validation page when applicable
            phase = str(sel.get('phase',''))
            division = sel.get('division')
            cutoff = sel.get('cutoff')
            if phase == 'phase5_validation' and division and cutoff:
                if st.button("View this validation run"):
                    st.session_state['preferred_validation'] = {'division': division, 'cutoff': cutoff}
                    st.info("Open the Validation page to view this run.")
