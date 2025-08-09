import json
from pathlib import Path

import pandas as pd
import streamlit as st

from gosales.utils.paths import OUTPUTS_DIR, MODELS_DIR
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
    # Industry coverage
    cov = OUTPUTS_DIR / 'industry_coverage_summary.csv'
    if cov.exists():
        try:
            df = _read_csv(cov)
            total = int(df.loc[df['metric']=='total_customers','value'].iloc[0]) if not df.empty else None
            with_ind = int(df.loc[df['metric']=='with_industry','value'].iloc[0]) if not df.empty else None
            pct = float(df.loc[df['metric']=='coverage_pct','value'].iloc[0]) if not df.empty else None
            c1, c2, c3 = st.columns(3)
            if total is not None: c1.metric("Total Customers", f"{total:,}")
            if with_ind is not None: c2.metric("With Industry", f"{with_ind:,}")
            if pct is not None: c3.metric("Coverage %", f"{pct:.2f}%")
        except Exception:
            st.info("Coverage summary could not be parsed.")
    # Contracts
    st.subheader("Data Contracts")
    cc1, cc2 = st.columns(2)
    rc = OUTPUTS_DIR / 'contracts' / 'row_counts.csv'
    if rc.exists():
        cc1.caption("Row counts")
        cc1.dataframe(_read_csv(rc), use_container_width=True)
    else:
        cc1.info("Row counts not available")
    viol = OUTPUTS_DIR / 'contracts' / 'violations.csv'
    if viol.exists():
        cc2.caption("Violations")
        cc2.dataframe(_read_csv(viol), use_container_width=True)
    else:
        cc2.info("Violations file not available or empty")

elif tab == "Metrics":
    st.header("Metrics & Training Artifacts")
    divisions = _discover_divisions()
    if not divisions:
        st.info("No divisions discovered (expected models/*_model or metrics_*.json)")
    else:
        div = st.selectbox("Division", divisions)
        # Model card
        mc_path = OUTPUTS_DIR / f"model_card_{div.lower()}.json"
        if mc_path.exists():
            st.subheader("Model Card")
            st.code(_read_text(mc_path))
        # Metrics JSON
        mt_path = OUTPUTS_DIR / f"metrics_{div.lower()}.json"
        if mt_path.exists():
            st.subheader("Training Metrics (JSON)")
            st.code(_read_text(mt_path))
        # Calibration
        cal_path = OUTPUTS_DIR / f"calibration_{div.lower()}.csv"
        if cal_path.exists():
            st.subheader("Calibration (train split)")
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
        if sg.exists():
            st.subheader("SHAP Global")
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
            st.subheader("SHAP Sample")
            st.dataframe(_read_csv(ss).head(200), use_container_width=True, height=320)
        if cf.exists():
            st.subheader("Logistic Regression Coefficients")
            st.dataframe(_read_csv(cf), use_container_width=True, height=320)
        if not any(p.exists() for p in [sg, ss, cf]):
            st.info("No explainability artifacts found for this division.")

elif tab == "Whitespace":
    st.header("Whitespace (Phase 4)")
    cutoffs = _discover_whitespace_cutoffs()
    if not cutoffs:
        st.info("No whitespace files found.")
    else:
        sel_cut = st.selectbox("Cutoff", cutoffs)
        ws = OUTPUTS_DIR / f"whitespace_{sel_cut}.csv"
        if ws.exists():
            # Filters
            df = _read_csv(ws)
            if not df.empty:
                # Simple filters on key columns when present
                cols = st.multiselect("Columns to show", df.columns.tolist(), default=df.columns.tolist()[:12])
                if cols:
                    st.dataframe(df[cols].head(200), use_container_width=True)
                else:
                    st.dataframe(df.head(200), use_container_width=True)
                st.download_button("Download whitespace", data=df.to_csv(index=False), file_name=ws.name)
        # Explanations
        ex = OUTPUTS_DIR / f"whitespace_explanations_{sel_cut}.csv"
        if ex.exists():
            st.subheader("Explanations")
            st.dataframe(_read_csv(ex).head(200), use_container_width=True)
        # Metrics
        wm = OUTPUTS_DIR / f"whitespace_metrics_{sel_cut}.json"
        if wm.exists():
            st.subheader("Whitespace Metrics")
            st.code(_read_text(wm))
        # Thresholds
        wthr = OUTPUTS_DIR / f"thresholds_whitespace_{sel_cut}.csv"
        if wthr.exists():
            st.subheader("Capacity Thresholds")
            st.dataframe(_read_csv(wthr), use_container_width=True)
        # Logs preview
        wlog = OUTPUTS_DIR / f"whitespace_log_{sel_cut}.jsonl"
        if wlog.exists():
            st.subheader("Log Preview")
            lines = _read_jsonl(wlog)
            st.code(json.dumps(lines[:50], indent=2))

elif tab == "Validation":
    st.header("Forward Validation (Phase 5)")
    runs = discover_validation_runs()
    if not runs:
        st.info("No validation runs found.")
    else:
        labels = [f"{div} @ {cut}" for div, cut, _ in runs]
        # Prefer selection from session state if provided by Runs page
        default_index = 0
        pref = st.session_state.get('preferred_validation')
        if isinstance(pref, dict):
            try:
                default_index = next((i for i,(d,c,_) in enumerate(runs) if d == pref.get('division') and c == pref.get('cutoff')), 0)
            except Exception:
                default_index = 0
        sel = st.selectbox("Pick run", options=list(range(len(runs))), index=default_index, format_func=lambda i: labels[i])
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
            st.code(metrics_path.read_text(encoding='utf-8'))
        # Drift
        drift_path = path / 'drift.json'
        if drift_path.exists():
            st.subheader("Drift")
            st.code(drift_path.read_text(encoding='utf-8'))
        # Calibration (holdout)
        cal_path = path / 'calibration.csv'
        if cal_path.exists():
            st.subheader("Calibration (holdout)")
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
                    st.download_button("Download manifest.json", data=man.read_bytes(), file_name='manifest.json')
                else:
                    st.info("manifest.json not found")
            with c2:
                st.caption("Resolved Config Snapshot")
                if cfg.exists():
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
