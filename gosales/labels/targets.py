from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import polars as pl

from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils import config as cfg
from gosales.utils.normalize import normalize_division
from gosales.etl.sku_map import get_model_targets


Mode = Literal["expansion", "all"]
TargetType = Literal["division", "goal", "rollup"]


@dataclass
class LabelParams:
    division: str
    cutoff: str
    window_months: int
    mode: Mode = "expansion"
    gp_min_threshold: float = 0.0
    # Optional: widen window for sparse divisions up to max_window_months to hit a minimum positives target
    min_positive_target: Optional[int] = None
    max_window_months: int = 12
    # Which categorization to use for labels: canonical division, Goal, or item_rollup
    target_type: Optional[TargetType] = None


def build_labels_for_division(
    engine,
    params: LabelParams,
) -> pl.DataFrame:
    # Load curated (legacy fact for compatibility + customers)
    try:
        facts = pd.read_sql("SELECT * FROM fact_transactions", engine)
        # Keep only relevant columns if present
        keep_cols = [
            c
            for c in (
                "customer_id",
                "order_date",
                "product_division",
                "product_goal",
                "product_sku",
                "gross_profit",
            )
            if c in facts.columns
        ]
        facts = facts[keep_cols] if keep_cols else pd.DataFrame()
    except Exception:
        facts = pd.DataFrame()
    try:
        customers = pd.read_sql("SELECT customer_id FROM dim_customer", engine)
    except Exception:
        customers = pd.DataFrame()
    if facts.empty or customers.empty:
        return pl.DataFrame()

    # Time windows
    cutoff_dt = pd.to_datetime(params.cutoff)
    win_end = cutoff_dt + relativedelta(months=params.window_months)
    win_start = cutoff_dt + pd.Timedelta(days=1)

    # Coerce
    facts['order_date'] = pd.to_datetime(facts['order_date'], errors='coerce')
    facts['customer_id'] = facts['customer_id'].astype(str)
    customers['customer_id'] = customers['customer_id'].astype(str)

    # Feature-period activity (legacy transactions); tolerate missing columns
    if 'order_date' in facts.columns:
        feature_df = facts[pd.to_datetime(facts['order_date'], errors='coerce') <= cutoff_dt].copy()
    else:
        feature_df = facts.copy()

    # Candidates by mode
    if params.mode == "expansion":
        cand = feature_df[['customer_id']].dropna().drop_duplicates()
    else:
        cand = customers[['customer_id']].dropna().drop_duplicates()

    # Window-period transactions (legacy)
    if 'order_date' in facts.columns:
        window_df = facts[(facts['order_date'] > cutoff_dt) & (facts['order_date'] <= win_end)].copy()
    else:
        window_df = facts.copy()
    # Normalize string comparisons to avoid whitespace/case issues
    if 'product_division' in window_df.columns:
        window_df['product_division'] = (
            window_df['product_division'].astype(str).str.strip().str.casefold()
        )
    if 'product_goal' in window_df.columns:
        window_df['product_goal'] = window_df['product_goal'].astype(str).str.strip().str.casefold()
    # Determine if caller passed a custom model (e.g., 'Printers'); if so, match by SKU set
    target_norm = normalize_division(params.division)
    sku_targets = tuple(get_model_targets(target_norm))
    # Optional division aliases (e.g., 'cad' -> ['solidworks','pdm',...])
    alias_divisions: list[str] = []
    try:
        cfg_obj = cfg.load_config()
        raw_alias = getattr(getattr(cfg_obj, 'features', object()), 'division_aliases', {})  # type: ignore[attr-defined]
        if isinstance(raw_alias, dict):
            alias_divisions = [normalize_division(d) for d in (raw_alias.get(target_norm, []) or []) if str(d).strip()]
    except Exception:
        alias_divisions = []
    # Resolve target type (categorization) from params or config
    try:
        cfg_obj = cfg.load_config()
        cfg_target_type = getattr(getattr(cfg_obj, 'labels', object()), 'target_type', None)
    except Exception:
        cfg_target_type = None

    raw_target_type = params.target_type or cfg_target_type or 'division'
    target_kind = str(raw_target_type or '').strip().lower()
    if target_kind == 'goal':
        target_kind = 'division'
    elif target_kind in {'sub_division', 'sub-division'}:
        target_kind = 'rollup'
    if target_kind not in {'division', 'rollup'}:
        target_kind = 'division'
    target_type: TargetType = target_kind  # type: ignore[assignment]

    rollup_mode = False

    # Helper: read line-item window with gp_value + categorizations
    def _read_line_window(_end: pd.Timestamp) -> pd.DataFrame:
        try:
            sql = (
                "SELECT CompanyId AS customer_id, Rec_Date AS order_date, "
                "COALESCE(GP_usd, GP, 0.0) AS gp_value, "
                "item_rollup, division_goal, division_canonical "
                "FROM fact_sales_line WHERE Rec_Date > :cutoff AND Rec_Date <= :win_end"
            )
            # Use ISO date strings for broad DBAPI compatibility (SQLite, Azure)
            params_sql = {
                "cutoff": pd.to_datetime(cutoff_dt).date().isoformat(),
                "win_end": pd.to_datetime(_end).date().isoformat(),
            }
            df = pd.read_sql_query(sql, engine, params=params_sql)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        df['customer_id'] = df['customer_id'].astype(str)
        # Normalized keys for robust matching
        norm = lambda s: (
            s.astype(str).str.strip().str.lower()
            .str.replace(r"[^0-9a-z]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        for col in ("item_rollup", "division_goal", "division_canonical"):
            if col in df.columns:
                df[col] = norm(df[col])
        # Ensure numeric gp_value
        df['gp_value'] = pd.to_numeric(df.get('gp_value'), errors='coerce').fillna(0.0)
        # Coerce order_date
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        return df

    if sku_targets:
        window_target = window_df[window_df['product_sku'].astype(str).isin(sku_targets)].copy()
    elif alias_divisions and target_type == 'division':
        window_target = window_df[window_df['product_division'].isin(alias_divisions)].copy()
    else:
        # Division (Goal) or rollup selection
        t = str(target_type).lower()
        if t == 'rollup':
            lw = _read_line_window(win_end)
            if not lw.empty and 'item_rollup' in lw.columns:
                window_target = lw[lw['item_rollup'] == target_norm].copy()
                rollup_mode = True
            else:
                # Last resort: match canonical division if available; otherwise legacy division
                if not lw.empty and 'division_canonical' in lw.columns:
                    window_target = lw[lw['division_canonical'] == target_norm].copy()
                else:
                    window_target = window_df[window_df.get('product_division') == target_norm].copy()
        else:
            lw = _read_line_window(win_end)
            if not lw.empty and 'division_goal' in lw.columns:
                window_target = lw[lw['division_goal'] == target_norm].copy()
            elif 'product_goal' in window_df.columns:
                window_target = window_df[window_df['product_goal'] == target_norm].copy()
            elif 'product_division' in window_df.columns and (window_df['product_division'] == target_norm).any():
                window_target = window_df[window_df['product_division'] == target_norm].copy()
            else:
                window_target = pd.DataFrame(columns=['customer_id'])
    # Optional denylist SKUs exclusion (e.g., trials/POC)
    try:
        cfg_obj = cfg.load_config()
        denylist = []
        if cfg_obj.labels.denylist_skus_csv and Path(cfg_obj.labels.denylist_skus_csv).exists():
            dl = pd.read_csv(cfg_obj.labels.denylist_skus_csv)
            col = None
            for c in dl.columns:
                if c.lower() in ("sku", "product_sku", "gp_col"):
                    col = c
                    break
            if col:
                denylist = dl[col].dropna().astype(str).str.strip().unique().tolist()
        if denylist and 'product_sku' in window_target.columns:
            window_target = window_target[~window_target['product_sku'].astype(str).isin(denylist)].copy()
    except Exception:
        pass

    # Net GP per customer in window
    def _compute_labels(df_window: pd.DataFrame) -> pd.DataFrame:
        gp_colname = None
        for gp_cand in ('gross_profit', 'gp_value', 'GP_usd', 'GP'):
            if gp_cand in df_window.columns:
                gp_colname = gp_cand
                break
        if gp_colname is None:
            # Nothing to sum; treat as zeros
            net_gp_local = (
                df_window[['customer_id']].dropna().drop_duplicates().assign(net_gp_window=0.0)
            )
        else:
            net_gp_local = (
                df_window.groupby('customer_id')[gp_colname]
                .sum()
                .rename('net_gp_window')
                .reset_index()
            )
        lab = cand.merge(net_gp_local, on='customer_id', how='left')
        lab['net_gp_window'] = lab['net_gp_window'].fillna(0.0)
        thr = float(params.gp_min_threshold if params.gp_min_threshold is not None else (cfg.load_config().labels.gp_min_threshold or 0.0))
        lab['label'] = (lab['net_gp_window'] > thr).astype('int8')
        return lab

    labels = _compute_labels(window_target)

    # Auto-widening for sparse divisions, if requested
    try:
        if params.min_positive_target and params.min_positive_target > 0:
            pos = int(labels['label'].sum()) if not labels.empty else 0
            widened = int(params.window_months)
            while pos < int(params.min_positive_target) and widened < int(params.max_window_months):
                widened = min(int(params.max_window_months), widened + 3)
                new_end = cutoff_dt + relativedelta(months=widened)
                # Rebuild selection under widened window respecting target_type
                if sku_targets:
                    window_df_w = facts[(facts['order_date'] > cutoff_dt) & (facts['order_date'] <= new_end)].copy()
                    window_target_w = window_df_w[
                        window_df_w['product_sku'].astype(str).isin(sku_targets)
                    ].copy()
                else:
                    t = str(target_type).lower()
                    if t == 'rollup':
                        lw = _read_line_window(new_end)
                        if not lw.empty and 'item_rollup' in lw.columns:
                            window_target_w = lw[lw['item_rollup'] == target_norm].copy()
                        else:
                            window_target_w = pd.DataFrame(columns=['customer_id'])
                    else:
                        lw = _read_line_window(new_end)
                        if not lw.empty and 'division_goal' in lw.columns:
                            window_target_w = lw[lw['division_goal'] == target_norm].copy()
                        else:
                            window_df_w = facts[(facts['order_date'] > cutoff_dt) & (facts['order_date'] <= new_end)].copy()
                            if 'product_goal' in window_df_w.columns:
                                window_df_w['product_goal'] = window_df_w.get('product_goal', '').astype(str).str.strip().str.casefold()
                                window_target_w = window_df_w[window_df_w['product_goal'] == target_norm].copy()
                            else:
                                window_df_w['product_division'] = (
                                    window_df_w.get('product_division', '').astype(str).str.strip().str.casefold()
                                )
                                window_target_w = window_df_w[window_df_w['product_division'] == target_norm].copy()
                labels = _compute_labels(window_target_w)
                pos = int(labels['label'].sum()) if not labels.empty else 0
            # Update window_end if widened
            win_end = cutoff_dt + relativedelta(months=widened)
    except Exception:
        pass

    # Cohorts from feature period
    if 'product_division' in feature_df.columns:
        feature_df['product_division'] = (
            feature_df['product_division'].astype(str).str.strip().str.casefold()
        )
    if 'product_goal' in feature_df.columns:
        feature_df['product_goal'] = feature_df['product_goal'].astype(str).str.strip().str.casefold()

    had_any_df = (
        feature_df[['customer_id']]
        .dropna()
        .drop_duplicates()
        .assign(had_any=1)
    )

    if sku_targets:
        had_div_df = (
            feature_df[feature_df['product_sku'].astype(str).isin(sku_targets)][['customer_id']]
            .dropna()
            .drop_duplicates()
            .assign(had_div=1)
        )
    elif alias_divisions and target_type == 'division':
        had_div_df = (
            feature_df[feature_df['product_division'].isin(alias_divisions)][['customer_id']]
            .dropna()
            .drop_duplicates()
            .assign(had_div=1)
        )
    else:
        t = str(target_type).lower()
        if t == 'division' and 'product_goal' in feature_df.columns and (feature_df['product_goal'] == target_norm).any():
            had_div_df = (
                feature_df[feature_df['product_goal'] == target_norm][['customer_id']]
                .dropna()
                .drop_duplicates()
                .assign(had_div=1)
            )
        else:
            try:
                sqlf = (
                    "SELECT CompanyId AS customer_id, Rec_Date AS order_date, item_rollup, division_goal "
                    "FROM fact_sales_line WHERE Rec_Date <= :cutoff"
                )
                rf = pd.read_sql_query(
                    sqlf,
                    engine,
                    params={"cutoff": pd.to_datetime(cutoff_dt).date().isoformat()},
                )
            except Exception:
                rf = pd.DataFrame()
            if not rf.empty:
                norm = lambda s: (
                    s.astype(str).str.strip().str.lower()
                    .str.replace(r"[^0-9a-z]+", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )
                rf['customer_id'] = rf['customer_id'].astype(str)
                if t == 'rollup' and 'item_rollup' in rf.columns:
                    rf['item_rollup'] = norm(rf['item_rollup'])
                    mask = rf['item_rollup'] == target_norm
                    had_div_df = (
                        rf.loc[mask, ['customer_id']]
                        .dropna()
                        .drop_duplicates()
                        .assign(had_div=1)
                    )
                elif t == 'division' and 'division_goal' in rf.columns:
                    rf['division_goal'] = norm(rf['division_goal'])
                    mask = rf['division_goal'] == target_norm
                    had_div_df = (
                        rf.loc[mask, ['customer_id']]
                        .dropna()
                        .drop_duplicates()
                        .assign(had_div=1)
                    )
                else:
                    had_div_df = (
                        feature_df[feature_df.get('product_division') == target_norm][['customer_id']]
                        .dropna()
                        .drop_duplicates()
                        .assign(had_div=1)
                    )
            else:
                had_div_df = (
                    feature_df[feature_df.get('product_division') == target_norm][['customer_id']]
                    .dropna()
                    .drop_duplicates()
                    .assign(had_div=1)
                )

    labels = (
        labels.merge(had_any_df, on='customer_id', how='left')
        .merge(had_div_df, on='customer_id', how='left')
    )

    labels[['had_any', 'had_div']] = labels[['had_any', 'had_div']].fillna(0).astype('int8')
    labels['is_new_logo'] = (1 - labels['had_any']).astype('int8')
    labels['is_renewal_like'] = ((labels['had_any'] == 1) & (labels['had_div'] == 1)).astype('int8')
    labels['is_expansion'] = ((labels['had_any'] == 1) & (labels['had_div'] == 0)).astype('int8')
    labels.drop(columns=['had_any', 'had_div'], inplace=True)

    # Censoring detection
    max_seen = pd.to_datetime(facts['order_date'].max(), errors='coerce')
    censored_flag = int(pd.isna(max_seen) or (max_seen < win_end))
    labels['censored_flag'] = censored_flag

    # Attach meta
    labels['division'] = params.division
    labels['window_start'] = win_start.date().isoformat()
    labels['window_end'] = win_end.date().isoformat()

    # Dedupe to one row per (customer, division)
    labels = labels[['customer_id', 'division', 'label', 'window_start', 'window_end', 'is_new_logo', 'is_expansion', 'is_renewal_like', 'censored_flag', 'net_gp_window']]
    labels = labels.drop_duplicates(subset=['customer_id', 'division'], keep='first')

    # Ensure customer_id is numeric for downstream tests/consumers that expect ints
    try:
        labels['customer_id'] = pd.to_numeric(labels['customer_id'], errors='coerce').astype('Int64')
    except Exception:
        # Fall back silently if conversion is unsafe
        pass

    return pl.from_pandas(labels)


def prevalence_report(labels_df: pl.DataFrame) -> pd.DataFrame:
    if labels_df.is_empty():
        return pd.DataFrame()
    df = labels_df.to_pandas()
    total = len(df)
    pos = int(df['label'].sum())
    prevalence = round(pos / total, 6) if total else 0.0
    cohorts = df[['is_new_logo', 'is_expansion', 'is_renewal_like']].sum().to_dict()
    return pd.DataFrame([
        {"total": total, "positives": pos, "prevalence": prevalence, **cohorts}
    ])

