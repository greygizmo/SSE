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


Mode = Literal["expansion", "all"]


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


def build_labels_for_division(
    engine,
    params: LabelParams,
) -> pl.DataFrame:
    # Load curated
    facts = pd.read_sql("SELECT customer_id, order_date, product_division, product_sku, gross_profit FROM fact_transactions", engine)
    customers = pd.read_sql("SELECT customer_id FROM dim_customer", engine)
    if facts.empty or customers.empty:
        return pl.DataFrame()

    # Time windows
    cutoff_dt = pd.to_datetime(params.cutoff)
    win_end = cutoff_dt + relativedelta(months=params.window_months)
    win_start = cutoff_dt + pd.Timedelta(days=1)

    # Coerce
    facts['order_date'] = pd.to_datetime(facts['order_date'], errors='coerce')
    facts['customer_id'] = pd.to_numeric(facts['customer_id'], errors='coerce').astype('Int64')
    customers['customer_id'] = pd.to_numeric(customers['customer_id'], errors='coerce').astype('Int64')

    # Feature-period activity
    feature_df = facts[facts['order_date'] <= cutoff_dt].copy()

    # Candidates by mode
    if params.mode == "expansion":
        cand = feature_df[['customer_id']].dropna().drop_duplicates()
    else:
        cand = customers[['customer_id']].dropna().drop_duplicates()

    # Window-period transactions for target division
    window_df = facts[(facts['order_date'] > cutoff_dt) & (facts['order_date'] <= win_end)].copy()
    # Normalize division string comparisons to avoid whitespace/case issues
    window_df['product_division'] = window_df['product_division'].astype(str).str.strip()
    window_target = window_df[window_df['product_division'] == normalize_division(params.division)].copy()
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
        if denylist:
            window_target = window_target[~window_target['product_sku'].astype(str).isin(denylist)].copy()
    except Exception:
        pass

    # Net GP per customer in window
    def _compute_labels(df_window: pd.DataFrame) -> pd.DataFrame:
        net_gp_local = df_window.groupby('customer_id')['gross_profit'].sum().rename('net_gp_window').reset_index()
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
                window_df_w = facts[(facts['order_date'] > cutoff_dt) & (facts['order_date'] <= new_end)].copy()
                window_df_w['product_division'] = window_df_w['product_division'].astype(str).str.strip()
                window_target_w = window_df_w[window_df_w['product_division'] == normalize_division(params.division)].copy()
                labels = _compute_labels(window_target_w)
                pos = int(labels['label'].sum()) if not labels.empty else 0
            # Update window_end if widened
            win_end = cutoff_dt + relativedelta(months=widened)
    except Exception:
        pass

    # Cohorts from feature period
    feature_df['product_division'] = feature_df['product_division'].astype(str).str.strip()

    had_any_df = (
        feature_df[['customer_id']]
        .dropna()
        .drop_duplicates()
        .assign(had_any=1)
    )

    had_div_df = (
        feature_df[feature_df['product_division'] == normalize_division(params.division)][['customer_id']]
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


