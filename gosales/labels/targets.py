from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import polars as pl

from gosales.utils.paths import OUTPUTS_DIR


Mode = Literal["expansion", "all"]


@dataclass
class LabelParams:
    division: str
    cutoff: str
    window_months: int
    mode: Mode = "expansion"
    gp_min_threshold: float = 0.0


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
    window_target = window_df[window_df['product_division'] == params.division].copy()

    # Net GP per customer in window
    net_gp = window_target.groupby('customer_id')['gross_profit'].sum().rename('net_gp_window').reset_index()
    labels = cand.merge(net_gp, on='customer_id', how='left')
    labels['net_gp_window'] = labels['net_gp_window'].fillna(0.0)
    labels['label'] = (labels['net_gp_window'] > float(params.gp_min_threshold)).astype('int8')

    # Cohorts from feature period
    had_any = set(feature_df['customer_id'].dropna().astype('int64').tolist())
    had_div = set(feature_df.loc[feature_df['product_division'] == params.division, 'customer_id'].dropna().astype('int64').tolist())
    def _cohorts(cid: int) -> tuple[int,int,int]:
        is_new_logo = 0 if cid in had_any else 1
        is_renewal_like = 1 if (cid in had_div and cid in had_any) else 0
        is_expansion = 1 if (cid in had_any and cid not in had_div) else 0
        return is_new_logo, is_expansion, is_renewal_like

    tmp = []
    for r in labels['customer_id'].dropna().astype('int64').tolist():
        n, e, rl = _cohorts(r)
        tmp.append((r, n, e, rl))
    cohorts = pd.DataFrame(tmp, columns=['customer_id', 'is_new_logo', 'is_expansion', 'is_renewal_like'])
    labels = labels.merge(cohorts, on='customer_id', how='left')
    labels[['is_new_logo', 'is_expansion', 'is_renewal_like']] = labels[['is_new_logo', 'is_expansion', 'is_renewal_like']].fillna(0).astype('int8')

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


