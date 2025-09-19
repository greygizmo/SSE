from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from gosales.utils.db import get_curated_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger


logger = get_logger(__name__)


SAFE_COLS = [
    "customer_id",
    "ns_standardized_territory",
    "ns_territory_name",
    "ns_region",
    "ns_date_created",
    "ns_last_modified",
    "ns_lead_source",
    "ns_lead_source_name",
    "ns_terms_value",
    "ns_taxable",
    "ns_known_competitor",
    "ns_cad_named_account",
    "ns_sim_named_account",
    "ns_am_named_account",
    "ns_electrical_named_account",
]


def load_enriched_customers(engine=None) -> pl.DataFrame:
    engine = engine or get_curated_connection()
    df = pl.read_database(
        f"SELECT {', '.join(SAFE_COLS)} FROM dim_customer_enriched",
        engine,
    )
    # Cast date-like to datetime
    for c in ["ns_date_created", "ns_last_modified"]:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Utf8, strict=False).str.strptime(pl.Datetime, strict=False))
    return df


def build_ns_customer_features(cutoff_date: str, engine=None) -> pl.DataFrame:
    """Cutoff-safe NetSuite features for existing customers (no transactions).

    Returns a keyed DataFrame (customer_id, cutoff_date, feature columns) that can be left-joined
    to the existing customer feature matrix.
    """
    engine = engine or get_curated_connection()
    cutoff = pd.to_datetime(cutoff_date)
    base = load_enriched_customers(engine)

    df = base.to_pandas()
    df["cutoff_date"] = cutoff.date()

    # Clean enums and presence flags
    def _presence(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.strip().ne("").astype(np.int8)

    df["cust_feat_account_age_days"] = (cutoff - pd.to_datetime(df["ns_date_created"], errors="coerce")).dt.days
    df["cust_feat_days_since_last_activity"] = (cutoff - pd.to_datetime(df["ns_last_modified"], errors="coerce")).dt.days

    df["cust_cat_territory_standardized"] = df["ns_standardized_territory"]
    df["cust_cat_territory_name"] = df["ns_territory_name"]
    df["cust_cat_region"] = df["ns_region"]
    df["cust_cat_lead_source"] = df["ns_lead_source_name"].fillna(df["ns_lead_source"])
    df["cust_cat_terms_value"] = df["ns_terms_value"]

    for flag, out in [
        ("ns_taxable", "cust_feat_taxable"),
        ("ns_known_competitor", "cust_feat_known_competitor"),
        ("ns_cad_named_account", "cust_feat_named_account_cad"),
        ("ns_sim_named_account", "cust_feat_named_account_sim"),
        ("ns_am_named_account", "cust_feat_named_account_am"),
        ("ns_electrical_named_account", "cust_feat_named_account_electrical"),
    ]:
        if flag in df.columns:
            if flag == "ns_known_competitor":
                df[out] = df[flag].notna().astype(np.int8)
            else:
                df[out] = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(np.int8)

    keep = [
        "customer_id",
        "cutoff_date",
        "cust_feat_account_age_days",
        "cust_feat_days_since_last_activity",
        "cust_feat_taxable",
        "cust_feat_known_competitor",
        "cust_feat_named_account_cad",
        "cust_feat_named_account_sim",
        "cust_feat_named_account_am",
        "cust_feat_named_account_electrical",
        "cust_cat_lead_source",
        "cust_cat_terms_value",
        "cust_cat_territory_standardized",
        "cust_cat_territory_name",
        "cust_cat_region",
    ]
    out = pl.from_pandas(df[keep])

    # Persist optional artifact for inspection
    try:
        out_dir = OUTPUTS_DIR / "features"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.write_parquet(out_dir / f"ns_customer_features_{cutoff.strftime('%Y%m%d')}.parquet")
    except Exception:
        pass

    return out


__all__ = ["build_ns_customer_features", "load_enriched_customers"]

