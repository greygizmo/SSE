"""Utilities for sourcing holdout outcomes for validation and gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from gosales.utils.paths import DATA_DIR
from gosales.etl.sku_map import get_sku_mapping
from gosales.utils.db import get_db_connection, validate_connection


@dataclass
class HoldoutData:
    buyers: Optional[pd.Series]
    realized_gp: Optional[pd.DataFrame]
    source: Optional[str]


def _resolve_holdout_source(cfg, source_override: Optional[str]) -> str:
    try:
        cfg_source = getattr(getattr(cfg, "validation", object()), "holdout_source", "auto")
    except Exception:
        cfg_source = "auto"
    src = (source_override or cfg_source or "auto").strip().lower()
    if src not in {"auto", "db", "csv"}:
        return "auto"
    return src


def _resolve_db_object(cfg) -> str:
    try:
        override = getattr(getattr(cfg, "validation", object()), "holdout_db_object", None)
        if override:
            return str(override).strip()
    except Exception:
        pass
    try:
        table = getattr(getattr(cfg, "database", object()), "source_tables", {}).get("sales_log", "")
        if table:
            return str(table).strip()
    except Exception:
        pass
    return "dbo.saleslog"


def _resolve_source_columns(cfg) -> dict[str, str]:
    try:
        cols = getattr(getattr(cfg, "etl", object()), "source_columns", {}) or {}
        return {str(k): str(v) for k, v in cols.items()}
    except Exception:
        return {}


def load_holdout_buyers(
    cfg,
    division: str,
    cutoff_dt: pd.Timestamp,
    window_months: int,
    *,
    source_override: Optional[str] = None,
) -> HoldoutData:
    """Return holdout buyer ids (and optional realized GP) for a division/window."""

    src_pref = _resolve_holdout_source(cfg, source_override)
    window_end = cutoff_dt + pd.DateOffset(months=window_months)

    def _from_db() -> HoldoutData:
        try:
            engine = get_db_connection()
        except Exception:
            return HoldoutData(None, None, None)
        if not validate_connection(engine):
            return HoldoutData(None, None, None)

        obj = _resolve_db_object(cfg)
        cols = _resolve_source_columns(cfg)
        cust_col = cols.get("customer_id", "CustomerId")
        date_col = cols.get("order_date", "Rec_Date")
        div_col = cols.get("division", "Division")

        sql = (
            f"SELECT {cust_col} AS customer_id, {date_col} AS rec_date "
            f"FROM {obj} "
            f"WHERE {date_col} > :cutoff AND {date_col} <= :window_end "
            f"AND LOWER(LTRIM(RTRIM(CAST({div_col} AS NVARCHAR(255))))) = :div_lower"
        )
        try:
            df = pd.read_sql_query(
                sql,
                engine,
                params={
                    "cutoff": cutoff_dt,
                    "window_end": window_end,
                    "div_lower": division.lower(),
                },
            )
        except Exception:
            return HoldoutData(None, None, None)

        if df.empty or "customer_id" not in df.columns:
            return HoldoutData(None, None, "db")
        buyers = pd.to_numeric(df["customer_id"], errors="coerce").dropna().astype("Int64")
        if buyers.empty:
            return HoldoutData(None, None, "db")
        return HoldoutData(buyers.drop_duplicates(), None, "db")

    def _from_csv() -> HoldoutData:
        holdout_dir = DATA_DIR / "holdout"
        if not holdout_dir.exists():
            return HoldoutData(None, None, None)
        parts: list[pd.DataFrame] = []
        for p in holdout_dir.glob("*.csv"):
            try:
                parts.append(pd.read_csv(p, dtype=str, low_memory=False))
            except Exception:
                continue
        if not parts:
            return HoldoutData(None, None, None)
        df = pd.concat(parts, ignore_index=True)
        if "Rec Date" in df.columns:
            df["Rec Date"] = pd.to_datetime(df["Rec Date"], errors="coerce")
            mask_window = (df["Rec Date"] > cutoff_dt) & (df["Rec Date"] <= window_end)
        else:
            mask_window = pd.Series(True, index=df.index)
        div_col = "Division" if "Division" in df.columns else None
        if div_col:
            mask_div = df[div_col].astype(str).str.strip().str.casefold() == division.lower()
        else:
            mask_div = pd.Series(True, index=df.index)
        cust_col = "CustomerId" if "CustomerId" in df.columns else "customer_id"
        buyers = pd.to_numeric(df.loc[mask_window & mask_div, cust_col], errors="coerce")
        buyers = buyers.dropna().astype("Int64")

        holdout_gp_map = None
        try:
            mapping = get_sku_mapping()
            div_cols = [gp for gp, meta in mapping.items() if meta.get("division", "").strip().lower() == division.lower()]
            div_cols = [c for c in div_cols if c in df.columns]
            if div_cols:
                gp_df = df.loc[mask_window, [cust_col] + div_cols].copy()
                for c in div_cols:
                    gp_df[c] = pd.to_numeric(gp_df[c], errors="coerce").fillna(0.0)
                gp_df["holdout_gp"] = gp_df[div_cols].sum(axis=1)
                holdout_gp_map = gp_df.groupby(cust_col)["holdout_gp"].sum().reset_index()
                holdout_gp_map[cust_col] = pd.to_numeric(holdout_gp_map[cust_col], errors="coerce").astype("Int64")
        except Exception:
            holdout_gp_map = None

        if buyers.empty:
            return HoldoutData(None, holdout_gp_map, "csv")
        return HoldoutData(buyers.drop_duplicates(), holdout_gp_map, "csv")

    if src_pref in {"auto", "db"}:
        data = _from_db()
        if data.buyers is not None and not data.buyers.empty:
            return data
        if src_pref == "db":
            return data

    if src_pref in {"auto", "csv"}:
        data = _from_csv()
        if data.buyers is not None and not data.buyers.empty:
            return data
        return data

    return HoldoutData(None, None, None)
