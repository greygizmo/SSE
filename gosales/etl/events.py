from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
from pathlib import Path

import pandas as pd
import polars as pl

from gosales.utils.db import get_curated_connection
from gosales.utils.paths import OUTPUTS_DIR, DATA_DIR
from gosales.utils.logger import get_logger
from gosales.etl.sku_map import get_model_targets
from gosales.utils.identifiers import normalize_identifier_series


logger = get_logger(__name__)


DEFAULT_MODELS: Tuple[str, ...] = (
    "SWX_Seats",
    "PDM_Seats",
    "Printers",
    "Services",
    "Success_Plan",
    "Training",
    "Simulation",
    "Scanning",
    "CAMWorks",
    "SW_Electrical",
    "SW_Inspection",
)


def _read_facts(curated_engine) -> pd.DataFrame:
    cols = [
        "customer_id",
        "order_date",
        "invoice_id",
        "product_sku",
        "product_division",
        "gross_profit",
        "quantity",
    ]
    sql = f"SELECT {', '.join(cols)} FROM fact_transactions"
    df = pd.read_sql(sql, curated_engine)
    if not df.empty:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        if "customer_id" in df.columns:
            df["customer_id"] = normalize_identifier_series(df["customer_id"])
        if "invoice_id" in df.columns:
            df["invoice_id"] = normalize_identifier_series(df["invoice_id"])
        for c in ("product_sku", "product_division"):
            if c in df.columns:
                df[c] = df[c].astype(str)
        for c in ("gross_profit", "quantity"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df = df.dropna(subset=["customer_id", "order_date"]).reset_index(drop=True)
    return df


def build_fact_events(curated_engine=None, models: Iterable[str] = DEFAULT_MODELS) -> pl.DataFrame:
    """Create invoice-level events with labels per target model.

    - Groups line items into an "event" by (invoice_id, customer_id, order_date).
    - For each requested model, computes a binary label indicating presence of any
      target SKU within the invoice, plus simple aggregates (gp, qty).
    - Writes `fact_events` to curated DB and returns the DataFrame.
    """
    curated_engine = curated_engine or get_curated_connection()
    facts = _read_facts(curated_engine)
    if facts.empty:
        logger.warning("No fact_transactions available; skipping eventization.")
        return pl.DataFrame()

    # Ensure invoice_id exists; if not, synthesize a surrogate from order_date+customer
    if "invoice_id" not in facts.columns:
        facts["invoice_id"] = (
            facts["customer_id"].fillna("UNK").astype(str)
            + "|"
            + pd.to_datetime(facts["order_date"]).dt.strftime("%Y%m%d")
        )
    else:
        missing_mask = facts["invoice_id"].isna() | (facts["invoice_id"] == "")
        if missing_mask.any():
            surrogate_keys = (
                facts.loc[missing_mask, "customer_id"].fillna("UNK").astype(str)
                + "|"
                + facts.loc[missing_mask, "order_date"].dt.strftime("%Y%m%d")
            )
            counts = surrogate_keys.groupby(surrogate_keys).cumcount()
            facts.loc[missing_mask, "invoice_id"] = (
                "AUTO-" + surrogate_keys.fillna("UNK") + "-" + (counts + 1).astype(str)
            )

    # Base event grain
    base = (
        facts[["invoice_id", "customer_id", "order_date"]]
        .dropna(subset=["customer_id", "order_date"])  # tolerate missing invoice ids
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Labels per model
    lab_frames = []
    for model in models:
        targets = list(get_model_targets(model))
        if not targets:
            continue
        mask = facts["product_sku"].isin(targets)
        sub = facts.loc[mask, ["invoice_id", "customer_id", "order_date", "quantity", "gross_profit"]].copy()
        group_cols = ["invoice_id", "customer_id", "order_date"]
        agg = (
            sub.groupby(group_cols, dropna=False)
            .agg({"quantity": "sum", "gross_profit": "sum"})
            .reset_index()
            .rename(
                columns={
                    "quantity": f"qty_{model}",
                    "gross_profit": f"gp_{model}",
                }
            )
        )
        # Label: any positive quantity or non-zero GP
        agg[f"label_{model}"] = (
            (agg[f"qty_{model}"] > 0) | (agg[f"gp_{model}"] != 0)
        ).astype("int8")
        lab_frames.append(agg)

    events = base
    group_cols = ["invoice_id", "customer_id", "order_date"]
    for lf in lab_frames:
        events = events.merge(lf, on=group_cols, how="left")

    # Fill NaNs for missing models with zeros
    for col in events.columns:
        if col.startswith("qty_") or col.startswith("gp_") or col.startswith("label_"):
            events[col] = pd.to_numeric(events[col], errors="coerce").fillna(0)

    ev = pl.from_pandas(events)
    try:
        ev.write_database("fact_events", curated_engine, if_table_exists="replace")
    except Exception as e:
        logger.warning(f"Failed to write fact_events to curated DB: {e}")

    # Persist sample for inspection
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        ev.head(100).to_pandas().to_csv(OUTPUTS_DIR / "fact_events_head.csv", index=False)
    except Exception:
        pass

    logger.info(f"Built fact_events with {len(ev)} events and {len(lab_frames)} model labels.")
    return ev


if __name__ == "__main__":
    # Simple manual run helper
    eng = get_curated_connection()
    build_fact_events(eng)

