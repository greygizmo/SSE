from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import polars as pl
from dateutil.relativedelta import relativedelta

from gosales.utils.db import get_curated_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProspectLabelConfig:
    """Configuration parameters for SolidWorks (CRE) prospect labels."""

    cutoff_dates: Sequence[str]
    window_months: int = 6


def _normalize_ns_snapshot(df: pl.DataFrame) -> pl.DataFrame:
    ts_cols = [
        "ns_date_created",
        "ns_last_modified",
        "ns_first_cre_date",
    ]
    existing = [col for col in ts_cols if col in df.columns]
    if existing:
        df = df.with_columns([
            pl.col(col).cast(pl.Utf8, strict=False).str.strptime(pl.Datetime, strict=False).dt.date()
            for col in existing
        ])
    return df


def load_ns_customer_snapshot(engine=None) -> pl.DataFrame:
    engine = engine or get_curated_connection()
    query = """
        SELECT
            internalid AS customer_id,
            ns_account_type,
            ns_is_inactive,
            ns_stage,
            ns_stage_value,
            ns_entity_status_value,
            ns_standardized_territory,
            ns_territory_name,
            ns_region,
            ns_date_created,
            ns_last_modified,
            ns_first_cre_date
        FROM dim_ns_customer
    """
    df = pl.read_database(query, engine)
    return _normalize_ns_snapshot(df)


def prospect_filter_expr(cutoff: datetime) -> pl.Expr:
    """Expression selecting active SolidWorks prospects at a cutoff."""

    has_cre_purchase = (
        pl.col("ns_first_cre_date").is_not_null()
        & (pl.col("ns_first_cre_date") <= pl.lit(cutoff.date()))
    )

    stage_text = pl.when(pl.col("ns_stage_value").is_not_null()).then(pl.col("ns_stage_value")).otherwise(pl.col("ns_stage"))
    stage_is_customer = stage_text.str.to_lowercase().str.contains("customer", literal=False).fill_null(False)
    status_is_customer = pl.col("ns_entity_status_value").str.to_lowercase().str.contains("customer", literal=False).fill_null(False)
    is_inactive = pl.col("ns_is_inactive").cast(pl.Boolean, strict=False).fill_null(False)

    return (~has_cre_purchase) & (~stage_is_customer) & (~status_is_customer) & (~is_inactive)


def build_labels(config: ProspectLabelConfig, engine=None) -> pl.DataFrame:
    engine = engine or get_curated_connection()
    base = load_ns_customer_snapshot(engine)

    if not config.cutoff_dates:
        raise ValueError("cutoff_dates must not be empty")

    results: list[pl.DataFrame] = []
    for cutoff_str in config.cutoff_dates:
        cutoff = datetime.fromisoformat(cutoff_str).replace(hour=0, minute=0, second=0, microsecond=0)
        horizon_end = cutoff + relativedelta(months=config.window_months)

        cohort = base.filter(prospect_filter_expr(cutoff))

        positive_expr = (
            pl.col("ns_first_cre_date").is_not_null()
            & (pl.col("ns_first_cre_date") > pl.lit(cutoff.date()))
            & (pl.col("ns_first_cre_date") <= pl.lit(horizon_end.date()))
        )

        snapshot = cohort.select([
            pl.col("customer_id"),
            pl.lit(cutoff.date()).alias("cutoff_date"),
            pl.lit(horizon_end.date()).alias("horizon_end"),
            pl.col("ns_standardized_territory"),
            pl.col("ns_territory_name"),
            pl.col("ns_region"),
            pl.when(positive_expr).then(1).otherwise(0).alias("label"),
            pl.when(positive_expr).then(pl.col("ns_first_cre_date")).otherwise(None).alias("first_cre_date"),
        ])
        results.append(snapshot)

    labels = pl.concat(results)

    out_dir = OUTPUTS_DIR / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solidworks_prospect_labels.parquet"
    labels.write_parquet(out_path)
    logger.info("Wrote SolidWorks prospect labels to %s", out_path)

    return labels


def default_cutoffs(months_back: int = 24) -> list[str]:
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    end = today.replace(day=1)
    dates: list[str] = []
    current = end
    for _ in range(months_back):
        current = current - relativedelta(months=1)
        last_day = (current + relativedelta(months=1)) - relativedelta(days=1)
        dates.append(last_day.date().isoformat())
    dates.reverse()
    return dates


__all__ = [
    "ProspectLabelConfig",
    "build_labels",
    "default_cutoffs",
    "load_ns_customer_snapshot",
    "prospect_filter_expr",
]
