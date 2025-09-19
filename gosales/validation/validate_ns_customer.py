from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from gosales.utils.db import get_curated_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class NSCustomerValidationReport:
    row_count: int
    distinct_ids: int
    null_internalid: int
    customers: int
    prospects: int
    inactive: int
    stage_customer_inconsistency: int
    parsed_dates_null_pct: dict[str, float]


def _read_ns_df() -> pl.DataFrame:
    eng = get_curated_connection()
    df = pl.read_database("SELECT * FROM dim_ns_customer", eng)
    # ensure textual types for sanity checks
    text_cols = [
        c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8 or dt == pl.Null
    ]
    if text_cols:
        df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in text_cols])
    return df


def _pct_null(df: pl.DataFrame, cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in cols:
        if c in df.columns:
            out[c] = float(df.select(pl.col(c).is_null().mean()).item())
    return out


def validate_dim_ns_customer() -> NSCustomerValidationReport:
    ns = _read_ns_df()
    rows = ns.height
    distinct_ids = ns.select(pl.col("internalid").n_unique()).item()
    null_internalid = ns.select(pl.col("internalid").is_null().sum()).item()

    # account_type distribution
    customers = ns.filter(pl.col("ns_account_type") == "customer").height if "ns_account_type" in ns.columns else 0
    prospects = ns.filter(pl.col("ns_account_type") == "prospect").height if "ns_account_type" in ns.columns else 0
    inactive = ns.filter(pl.col("ns_is_inactive").cast(pl.Int8, strict=False) == 1).height if "ns_is_inactive" in ns.columns else 0

    # Inconsistency: stage/status says prospect but first CRE is present
    stage_text = (
        pl.when(pl.col("ns_stage_value").is_not_null())
        .then(pl.col("ns_stage_value"))
        .otherwise(pl.col("ns_stage"))
    )
    is_stage_customer = stage_text.str.to_lowercase().str.contains("customer", literal=False).fill_null(False)
    has_cre = pl.col("ns_first_cre_date").is_not_null()
    inconsistent = ns.filter((~is_stage_customer) & has_cre).height if "ns_first_cre_date" in ns.columns else 0

    parsed_dates_null_pct = _pct_null(ns, [
        "ns_date_created",
        "ns_last_modified",
        "ns_first_cre_date",
        "ns_first_cpe_date",
        "ns_first_hw_date",
    ])

    report = NSCustomerValidationReport(
        row_count=rows,
        distinct_ids=distinct_ids,
        null_internalid=null_internalid,
        customers=customers,
        prospects=prospects,
        inactive=inactive,
        stage_customer_inconsistency=inconsistent,
        parsed_dates_null_pct=parsed_dates_null_pct,
    )

    # Save JSON + CSV summaries
    out_dir = OUTPUTS_DIR / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dim_ns_customer_validation.json").write_text(
        json.dumps(report.__dict__, indent=2), encoding="utf-8"
    )

    # Column-wise nulls & basic stats
    basic = ns.to_pandas()
    summary = []
    for c in basic.columns:
        s = basic[c]
        summary.append({
            "column": c,
            "dtype": str(s.dtype),
            "null_pct": float(s.isna().mean()),
            "nunique": int(s.nunique(dropna=True)),
        })
    pd.DataFrame(summary).sort_values(["null_pct", "nunique"], ascending=[False, False]).to_csv(
        out_dir / "dim_ns_customer_columns.csv", index=False
    )

    logger.info("dim_ns_customer validation complete; rows=%d, customers=%d, prospects=%d", rows, customers, prospects)
    return report


if __name__ == "__main__":
    validate_dim_ns_customer()

