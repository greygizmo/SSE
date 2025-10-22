"""Compute revenue/GP parity deltas between legacy Sales Log (deprecated) and line-item facts.

This validation aggregates both sources at the division level, compares totals,
and emits CSV/JSON artifacts under ``gosales/outputs/validation/line_item_parity``.
It is intended to be run whenever the line-item fact or division mapping logic
changes so that downstream stakeholders can review deterministic deltas before
re-enabling modeling phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, Optional

import click
import pandas as pd
import polars as pl
from sqlalchemy import text

from gosales.utils.config import load_config
from gosales.utils.db import get_db_connection, get_curated_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.sql import ensure_allowed_identifier, validate_identifier
from gosales.sql.queries import select_all


logger = get_logger(__name__)

DivisionMetricFrame = pl.DataFrame


@dataclass(slots=True)
class ParityResult:
    frame: DivisionMetricFrame
    summary: dict[str, object]


def _determine_metric_column(frame: DivisionMetricFrame, candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in frame.columns:
            return col
    raise ValueError(f"None of the candidate columns {candidates} exist in the frame; available={frame.columns}")


def _prepare_division_metrics(
    frame: DivisionMetricFrame,
    *,
    division_column: str,
    revenue_candidates: Iterable[str],
    gp_candidates: Iterable[str],
    count_column_name: str = "row_count",
    date_column: str | None = None,
    cutoff: str | None = None,
) -> DivisionMetricFrame:
    if frame.is_empty():
        return pl.DataFrame(
            {
                "division": pl.Series([], dtype=pl.Utf8),
                "revenue": pl.Series([], dtype=pl.Float64),
                "gross_profit": pl.Series([], dtype=pl.Float64),
                count_column_name: pl.Series([], dtype=pl.Int64),
            }
        )

    division_col = division_column if division_column in frame.columns else None
    if division_col is None:
        raise ValueError(f"Division column `{division_column}` not found in source frame (columns={frame.columns}).")

    revenue_col = _determine_metric_column(frame, revenue_candidates)
    gp_col = _determine_metric_column(frame, gp_candidates)

    prepared = frame
    if date_column and cutoff and date_column in prepared.columns:
        try:
            cutoff_dt = datetime.fromisoformat(cutoff)
        except ValueError as exc:
            raise ValueError(f"Invalid cutoff `{cutoff}`. Expected ISO date like '2025-06-30'.") from exc

        date_dtype = prepared.schema.get(date_column)
        date_expr = pl.col(date_column)
        if date_dtype != pl.Datetime:
            date_expr = date_expr.cast(pl.Utf8, strict=False).str.strptime(pl.Datetime, strict=False)

        prepared = prepared.with_columns(
            date_expr.alias(date_column)
        ).filter(pl.col(date_column).is_not_null() & (pl.col(date_column) <= cutoff_dt))

    prepared = prepared.with_columns(
        pl.coalesce([pl.col(division_col).cast(pl.Utf8), pl.lit("UNKNOWN")])
        .str.strip_chars()
        .str.to_uppercase()
        .alias("division"),
        pl.col(revenue_col).cast(pl.Float64, strict=False).fill_null(0.0).alias("revenue"),
        pl.col(gp_col).cast(pl.Float64, strict=False).fill_null(0.0).alias("gross_profit"),
    )

    aggregated = (
        prepared.group_by("division", maintain_order=False)
        .agg(
            [
                pl.col("revenue").sum().alias("revenue"),
                pl.col("gross_profit").sum().alias("gross_profit"),
                pl.len().alias(count_column_name),
            ]
        )
        .sort("division")
    )

    return aggregated


def _read_sql_as_polars(engine, query: str, params: Optional[dict[str, object]] = None) -> DivisionMetricFrame:
    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn, params=params)
    if df.empty:
        return pl.DataFrame()
    return pl.from_pandas(df, include_index=False)


def _load_line_fact_metrics(curated_engine, cutoff: str | None) -> DivisionMetricFrame:
    columns = [
        "division_canonical",
        "division_goal",
        "Division",
        "Revenue_usd",
        "Revenue",
        "GP_usd",
        "GP",
        "Rec_Date",
    ]
    try:
        query = f"SELECT {', '.join(columns)} FROM fact_sales_line"
        frame = _read_sql_as_polars(curated_engine, query)
    except Exception as exc:
        logger.warning("Unable to read fact_sales_line: %s", exc)
        return pl.DataFrame()

    if frame.is_empty():
        return frame

    revenue_candidates = ["Revenue_usd", "Revenue"]
    gp_candidates = ["GP_usd", "GP"]

    if "division_canonical" not in frame.columns:
        # For back-compat runs during migration
        frame = frame.with_columns(
            pl.coalesce(
                [pl.col("division_goal").cast(pl.Utf8), pl.col("Division").cast(pl.Utf8), pl.lit("UNKNOWN")]
            ).alias("division_canonical")
        )

    return _prepare_division_metrics(
        frame,
        division_column="division_canonical",
        revenue_candidates=revenue_candidates,
        gp_candidates=gp_candidates,
        count_column_name="line_count",
        date_column="Rec_Date",
        cutoff=cutoff,
    )


def _load_legacy_metrics(raw_engine, table_name: str, cutoff: str | None, allowed_identifiers: set[str]) -> DivisionMetricFrame:
    if not table_name:
        logger.warning("Legacy sales log table name not provided; skipping legacy comparison.")
        return pl.DataFrame()

    try:
        if allowed_identifiers:
            ensure_allowed_identifier(table_name, allowed_identifiers)
        else:
            validate_identifier(table_name)
    except Exception as exc:
        raise ValueError(f"Legacy table `{table_name}` failed identifier validation: {exc}") from exc

    query = select_all(table_name, allowlist=allowed_identifiers or None)
    frame = _read_sql_as_polars(raw_engine, query)
    if frame.is_empty():
        logger.warning("Legacy table `%s` returned zero rows.", table_name)
        return frame

    date_col = None
    for candidate in ("Rec_Date", "tran_date", "Invoice_Date", "order_date"):
        if candidate in frame.columns:
            date_col = candidate
            break

    division_candidates = ("division_canonical", "product_division", "division", "Division")
    division_col = next((col for col in division_candidates if col in frame.columns), None)
    if division_col is None:
        raise ValueError(
            f"Legacy table `{table_name}` missing division column; expected one of {division_candidates} (columns={frame.columns})."
        )

    revenue_candidates = ["Revenue_usd", "Revenue", "total_amount", "line_amount"]
    gp_candidates = ["GP_usd", "GP", "gross_profit"]

    return _prepare_division_metrics(
        frame,
        division_column=division_col,
        revenue_candidates=revenue_candidates,
        gp_candidates=gp_candidates,
        count_column_name="legacy_count",
        date_column=date_col,
        cutoff=cutoff,
    )


def _compare_metrics(
    line_metrics: DivisionMetricFrame,
    legacy_metrics: DivisionMetricFrame,
    *,
    delta_threshold: float,
) -> ParityResult:
    line_df = line_metrics.rename({"revenue": "line_revenue", "gross_profit": "line_gross_profit"})
    legacy_df = legacy_metrics.rename({"revenue": "legacy_revenue", "gross_profit": "legacy_gross_profit"})

    combined = line_df.join(legacy_df, on="division", how="full")

    combined = combined.with_columns(
        [
            pl.col("line_revenue").fill_null(0.0),
            pl.col("legacy_revenue").fill_null(0.0),
            pl.col("line_gross_profit").fill_null(0.0),
            pl.col("legacy_gross_profit").fill_null(0.0),
            pl.col("line_count").fill_null(0),
            pl.col("legacy_count").fill_null(0),
        ]
    )

    combined = combined.with_columns(
        [
            (pl.col("line_revenue") - pl.col("legacy_revenue")).alias("delta_revenue"),
            pl.when(pl.col("legacy_revenue").abs() > 0.0)
            .then((pl.col("line_revenue") - pl.col("legacy_revenue")) / pl.col("legacy_revenue"))
            .otherwise(0.0)
            .alias("delta_revenue_pct"),
            (pl.col("line_gross_profit") - pl.col("legacy_gross_profit")).alias("delta_gross_profit"),
            pl.when(pl.col("legacy_gross_profit").abs() > 0.0)
            .then((pl.col("line_gross_profit") - pl.col("legacy_gross_profit")) / pl.col("legacy_gross_profit"))
            .otherwise(0.0)
            .alias("delta_gross_profit_pct"),
            (pl.col("line_count") - pl.col("legacy_count")).alias("delta_count"),
        ]
    ).select(
        [
            "division",
            "line_revenue",
            "legacy_revenue",
            "delta_revenue",
            "delta_revenue_pct",
            "line_gross_profit",
            "legacy_gross_profit",
            "delta_gross_profit",
            "delta_gross_profit_pct",
            "line_count",
            "legacy_count",
            "delta_count",
        ]
    ).sort("division")

    max_revenue_delta_pct = float(
        combined.select(pl.col("delta_revenue_pct").abs().max()).item() or 0.0
    )
    max_gp_delta_pct = float(
        combined.select(pl.col("delta_gross_profit_pct").abs().max()).item() or 0.0
    )

    status = "pass" if max(max_revenue_delta_pct, max_gp_delta_pct) <= delta_threshold else "fail"

    summary = {
        "status": status,
        "delta_threshold": delta_threshold,
        "max_revenue_delta_pct": max_revenue_delta_pct,
        "max_gross_profit_delta_pct": max_gp_delta_pct,
        "total_line_revenue": float(combined.select(pl.col("line_revenue").sum()).item() or 0.0),
        "total_legacy_revenue": float(combined.select(pl.col("legacy_revenue").sum()).item() or 0.0),
        "total_line_gross_profit": float(combined.select(pl.col("line_gross_profit").sum()).item() or 0.0),
        "total_legacy_gross_profit": float(combined.select(pl.col("legacy_gross_profit").sum()).item() or 0.0),
        "divisions": combined.height,
    }

    return ParityResult(frame=combined, summary=summary)


def _write_outputs(result: ParityResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "division_parity.csv"
    json_path = output_dir / "summary.json"

    result.frame.write_csv(csv_path)
    json_path.write_text(json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Parity artifacts written: %s, %s", csv_path, json_path)


@click.command()
@click.option(
    "--config",
    type=str,
    default=str((Path(__file__).parents[1] / "config.yaml").resolve()),
    help="Path to config.yaml.",
)
@click.option("--cutoff", type=str, default=None, help="Optional ISO date to filter transactions (<= cutoff).")
@click.option(
    "--delta-threshold",
    type=float,
    default=0.05,
    show_default=True,
    help="Acceptable max absolute delta (ratio) between legacy and line-item totals.",
)
@click.option(
    "--output-dir",
    type=str,
    default=str((OUTPUTS_DIR / "validation" / "line_item_parity").resolve()),
    help="Output directory for parity artifacts.",
)
@click.option(
    "--legacy-table",
    type=str,
    default=None,
    help=(
        "Optional override for legacy table (deprecated). "
        "Defaults to database.source_tables.sales_log when present; "
        "Phase 0 no longer reads Sales Log for curated builds."
    ),
)
def main(config: str, cutoff: str | None, delta_threshold: float, output_dir: str, legacy_table: str | None) -> None:
    """CLI entry point to compute division-level parity between legacy and line-item sources."""

    cfg = load_config(config)

    allowed_identifiers = set(getattr(getattr(cfg, "database", object()), "allowed_identifiers", []) or [])
    legacy_source = legacy_table or getattr(getattr(cfg, "database", object()), "source_tables", {}).get("sales_log")

    curated_engine = get_curated_connection()
    raw_engine = get_db_connection()

    line_metrics = _load_line_fact_metrics(curated_engine, cutoff)
    legacy_metrics = _load_legacy_metrics(raw_engine, legacy_source, cutoff, allowed_identifiers)

    if line_metrics.is_empty():
        raise RuntimeError("fact_sales_line produced no rows; ensure the line-item fact is built before running parity.")
    if legacy_metrics.is_empty():
        logger.warning("Legacy metrics frame is empty; parity will reflect line-only totals.")

    result = _compare_metrics(line_metrics, legacy_metrics, delta_threshold=delta_threshold)
    _write_outputs(result, Path(output_dir))

    logger.info(
        "Parity status: %s (max revenue delta pct=%.4f, max GP delta pct=%.4f)",
        result.summary["status"],
        result.summary["max_revenue_delta_pct"],
        result.summary["max_gross_profit_delta_pct"],
    )


if __name__ == "__main__":
    main()
