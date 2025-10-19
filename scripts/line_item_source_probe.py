"""Utility to capture metadata and coverage diagnostics for new line-item sources.

This script executes the non-destructive SQL probes documented in
``docs/migration_line_item_table_interrogation.md`` and persists their outputs
under ``docs/appendices/migration_line_item/`` (configurable). It helps maintain an
auditable record of the schema discovery work without repeatedly hammering the
data warehouse.

Usage example::

    PYTHONPATH="$PWD" python scripts/line_item_source_probe.py \
        --sales-table "[dbo].[table_saleslog_detail]" \
        --product-table "[dbo].[table_Product_Info_cleaned_headers]" \
        --tags-table "[dbo].[analytics_product_tags]"

The script relies on ``gosales.utils.db.get_db_connection``, so ensure the
appropriate ``AZSQL_*`` environment variables (or SQLite fallbacks) are configured
before running. All queries are scoped to metadata or bounded time windows to
remain lightweight; adjust the CLI options if the warehouse requires different
limits.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import click
import pandas as pd

from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import ROOT_DIR


logger = get_logger(__name__)


def _write_table(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{ts}__{stem}.csv"
    df.to_csv(path, index=False)
    logger.info("Wrote %s rows to %s", len(df), path)
    return path


def _object_id_query(table_name: str) -> str:
    return f"""
SELECT c.name AS column_name,
       t.name AS data_type,
       c.max_length,
       c.is_nullable,
       c.column_id
FROM sys.columns c
JOIN sys.types t ON c.user_type_id = t.user_type_id
WHERE c.object_id = OBJECT_ID('{table_name}')
ORDER BY c.column_id;
"""


def _run_query(engine, query: str, params: dict | tuple | None = None) -> pd.DataFrame:
    logger.debug("Executing query: %s", query.replace("\n", " "))
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def _quote_identifier(identifier: str | None) -> str:
    if identifier is None or identifier == "":
        raise ValueError("Identifier cannot be empty when quoting.")
    stripped = identifier.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    return f"[{stripped}]"


@click.command()
@click.option(
    "--sales-table",
    default="[dbo].[table_saleslog_detail]",
    show_default=True,
    help="Fully-qualified table name for line-item sales detail.",
)
@click.option(
    "--product-table",
    default="[dbo].[table_Product_Info_cleaned_headers]",
    show_default=True,
    help="Fully-qualified table name for product metadata.",
)
@click.option(
    "--tags-table",
    default="[dbo].[analytics_product_tags]",
    show_default=True,
    help="Fully-qualified table name for division tags (Goal).",
)
@click.option(
    "--tag-product-column",
    default="item_rollup",
    show_default=True,
    help="Column in the tags table identifying the product/item key.",
)
@click.option(
    "--date-column",
    default="Rec_Date",
    show_default=True,
    help="Date column in the sales detail table.",
)
@click.option(
    "--order-column",
    default="Sales_Order",
    show_default=True,
    help="Order identifier column in the sales detail table.",
)
@click.option(
    "--item-column",
    default="Item_internalid",
    show_default=True,
    help="Item identifier column in the sales detail table (set blank to skip item-level duplicate checks).",
)
@click.option(
    "--customer-column",
    default="CompanyId",
    show_default=True,
    help="Customer identifier column used for distinct counts.",
)
@click.option(
    "--amount-column",
    default="Revenue",
    show_default=True,
    help="Column representing monetary line amount.",
)
@click.option(
    "--last-update-column",
    default="last_update",
    show_default=True,
    help="Timestamp column used for dedupe diagnostics.",
)
@click.option(
    "--product-join-column",
    default="internalid",
    show_default=True,
    help="Column in the product table used to join to the sales detail item column.",
)
@click.option(
    "--legacy-table",
    default="[dbo].[saleslog]",
    show_default=True,
    help="Legacy header-level view/table used for parity comparisons.",
)
@click.option(
    "--legacy-order-column",
    default="SalesOrderId",
    show_default=True,
    help="Order identifier column in the legacy table.",
)
@click.option(
    "--legacy-amount-column",
    default="Revenue",
    show_default=True,
    help="Revenue/amount column in the legacy table.",
)
@click.option(
    "--legacy-date-column",
    default="Rec_Date",
    show_default=True,
    help="Date column in the legacy table.",
)
@click.option(
    "--customer-rollup-table",
    default="[dbo].[Customer_asset_rollups]",
    show_default=True,
    help="Optional table name for customer asset rollups (used as validator).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=ROOT_DIR / "docs" / "appendices" / "migration_line_item",
    show_default=True,
    help="Directory to store query outputs (CSV).",
)
@click.option(
    "--monthly-window",
    default=24,
    show_default=True,
    help="Number of trailing months for the monthly line count probe.",
)
@click.option(
    "--distinct-window",
    default=12,
    show_default=True,
    help="Number of trailing months for the distinct entity probe.",
)
@click.option(
    "--sample-top",
    default=200,
    show_default=True,
    help="Row limit for joinability and sampling queries.",
)
@click.option(
    "--include-customer-rollup/--skip-customer-rollup",
    default=True,
    show_default=True,
    help="Whether to run the optional customer asset rollup sample.",
)
def main(
    sales_table: str,
    product_table: str,
    tags_table: str,
    tag_product_column: str,
    date_column: str,
    order_column: str,
    item_column: str,
    customer_column: str,
    amount_column: str,
    last_update_column: str,
    product_join_column: str,
    legacy_table: str,
    legacy_order_column: str,
    legacy_amount_column: str,
    legacy_date_column: str,
    customer_rollup_table: str,
    output_dir: Path,
    monthly_window: int,
    distinct_window: int,
    sample_top: int,
    include_customer_rollup: bool,
):
    """Run metadata queries against the specified tables and persist CSV outputs."""

    logger.info(
        "Starting line-item source interrogation",
        extra={
            "sales_table": sales_table,
            "product_table": product_table,
            "tags_table": tags_table,
            "date_column": date_column,
            "order_column": order_column,
            "item_column": item_column,
            "amount_column": amount_column,
            "tag_product_column": tag_product_column,
            "monthly_window": monthly_window,
            "distinct_window": distinct_window,
            "sample_top": sample_top,
        },
    )

    engine = get_db_connection()

    date_col = _quote_identifier(date_column)
    order_col = _quote_identifier(order_column)
    item_col = _quote_identifier(item_column) if item_column else None
    customer_col = _quote_identifier(customer_column)
    amount_col = _quote_identifier(amount_column)
    last_update_col = _quote_identifier(last_update_column)
    product_join_col = _quote_identifier(product_join_column)
    tag_product_col = _quote_identifier(tag_product_column)
    legacy_order_col = _quote_identifier(legacy_order_column)
    legacy_amount_col = _quote_identifier(legacy_amount_column)
    legacy_date_col = _quote_identifier(legacy_date_column)

    # 1) Column metadata for each table
    sales_meta = _run_query(engine, _object_id_query(sales_table))
    _write_table(sales_meta, output_dir, "sales_detail_columns")

    product_meta = _run_query(engine, _object_id_query(product_table))
    _write_table(product_meta, output_dir, "product_info_columns")

    tags_meta = _run_query(engine, _object_id_query(tags_table))
    _write_table(tags_meta, output_dir, "product_tags_columns")

    # 2) Candidate key checks (sales line uniqueness)
    group_cols = order_col if item_col is None else f"{order_col}, {item_col}"
    duplicates_query = f"""
SELECT TOP (1) 1 AS has_duplicates
FROM (
  SELECT {group_cols},
         COUNT(*) AS cnt
  FROM {sales_table} WITH (NOLOCK)
  GROUP BY {group_cols}
  HAVING COUNT(*) > 1
) dups;
"""
    dup_df = _run_query(engine, duplicates_query)
    _write_table(dup_df, output_dir, "sales_line_duplicate_scan")

    alt_select_cols = [
        f"{order_col} AS order_id",
    ]
    if item_col:
        alt_select_cols.append(f"{item_col} AS item_key")
    alt_select_cols.append("COUNT(*) AS cnt")
    alt_select_cols.append(f"MIN({last_update_col}) AS min_last_update")
    alt_select_cols.append(f"MAX({last_update_col}) AS max_last_update")
    alt_key_query = f"""
SELECT TOP ({sample_top})
       {', '.join(alt_select_cols)}
FROM {sales_table} WITH (NOLOCK)
GROUP BY {group_cols}
HAVING COUNT(*) > 1
ORDER BY cnt DESC;
"""
    alt_df = _run_query(engine, alt_key_query)
    _write_table(alt_df, output_dir, "sales_line_duplicate_examples")

    # 3) Division tag multiplicity
    multiplicity_query = f"""
SELECT {tag_product_col} AS product_key,
       COUNT(*) AS tag_rows,
       COUNT(DISTINCT Goal) AS goal_count
FROM {tags_table} WITH (NOLOCK)
GROUP BY {tag_product_col}
ORDER BY goal_count DESC, tag_rows DESC;
"""
    mult_df = _run_query(engine, multiplicity_query)
    _write_table(mult_df, output_dir, "product_tag_multiplicity")

    # 4) Monthly line counts (bounded window)
    months_back = max(monthly_window, 1)
    start_date = (dt.date.today().replace(day=1) - pd.DateOffset(months=months_back)).date()
    monthly_query = f"""
SELECT CONVERT(char(7), {date_col}, 126) AS yyyymm,
       COUNT_BIG(*) AS line_count
FROM {sales_table} WITH (NOLOCK)
WHERE {date_col} >= ?
GROUP BY CONVERT(char(7), {date_col}, 126)
ORDER BY yyyymm;
"""
    monthly_df = _run_query(engine, monthly_query, params=(start_date,))
    _write_table(monthly_df, output_dir, "sales_line_monthly_counts")

    # 5) Distinct entities in trailing window
    distinct_months = max(distinct_window, 1)
    distinct_start = (dt.date.today() - pd.DateOffset(months=distinct_months)).date()
    distinct_query = f"""
SELECT COUNT(DISTINCT {customer_col}) AS customers,
       COUNT(DISTINCT {item_col}) AS items,
       COUNT(DISTINCT {order_col}) AS sales_orders
FROM {sales_table} WITH (NOLOCK)
WHERE {date_col} >= ?;
"""
    distinct_df = _run_query(engine, distinct_query, params=(distinct_start,))
    _write_table(distinct_df, output_dir, "sales_line_distinct_entities")

    # 6) Legacy parity diagnostics (if legacy view remains reachable)
    parity_end = dt.date.today()
    parity_start = parity_end - pd.DateOffset(months=6)
    parity_query = f"""
;WITH line_header AS (
  SELECT {order_col} AS order_id,
         SUM({amount_col}) AS total_amount,
         MIN({date_col}) AS tran_date
  FROM {sales_table} WITH (NOLOCK)
  WHERE {date_col} BETWEEN ? AND ?
  GROUP BY {order_col}
),
legacy_header AS (
  SELECT {legacy_order_col} AS order_id,
         SUM({legacy_amount_col}) AS total_amount
  FROM {legacy_table} WITH (NOLOCK)
  WHERE {legacy_date_col} BETWEEN ? AND ?
  GROUP BY {legacy_order_col}
)
SELECT 'only_in_lines' AS diff_side, COUNT(*) AS n
FROM line_header lh
LEFT JOIN legacy_header s ON s.order_id = lh.order_id
WHERE s.order_id IS NULL
UNION ALL
SELECT 'only_in_saleslog', COUNT(*)
FROM legacy_header s
LEFT JOIN line_header lh ON lh.order_id = s.order_id
WHERE lh.order_id IS NULL
UNION ALL
SELECT 'mismatched_amounts', COUNT(*)
FROM legacy_header s
JOIN line_header lh ON lh.order_id = s.order_id
WHERE ABS(ISNULL(s.total_amount,0) - ISNULL(lh.total_amount,0)) > 0.01;
"""
    parity_df = _run_query(
        engine,
        parity_query,
        params=(parity_start, parity_end, parity_start, parity_end),
    )
    _write_table(parity_df, output_dir, "legacy_parity_summary")

    # 7) Division coverage stats
    coverage_query = f"""
SELECT COUNT(*) AS rows,
       SUM(CASE WHEN Goal IS NULL OR Goal = '' THEN 1 ELSE 0 END) AS null_goal
FROM {tags_table} WITH (NOLOCK);
"""
    coverage_df = _run_query(engine, coverage_query)
    _write_table(coverage_df, output_dir, "division_goal_coverage")

    goal_list_query = f"""
SELECT Goal,
       COUNT(*) AS rows
FROM {tags_table} WITH (NOLOCK)
GROUP BY Goal
ORDER BY rows DESC;
"""
    goal_list_df = _run_query(engine, goal_list_query)
    _write_table(goal_list_df, output_dir, "division_goal_inventory")

    # 8) Joinability sample
    if not item_col:
        logger.info("Item column not provided; skipping joinability sample.")
    else:
        join_query = f"""
SELECT TOP ({sample_top})
       s.{order_col} AS order_id,
       s.{item_col} AS item_key,
       p.{product_join_col} AS product_key,
       p.item_rollup,
       s.{amount_col} AS line_amount,
       s.{date_col} AS tran_date
FROM {sales_table} AS s WITH (NOLOCK)
LEFT JOIN {product_table} AS p
  ON p.{product_join_col} = s.{item_col}
WHERE s.{date_col} >= DATEADD(MONTH, -3, GETDATE())
ORDER BY s.{date_col} DESC;
"""
        join_df = _run_query(engine, join_query)
        _write_table(join_df, output_dir, "sales_vs_product_join_sample")

    # 9) Optional customer asset rollup sample
    if include_customer_rollup:
        rollup_query = f"""
SELECT TOP ({sample_top})
       car.customer_id,
       car.item_rollup,
       car.asset_count,
       car.last_purchase_date
FROM {customer_rollup_table} AS car WITH (NOLOCK)
ORDER BY car.last_purchase_date DESC;
"""
        try:
            rollup_df = _run_query(engine, rollup_query)
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.warning(
                "Skipping customer asset rollup probe due to query failure",
                extra={"error": str(exc)},
            )
        else:
            _write_table(rollup_df, output_dir, "customer_asset_rollup_sample")

    logger.info("Completed line-item source interrogation", extra={"output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
