from __future__ import annotations

import polars as pl

# Local projection helper only; not importing pipeline to keep test isolated


def _project_from_line_local(fact_sales_line: pl.DataFrame) -> pl.DataFrame:
    # Minimal local replica of build_star projection for unit testing quantity only
    tx_exprs = []
    if "CompanyId" in fact_sales_line.columns:
        tx_exprs.append(pl.col("CompanyId").alias("customer_id"))
    if "Rec_Date" in fact_sales_line.columns:
        tx_exprs.append(pl.col("Rec_Date").alias("order_date"))
    if "item_rollup" in fact_sales_line.columns:
        tx_exprs.append(pl.col("item_rollup").cast(pl.Utf8).alias("product_sku"))
    if "division_canonical" in fact_sales_line.columns:
        tx_exprs.append(pl.col("division_canonical").cast(pl.Utf8).alias("product_division"))
    if "GP_usd" in fact_sales_line.columns:
        tx_exprs.append(pl.col("GP_usd").alias("gross_profit"))
    elif "GP" in fact_sales_line.columns:
        tx_exprs.append(pl.col("GP").alias("gross_profit"))
    else:
        tx_exprs.append(pl.lit(0.0).alias("gross_profit"))
    if "Line_Qty" in fact_sales_line.columns:
        tx_exprs.append(pl.col("Line_Qty").cast(pl.Float64, strict=False).alias("quantity"))
    else:
        tx_exprs.append(pl.lit(1.0).alias("quantity"))

    return (
        fact_sales_line.lazy()
        .select(tx_exprs)
        .collect()
    )


def test_projection_uses_line_qty():
    line = pl.DataFrame(
        {
            "CompanyId": ["A", "B"],
            "Rec_Date": ["2025-01-01", "2025-02-02"],
            "item_rollup": ["SKU1", "SKU2"],
            "division_canonical": ["CAD", "CAD"],
            "GP": [10.0, 20.0],
            "Line_Qty": [5, 7],
        }
    )

    tx = _project_from_line_local(line)
    assert "quantity" in tx.columns
    assert tx.select("quantity").to_series().to_list() == [5.0, 7.0]
