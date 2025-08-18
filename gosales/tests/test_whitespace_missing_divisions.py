import pandas as pd
import polars as pl
from sqlalchemy import create_engine

from gosales.pipeline.score_customers import generate_whitespace_opportunities


def test_generate_whitespace_opportunities_skips_missing_divisions():
    engine = create_engine("sqlite:///:memory:")

    transactions_df = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 3],
            "product_division": ["A", None, "", "B"],
            "gross_profit": [100, 200, 150, 250],
        }
    )
    customers_df = pd.DataFrame({"customer_id": [1, 2, 3]})

    transactions_df.to_sql("fact_transactions", engine, index=False)
    customers_df.to_sql("dim_customer", engine, index=False)

    ws_df = generate_whitespace_opportunities(engine)
    missing = ws_df.filter(
        pl.col("whitespace_division").is_null()
        | (pl.col("whitespace_division") == "")
    )
    assert missing.height == 0

