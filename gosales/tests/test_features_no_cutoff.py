from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine
import polars as pl

from gosales.features.engine import create_feature_matrix


def _seed_no_cutoff(engine):
    fact = pd.DataFrame(
        [
            {"customer_id": 1, "order_date": "2024-01-01", "product_division": "Solidworks", "product_sku": "SWX_Core", "gross_profit": 100, "quantity": 1},
            {"customer_id": 1, "order_date": "2024-02-01", "product_division": "Services", "product_sku": "Training", "gross_profit": 5, "quantity": 1},
            {"customer_id": 2, "order_date": "2023-12-15", "product_division": "Simulation", "product_sku": "Simulation", "gross_profit": 50, "quantity": 1},
        ]
    )
    fact.to_sql("fact_transactions", engine, if_exists="replace", index=False)
    pd.DataFrame({"customer_id": [1, 2]}).to_sql("dim_customer", engine, if_exists="replace", index=False)


def test_feature_matrix_without_cutoff_returns_rows(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/test_features_no_cutoff.db")
    _seed_no_cutoff(eng)
    fm = create_feature_matrix(eng, "Solidworks", cutoff_date=None, prediction_window_months=1)
    assert isinstance(fm, pl.DataFrame)
    assert not fm.is_empty()
    assert "customer_id" in fm.columns
