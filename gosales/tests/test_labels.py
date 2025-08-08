from __future__ import annotations

import pandas as pd
import polars as pl
from sqlalchemy import create_engine

from gosales.labels.targets import LabelParams, build_labels_for_division


def _make_engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path}/test_labels.db")


def _seed_curated(engine):
    # Tiny curated set to exercise windowing/positives/returns
    fact = pd.DataFrame([
        # pre-cutoff activity for cust 1 and 2
        {"customer_id": 1, "order_date": "2024-06-01", "product_division": "Solidworks", "product_sku": "SWX_Core", "gross_profit": 100},
        {"customer_id": 2, "order_date": "2024-05-15", "product_division": "Services", "product_sku": "Training", "gross_profit": 50},
        # window positives/negatives
        {"customer_id": 1, "order_date": "2024-08-01", "product_division": "Solidworks", "product_sku": "SWX_Core", "gross_profit": 20},
        {"customer_id": 2, "order_date": "2024-07-05", "product_division": "Solidworks", "product_sku": "SWX_Core", "gross_profit": -5},  # return-only
        {"customer_id": 3, "order_date": "2024-07-10", "product_division": "Services", "product_sku": "Training", "gross_profit": 10},
    ])
    fact.to_sql("fact_transactions", engine, if_exists="replace", index=False)
    dim = pd.DataFrame({"customer_id": [1, 2, 3]})
    dim.to_sql("dim_customer", engine, if_exists="replace", index=False)


def test_build_labels_modes(tmp_path):
    eng = _make_engine(tmp_path)
    _seed_curated(eng)
    params = LabelParams(division="Solidworks", cutoff="2024-06-30", window_months=6, mode="expansion", gp_min_threshold=0.0)
    df = build_labels_for_division(eng, params)
    pdf = df.to_pandas()
    # expansion should include cust 1 and 2 (have pre-cutoff history), not 3
    assert set(pdf["customer_id"]) == {1, 2}
    assert int(pdf.loc[pdf["customer_id"] == 1, "label"].iloc[0]) == 1
    assert int(pdf.loc[pdf["customer_id"] == 2, "label"].iloc[0]) == 0

    params_all = LabelParams(division="Solidworks", cutoff="2024-06-30", window_months=6, mode="all", gp_min_threshold=0.0)
    df_all = build_labels_for_division(eng, params_all)
    assert set(df_all.to_pandas()["customer_id"]) == {1, 2, 3}


def test_censoring_flag(tmp_path):
    eng = _make_engine(tmp_path)
    _seed_curated(eng)
    params = LabelParams(division="Solidworks", cutoff="2024-06-30", window_months=12, mode="all", gp_min_threshold=0.0)
    df = build_labels_for_division(eng, params)
    pdf = df.to_pandas()
    assert "censored_flag" in pdf.columns
    assert int(pdf["censored_flag"].iloc[0]) in (0, 1)


