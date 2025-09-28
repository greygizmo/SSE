from __future__ import annotations

import pandas as pd
import polars as pl
from sqlalchemy import create_engine

from gosales.labels.targets import LabelParams, build_labels_for_division
from pathlib import Path


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


def test_denylist_threshold(tmp_path, monkeypatch):
    eng = _make_engine(tmp_path)
    _seed_curated(eng)
    # Create denylist file that excludes SWX_Core
    denylist_path = Path(tmp_path) / "deny.csv"
    pd.DataFrame({"sku": ["SWX_Core"]}).to_csv(denylist_path, index=False)

    # Monkeypatch config loader to point to temp denylist and threshold 10.0
    from gosales.utils import config as cfgmod
    orig = cfgmod.load_config
    def _fake():
        c = orig()
        c.labels.gp_min_threshold = 10.0
        c.labels.denylist_skus_csv = denylist_path
        return c
    monkeypatch.setattr(cfgmod, "load_config", _fake)

    params = LabelParams(division="Solidworks", cutoff="2024-06-30", window_months=6, mode="all", gp_min_threshold=10.0)
    df = build_labels_for_division(eng, params)
    pdf = df.to_pandas()
    # Customer 1 had +20 GP in window but SKU is denied -> label must be 0
    assert int(pdf.loc[pdf["customer_id"] == 1, "label"].iloc[0]) == 0


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

def test_build_labels_mixed_case_division(tmp_path):
    eng = _make_engine(tmp_path)
    fact = pd.DataFrame(
        [
            {
                "customer_id": 1,
                "order_date": "2024-03-01",
                "product_division": "SOLIDWORKS",
                "product_sku": "SWX_Core",
                "gross_profit": 10,
            },
            {
                "customer_id": 1,
                "order_date": "2024-05-15",
                "product_division": "solidworks",
                "product_sku": "SWX_Core",
                "gross_profit": 15,
            },
            {
                "customer_id": 2,
                "order_date": "2024-05-20",
                "product_division": "Services",
                "product_sku": "Training",
                "gross_profit": 5,
            },
        ]
    )
    fact.to_sql("fact_transactions", eng, if_exists="replace", index=False)
    pd.DataFrame({"customer_id": [1, 2]}).to_sql("dim_customer", eng, if_exists="replace", index=False)

    params = LabelParams(
        division="SOLIDWORKS",
        cutoff="2024-04-30",
        window_months=2,
        mode="all",
        gp_min_threshold=0.0,
    )
    df = build_labels_for_division(eng, params)
    pdf = df.to_pandas()

    assert int(pdf.loc[pdf["customer_id"] == 1, "label"].iloc[0]) == 1


def test_widening_respects_sku_targets(tmp_path):
    """Ensure widening only counts intended SKUs when sku_targets exist.

    Use the logical model 'Printers' (SKU-targeted) and include only a
    non-target Hardware SKU ('Consumables') in the window and widened range.
    Labels should remain 0 even after widening to max months.
    """
    eng = _make_engine(tmp_path)
    # Only non-target SKU for Printers present after cutoff
    fact = pd.DataFrame(
        [
            {"customer_id": 1, "order_date": "2024-05-01", "product_division": "Hardware", "product_sku": "Consumables", "gross_profit": 5},
            {"customer_id": 1, "order_date": "2024-07-05", "product_division": "Hardware", "product_sku": "Consumables", "gross_profit": 50},
        ]
    )
    fact.to_sql("fact_transactions", eng, if_exists="replace", index=False)
    pd.DataFrame({"customer_id": [1]}).to_sql("dim_customer", eng, if_exists="replace", index=False)

    params = LabelParams(
        division="Printers",
        cutoff="2024-06-30",
        window_months=1,
        mode="all",
        gp_min_threshold=0.0,
        min_positive_target=1,
        max_window_months=6,
    )
    df = build_labels_for_division(eng, params)
    pdf = df.to_pandas()
    # No printer SKUs in any widened window -> remain negative
    assert int(pdf.loc[pdf["customer_id"] == 1, "label"].iloc[0]) == 0
