from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from gosales.features.engine import create_feature_matrix
from gosales.features.build import main as build_cli
from click.testing import CliRunner
from gosales.utils.paths import OUTPUTS_DIR


def _seed(engine):
    fact = pd.DataFrame([
        {"customer_id": 1, "order_date": "2024-01-01", "product_division": "Solidworks", "product_sku": "SWX_Core", "gross_profit": 100},
        {"customer_id": 1, "order_date": "2024-02-01", "product_division": "Services", "product_sku": "Training", "gross_profit": 5},
        {"customer_id": 2, "order_date": "2023-12-15", "product_division": "Simulation", "product_sku": "Simulation", "gross_profit": 50},
    ])
    fact.to_sql("fact_transactions", engine, if_exists="replace", index=False)
    pd.DataFrame({"customer_id": [1, 2]}).to_sql("dim_customer", engine, if_exists="replace", index=False)


def test_feature_window_and_target(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/test_features.db")
    _seed(eng)
    fm = create_feature_matrix(eng, "Solidworks", cutoff_date="2024-01-31", prediction_window_months=1)
    pdf = fm.to_pandas()
    # Customer 1 has Solidworks pre-cutoff and Services pre-cutoff; in Feb window, no Solidworks, so target=0
    assert int(pdf.loc[pdf["customer_id"] == 1, "bought_in_division"].iloc[0]) == 0
    # Customer 2 had only Simulation pre-cutoff; also 0
    assert int(pdf.loc[pdf["customer_id"] == 2, "bought_in_division"].iloc[0]) == 0


def test_feature_cli_checksum(tmp_path, monkeypatch):
    # Use temp output dir by monkeypatching OUTPUTS_DIR if needed
    eng = create_engine(f"sqlite:///{tmp_path}/test_features_cli.db")
    _seed(eng)
    # Build via engine first
    fm = create_feature_matrix(eng, "Solidworks", cutoff_date="2024-01-31", prediction_window_months=1)
    assert not fm.is_empty()
    # Run CLI
    runner = CliRunner()
    result = runner.invoke(build_cli, ["--division","Solidworks","--cutoff","2024-01-31"]) 
    assert result.exit_code == 0

