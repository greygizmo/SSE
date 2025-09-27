from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from gosales.features.engine import create_feature_matrix
from gosales.features.build import main as build_cli
from click.testing import CliRunner
from gosales.utils.config import load_config
import json
import yaml
import polars as pl


def _seed(engine):
    fact = pd.DataFrame([
        {"customer_id": 1, "order_date": "2024-01-01", "product_division": "Solidworks", "product_sku": "SWX_Core", "gross_profit": 100, "quantity": 1},
        {"customer_id": 1, "order_date": "2024-02-01", "product_division": "Services", "product_sku": "Training", "gross_profit": 5, "quantity": 1},
        {"customer_id": 2, "order_date": "2023-12-15", "product_division": "Simulation", "product_sku": "Simulation", "gross_profit": 50, "quantity": 1},
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


def test_feature_matrix_without_cutoff(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/test_features_no_cutoff.db")
    _seed(eng)

    feature_frame = create_feature_matrix(eng, "Solidworks", cutoff_date=None, prediction_window_months=1)

    assert not feature_frame.is_empty()
    assert feature_frame.height > 0


def test_feature_cli_checksum(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{tmp_path}/test_features_cli.db")
    _seed(eng)
    monkeypatch.setattr("gosales.features.build.get_db_connection", lambda: eng)
    out_dir = tmp_path / "out_cli"
    monkeypatch.setattr("gosales.features.build.OUTPUTS_DIR", out_dir)
    monkeypatch.setattr("gosales.ops.run.OUTPUTS_DIR", out_dir)
    fm = create_feature_matrix(eng, "Solidworks", cutoff_date="2024-01-31", prediction_window_months=1)
    assert not fm.is_empty()
    runner = CliRunner()
    cfg = load_config()
    cfg.features.use_als_embeddings = False
    cfg_path = tmp_path / "config_cli.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f)
    result = runner.invoke(
        build_cli,
        ["--division", "Solidworks", "--cutoff", "2024-01-31", "--config", str(cfg_path)],
    )
    assert result.exit_code == 0


def test_cli_config_override_persist(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    monkeypatch.setattr("gosales.features.build.OUTPUTS_DIR", out_dir)
    monkeypatch.setattr("gosales.ops.run.OUTPUTS_DIR", out_dir)
    monkeypatch.setattr("gosales.features.build.get_db_connection", lambda: None)

    def fake_create_feature_matrix(engine, division, cut, pred_win):
        return pl.DataFrame(
            {
                "customer_id": [1, 2],
                "rfm__div__gp_sum__3m": [100.0, 50.0],
                "rfm__div__gp_sum__6m": [100.0, 50.0],
                "rfm__div__gp_sum__12m": [100.0, 50.0],
                "rfm__div__gp_sum__24m": [100.0, 50.0],
            }
        )

    monkeypatch.setattr("gosales.features.build.create_feature_matrix", fake_create_feature_matrix)

    cfg = load_config()
    cfg.features.gp_winsor_p = 0.5
    cfg.features.use_als_embeddings = False
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f)

    runner = CliRunner()
    result = runner.invoke(
        build_cli,
        ["--division", "Solidworks", "--cutoff", "2024-01-31", "--config", str(cfg_path)],
    )
    assert result.exit_code == 0

    stats_path = out_dir / "feature_stats_solidworks_2024-01-31.json"
    assert stats_path.exists()
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    assert stats["winsor_caps"]["rfm__div__gp_sum__3m"]["upper"] == 75.0

