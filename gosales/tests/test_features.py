from __future__ import annotations

import json
import pandas as pd
import polars as pl
import yaml
from click.testing import CliRunner
from sqlalchemy import create_engine

from gosales.features.build import main as build_cli
from gosales.features.engine import create_feature_matrix
from gosales.pipeline.rank_whitespace import _compute_assets_als_norm
from gosales.utils.config import load_config


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



def test_feature_matrix_handles_mixed_case_division(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/test_features_case.db")
    fact = pd.DataFrame(
        [
            {
                "customer_id": 1,
                "order_date": "2024-01-10",
                "product_division": "SOLIDWORKS",
                "product_sku": "SWX_Core",
                "gross_profit": 100,
                "quantity": 1,
            },
            {
                "customer_id": 1,
                "order_date": "2024-02-05",
                "product_division": "solidworks",
                "product_sku": "SWX_Core",
                "gross_profit": 50,
                "quantity": 1,
            },
            {
                "customer_id": 2,
                "order_date": "2024-01-15",
                "product_division": "services",
                "product_sku": "Training",
                "gross_profit": 20,
                "quantity": 1,
            },
        ]
    )
    fact.to_sql("fact_transactions", eng, if_exists="replace", index=False)
    pd.DataFrame({"customer_id": [1, 2]}).to_sql("dim_customer", eng, if_exists="replace", index=False)

    fm = create_feature_matrix(
        eng,
        "sOlIdWoRkS",
        cutoff_date="2024-01-31",
        prediction_window_months=1,
    )

    pdf = fm.to_pandas()
    assert int(pdf.loc[pdf["customer_id"] == 1, "bought_in_division"].iloc[0]) == 1


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


def test_missingness_flags_capture_original_nulls(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/test_missingness.db")
    _seed(eng)
    # Add a customer with no transactions so engineered columns are NaN before fillna
    pd.DataFrame({"customer_id": [3]}).to_sql("dim_customer", eng, if_exists="append", index=False)

    fm = create_feature_matrix(eng, "Solidworks", cutoff_date="2024-01-31", prediction_window_months=1)
    pdf = fm.to_pandas()
    # Normalize type for safer lookups
    pdf["customer_id"] = pdf["customer_id"].astype(str)

    # Ensure a canonical feature gets a corresponding _missing flag
    missing_cols = [c for c in pdf.columns if c.endswith("_missing")]
    assert "total_transactions_all_time_missing" in missing_cols

    cust3_flag = int(pdf.loc[pdf["customer_id"] == "3", "total_transactions_all_time_missing"].iloc[0])
    cust1_flag = int(pdf.loc[pdf["customer_id"] == "1", "total_transactions_all_time_missing"].iloc[0])

    # Customer without transactions should be flagged as missing prior to fillna; customer with history should not
    assert cust3_flag == 1
    assert cust1_flag == 0


def test_load_config_parses_assets_als(tmp_path):
    cfg_path = tmp_path / "custom_config.yaml"
    paths = {
        "raw": str(tmp_path / "raw"),
        "staging": str(tmp_path / "staging"),
        "curated": str(tmp_path / "curated"),
        "outputs": str(tmp_path / "outputs"),
    }
    cfg_payload = {
        "paths": paths,
        "features": {
            "use_assets_als": True,
            "assets_als": {
                "max_rows": 123,
                "max_cols": 7,
                "factors": 8,
                "iters": 3,
                "reg": 0.25,
            },
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_payload, f)

    cfg = load_config(config_path=cfg_path)

    assert cfg.features.use_assets_als is True
    assert cfg.features.assets_als.max_rows == 123
    assert cfg.features.assets_als.max_cols == 7
    assert cfg.features.assets_als.factors == 8
    assert cfg.features.assets_als.iters == 3
    assert cfg.features.assets_als.reg == 0.25


def test_assets_als_norm_respects_limits(monkeypatch):
    class _AssetsCfg:
        max_rows = 2
        max_cols = 2

    class _FeaturesCfg:
        assets_als = _AssetsCfg()

    class _Cfg:
        features = _FeaturesCfg()

    monkeypatch.setattr("gosales.utils.config.load_config", lambda *args, **kwargs: _Cfg())

    df = pd.DataFrame(
        {
            "division_name": ["A"] * 3,
            "als_assets_f0": [0.1, 0.2, 0.3],
            "als_assets_f1": [0.4, 0.5, 0.6],
            "als_assets_f2": [0.7, 0.8, 0.9],
        }
    )

    result = _compute_assets_als_norm(df)

    assert (result == 0).all()
