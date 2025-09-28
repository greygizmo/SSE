from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import yaml
from click.testing import CliRunner
from sqlalchemy import create_engine

from gosales.features.build import main as build_cli
from gosales.utils import config as cfgmod


def _seed_sparse_forced_windows(engine) -> None:
    fact = pd.DataFrame(
        [
            {
                "customer_id": 1,
                "order_date": "2024-02-15",
                "product_division": "Solidworks",
                "product_sku": "SWX_Core",
                "gross_profit": 100.0,
                "quantity": 1,
            },
            {
                "customer_id": 2,
                "order_date": "2024-02-20",
                "product_division": "Services",
                "product_sku": "Training",
                "gross_profit": 50.0,
                "quantity": 1,
            },
        ]
    )
    fact.to_sql("fact_transactions", engine, if_exists="replace", index=False)
    pd.DataFrame({"customer_id": [1, 2]}).to_sql(
        "dim_customer", engine, if_exists="replace", index=False
    )


def test_phase2_outputs_include_sparse_als_and_market_basket(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{tmp_path}/phase2_sparse.db")
    _seed_sparse_forced_windows(eng)

    out_dir = tmp_path / "outputs"
    monkeypatch.setattr("gosales.utils.paths.OUTPUTS_DIR", out_dir)
    monkeypatch.setattr("gosales.features.build.OUTPUTS_DIR", out_dir)
    monkeypatch.setattr("gosales.features.engine.OUTPUTS_DIR", out_dir)
    monkeypatch.setattr("gosales.ops.run.OUTPUTS_DIR", out_dir)

    monkeypatch.setattr("gosales.features.build.get_curated_connection", lambda: eng)
    monkeypatch.setattr("gosales.features.build.get_db_connection", lambda: eng)
    monkeypatch.setattr("gosales.features.build.validate_connection", lambda _: True)

    orig_load_config = cfgmod.load_config

    cfg = orig_load_config()
    cfg.features.use_als_embeddings = True
    cfg.features.als_lookback_months = 1
    cfg.features.use_market_basket = True
    cfg.features.affinity_lag_days = 60

    config_path = tmp_path / "phase2_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f)

    def _fake_load_config(config_path_override: str | Path | None = None, cli_overrides=None):
        path = Path(config_path_override) if config_path_override else config_path
        return orig_load_config(path, cli_overrides=cli_overrides)

    monkeypatch.setattr("gosales.utils.config.load_config", _fake_load_config)
    monkeypatch.setattr("gosales.features.build.load_config", _fake_load_config)
    monkeypatch.setattr("gosales.features.engine.cfg.load_config", _fake_load_config)
    monkeypatch.setattr("gosales.ops.run.load_config", _fake_load_config)

    runner = CliRunner()
    result = runner.invoke(
        build_cli,
        [
            "--division",
            "Solidworks",
            "--cutoff",
            "2024-03-31",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output

    feat_path = out_dir / "features_solidworks_2024-03-31.parquet"
    catalog_path = out_dir / "feature_catalog_solidworks_2024-03-31.csv"
    assert feat_path.exists()
    assert catalog_path.exists()

    df = pl.read_parquet(feat_path)
    expected_als_cols = [f"als_f{i}" for i in range(16)]
    expected_mb_cols = [
        "mb_lift_max_lag60d",
        "mb_lift_mean_lag60d",
        "affinity__div__lift_topk__12m_lag60d",
    ]

    for col in expected_als_cols + expected_mb_cols:
        assert col in df.columns
        assert df.select(pl.col(col)).to_series().is_null().sum() == 0

    catalog = pd.read_csv(catalog_path)
    names = set(catalog["name"].tolist())
    for col in expected_als_cols + expected_mb_cols:
        assert col in names

    stats_path = out_dir / "feature_stats_solidworks_2024-03-31.json"
    assert stats_path.exists()
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    for col in expected_als_cols + expected_mb_cols:
        assert col in stats["columns"]
