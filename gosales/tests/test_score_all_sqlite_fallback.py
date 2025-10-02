import copy
import sqlite3
from pathlib import Path

import pytest

from gosales.pipeline import score_all
from gosales.utils import config as config_mod
from gosales.utils import db as db_mod
from gosales.utils import paths as paths_mod
from gosales.etl import build_star as build_star_mod
from gosales.etl import load_csv as load_csv_mod


def test_score_all_etl_uses_sample_csvs_when_sqlite(monkeypatch, tmp_path):
    for var in ["AZSQL_SERVER", "AZSQL_DB", "AZSQL_USER", "AZSQL_PWD"]:
        monkeypatch.delenv(var, raising=False)

    cfg = copy.deepcopy(config_mod.load_config())
    cfg.database.sqlite_path = tmp_path / "gosales.db"
    cfg.database.curated_sqlite_path = tmp_path / "gosales_curated.db"
    cfg.paths.raw = (tmp_path / "raw").resolve()
    cfg.paths.staging = (tmp_path / "staging").resolve()
    cfg.paths.curated = (tmp_path / "curated").resolve()
    cfg.paths.outputs = (tmp_path / "outputs").resolve()

    data_dir = tmp_path / "data"
    samples_dir = data_dir / "database_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    sales_log_csv = samples_dir / "Sales_Log.csv"
    sales_log_csv.write_text(
        """CustomerId,Rec Date,Division,Customer,InvoiceId,branch,rep,SWX_Core,SWX_Core_Qty,Services,Services_Qty
ACME-001,2024-01-15,Solidworks,Acme Corp,INV-001,North,Jane Doe,1000,2,0,0
BETA-002,2024-02-20,Services,Beta LLC,INV-002,East,John Roe,0,0,500,5
""",
        encoding="utf-8",
    )

    industry_csv = samples_dir / "TR - Industry Enrichment.csv"
    industry_csv.write_text(
        """Customer,Cleaned Customer Name,Web Address,Industry,Industry Sub List,Reasoning,ID
Acme Corp,Acme Corporation,acme.com,Manufacturing,Manufacturing-General,Sample,1
Beta LLC,Beta Limited,beta.com,Technology,Technology-General,Sample,2
""",
        encoding="utf-8",
    )

    cfg.etl.industry_enrichment_csv = industry_csv

    monkeypatch.setattr(config_mod, "load_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(db_mod, "load_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(score_all, "load_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(build_star_mod, "load_config", lambda *args, **kwargs: cfg)

    outputs_dir = cfg.paths.outputs
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(paths_mod, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(paths_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(score_all, "DATA_DIR", data_dir)
    monkeypatch.setattr(load_csv_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(build_star_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(score_all, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(score_all, "MODELS_DIR", models_dir)
    monkeypatch.setattr(build_star_mod, "OUTPUTS_DIR", outputs_dir)

    monkeypatch.setattr(score_all, "compute_label_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_all, "validate_holdout", lambda *args, **kwargs: None)

    from gosales.features import engine as features_engine
    monkeypatch.setattr(features_engine, "create_feature_matrix", lambda *args, **kwargs: None)

    from gosales.pipeline import score_customers as score_customers_mod

    monkeypatch.setattr(score_customers_mod, "generate_scoring_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_all, "generate_scoring_outputs", lambda *args, **kwargs: None)

    monkeypatch.setattr(score_all, "_derive_targets", lambda: ["Solidworks"])

    def _fake_run(*args, **kwargs):
        return None

    monkeypatch.setattr(score_all.subprocess, "run", _fake_run)

    score_all.score_all()

    curated_path = Path(cfg.database.curated_sqlite_path)
    assert curated_path.exists()

    with sqlite3.connect(curated_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM fact_transactions")
        assert cur.fetchone()[0] > 0
        cur = conn.execute("SELECT COUNT(*) FROM dim_customer")
        assert cur.fetchone()[0] >= 2
