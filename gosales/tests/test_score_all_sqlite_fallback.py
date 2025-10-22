import copy
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine

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

    # Minimal line-item sample to back the Phase 0 line-item ETL path
    sales_detail_csv = samples_dir / "table_saleslog_detail.csv"
    sales_detail_csv.write_text(
        """Rec_Date,Sales_Order,Item_internalid,Revenue,Amount2,GP,Term_GP,Division,CompanyId
2024-01-15,INV-001,SKU-1,1000,200,800,0,Solidworks,ACME-001
2024-02-20,INV-002,SKU-2,500,100,400,0,Services,BETA-002
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

    # Configure sales_detail to use the locally loaded CSV-backed table on SQLite
    # and relax the allow-list to permit the test table identifier.
    cfg.database.source_tables["sales_detail"] = "sales_detail"
    # Override line-item source to point at our local table; skip other DB-backed sources
    cfg.etl.line_items.sources.sales_detail = "sales_detail"
    cfg.etl.line_items.sources.product_info = "csv"
    cfg.etl.line_items.sources.items_category_limited = "csv"
    cfg.etl.line_items.sources.product_tags = "csv"
    cfg.database.allowed_identifiers = []
    # Disable line-type exclusions to keep tiny sample rows
    cfg.etl.line_items.behavior.exclude_line_types = []

    # Load the sample line-item CSV into the primary (SQLite) engine as 'sales_detail'
    engine = db_mod.get_db_connection()
    load_csv_mod.load_csv_to_db(sales_detail_csv, "sales_detail", engine)

    # Stub legacy CSV ingest used by score_all for local runs; Phase 0 ignores Sales_Log
    monkeypatch.setattr(load_csv_mod, "load_csv_to_db", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_all, "load_csv_to_db", lambda *args, **kwargs: None)

    score_all.score_all()

    curated_path = Path(cfg.database.curated_sqlite_path)
    assert curated_path.exists()

    with sqlite3.connect(curated_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM fact_transactions")
        assert cur.fetchone()[0] > 0
        cur = conn.execute("SELECT COUNT(*) FROM dim_customer")
        assert cur.fetchone()[0] >= 2


def test_score_all_holdout_failure_propagates(monkeypatch, tmp_path):
    for var in ["AZSQL_SERVER", "AZSQL_DB", "AZSQL_USER", "AZSQL_PWD"]:
        monkeypatch.delenv(var, raising=False)

    cfg = copy.deepcopy(config_mod.load_config())
    cfg.database.sqlite_path = tmp_path / "primary.db"
    cfg.database.curated_sqlite_path = tmp_path / "curated.db"
    cfg.paths.raw = (tmp_path / "raw").resolve()
    cfg.paths.staging = (tmp_path / "staging").resolve()
    cfg.paths.curated = (tmp_path / "curated").resolve()
    cfg.paths.outputs = (tmp_path / "outputs").resolve()

    cfg.paths.raw.mkdir(parents=True, exist_ok=True)
    cfg.paths.staging.mkdir(parents=True, exist_ok=True)
    cfg.paths.curated.mkdir(parents=True, exist_ok=True)
    cfg.paths.outputs.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config_mod, "load_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(db_mod, "load_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(score_all, "load_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(build_star_mod, "load_config", lambda *args, **kwargs: cfg)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = cfg.paths.outputs
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(paths_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(paths_mod, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(paths_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(score_all, "DATA_DIR", data_dir)
    monkeypatch.setattr(score_all, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(score_all, "MODELS_DIR", models_dir)

    primary_engine = create_engine(f"sqlite:///{cfg.database.sqlite_path}")
    curated_engine = create_engine(f"sqlite:///{cfg.database.curated_sqlite_path}")
    monkeypatch.setattr(score_all, "get_db_connection", lambda: primary_engine)
    monkeypatch.setattr(db_mod, "get_curated_connection", lambda: curated_engine)
    monkeypatch.setattr(score_all, "validate_connection", lambda engine: True)

    monkeypatch.setattr(load_csv_mod, "load_csv_to_db", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_all, "load_csv_to_db", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_all, "build_star_schema", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(score_all, "compute_label_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_all, "_prune_legacy_model_dirs", lambda *args, **kwargs: None)

    from gosales.features import engine as features_engine

    monkeypatch.setattr(features_engine, "create_feature_matrix", lambda *args, **kwargs: None)

    def fake_generate_scoring_outputs(*args, run_manifest=None, **kwargs):
        path = outputs_dir / "icp_scores.csv"
        path.write_text(
            "division_name,icp_score,bought_in_division\nSolidworks,0.8,0\n",
            encoding="utf-8",
        )
        if isinstance(run_manifest, dict):
            run_manifest["icp_scores"] = str(path)
        return path

    monkeypatch.setattr(score_all, "generate_scoring_outputs", fake_generate_scoring_outputs)
    monkeypatch.setattr(score_all, "_derive_targets", lambda: ["Solidworks"])
    monkeypatch.setattr(score_all.subprocess, "run", lambda *args, **kwargs: None)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated holdout failure")

    monkeypatch.setattr(score_all, "validate_holdout", boom)

    with pytest.raises(RuntimeError, match="Hold-out validation step failed"):
        score_all.score_all()
