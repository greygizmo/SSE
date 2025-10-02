import json

import pytest

from gosales.pipeline import score_all
from gosales.ops import run as run_ops
from gosales.utils import paths as path_utils


class _DummyConfig:
    def __init__(self):
        self.database = type("DB", (), {"strict_db": False, "source_tables": {}})()
        self.run = type("Run", (), {"cutoff_date": None})()

    def to_dict(self):
        return {}


class _FailingInspector:
    def __init__(self, tables=None):
        self._tables = tables or []

    def get_table_names(self):
        return list(self._tables)


def test_pipeline_stops_and_records_failure_on_star_error(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    monkeypatch.setattr(score_all, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(path_utils, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(run_ops, "OUTPUTS_DIR", outputs_dir)

    dummy_engine = object()
    monkeypatch.setattr(score_all, "get_db_connection", lambda: dummy_engine)
    monkeypatch.setattr(score_all, "validate_connection", lambda engine: True)
    monkeypatch.setattr(score_all, "load_csv_to_db", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_all, "division_set", lambda: {"Solidworks"})
    monkeypatch.setattr(score_all, "get_supported_models", lambda: {"Solidworks"})

    import gosales.utils.db as db_utils

    monkeypatch.setattr(db_utils, "get_curated_connection", lambda: object())

    monkeypatch.setattr(score_all, "inspect", lambda engine: _FailingInspector([]))

    def _boom(*args, **kwargs):
        raise RuntimeError("star exploded")

    monkeypatch.setattr(score_all, "build_star_schema", _boom)

    dummy_cfg = _DummyConfig()
    monkeypatch.setattr(score_all, "load_config", lambda *args, **kwargs: dummy_cfg)
    monkeypatch.setattr(run_ops, "load_config", lambda *args, **kwargs: dummy_cfg)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("downstream stages should not execute when ETL fails")

    monkeypatch.setattr(score_all, "compute_label_audit", _should_not_run)
    monkeypatch.setattr(score_all, "generate_scoring_outputs", _should_not_run)
    monkeypatch.setattr(score_all.subprocess, "run", _should_not_run)

    with pytest.raises(RuntimeError, match="star exploded"):
        score_all.score_all()

    runs_dir = outputs_dir / "runs"
    manifest_paths = list(runs_dir.glob("*/manifest.json"))
    assert manifest_paths, "failure run should emit a manifest"
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert manifest["files"]["status"] == "error"
    assert "star schema build" in manifest["files"]["message"].lower()

    registry_path = runs_dir / "runs.jsonl"
    assert registry_path.exists()
    with registry_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    statuses = [rec.get("status") for rec in records if rec.get("phase") == "pipeline_score_all"]
    assert "error" in statuses
