from contextlib import contextmanager
import types

import pytest

from gosales.pipeline import score_all
from gosales.utils.normalize import normalize_model_key


class _StubLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def warning(self, msg):
        self.messages.append(("warning", msg))


def test_pruning_keeps_solidworks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        score_all,
        "division_set",
        lambda: {"Solidworks", "Services", "Success Plan"},
    )
    monkeypatch.setattr(
        score_all,
        "get_supported_models",
        lambda: {"Printers", "Success_Plan"},
    )

    targets = score_all._derive_targets()

    # Ensure Solidworks is part of the computed targets
    assert "Solidworks" in targets

    success_plan_targets = [t for t in targets if normalize_model_key(t) == "success plan"]
    assert len(success_plan_targets) == 1, "Success Plan variants should collapse"

    solidworks_dir = tmp_path / "solidworks_model"
    solidworks_cold_dir = tmp_path / "solidworks_cold_model"
    legacy_dir = tmp_path / "legacy_model"
    solidworks_dir.mkdir()
    solidworks_cold_dir.mkdir()
    legacy_dir.mkdir()

    score_all._prune_legacy_model_dirs(targets, tmp_path, log=_StubLogger())

    assert solidworks_dir.exists(), "solidworks_model should be preserved during pruning"
    assert solidworks_cold_dir.exists(), "solidworks_cold_model should be preserved when warm target is active"
    assert not legacy_dir.exists(), "legacy_model should be removed as legacy"


def test_pruning_respects_alias_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(score_all, "division_set", lambda: {"SW Inspection"})
    monkeypatch.setattr(score_all, "get_supported_models", lambda: set())

    targets = score_all._derive_targets()
    assert {t for t in targets if t}  # sanity: have at least one target

    warm_alias_dir = tmp_path / "sw-inspection_model"
    cold_alias_dir = tmp_path / "sw-inspection_cold_model"
    other_dir = tmp_path / "unused_model"
    warm_alias_dir.mkdir()
    cold_alias_dir.mkdir()
    other_dir.mkdir()

    score_all._prune_legacy_model_dirs(targets, tmp_path, log=_StubLogger())

    assert warm_alias_dir.exists(), "Alias warm directory should be retained"
    assert cold_alias_dir.exists(), "Alias cold directory should be retained"
    assert not other_dir.exists(), "Unrelated model directory should be pruned"


def test_score_all_propagates_scoring_errors(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "outputs"
    models_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    outputs_dir.mkdir()
    models_dir.mkdir()
    data_dir.mkdir()

    monkeypatch.setattr(score_all, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(score_all, "MODELS_DIR", models_dir)
    monkeypatch.setattr(score_all, "DATA_DIR", data_dir)

    monkeypatch.setattr(score_all, "get_db_connection", lambda: object())
    monkeypatch.setattr("gosales.utils.db.get_curated_connection", lambda: object())
    monkeypatch.setattr(score_all, "validate_connection", lambda _conn: True)
    monkeypatch.setattr(score_all, "load_csv_to_db", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_all, "build_star_schema", lambda *args, **kwargs: None)
    monkeypatch.setattr("gosales.etl.events.build_fact_events", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_all, "compute_label_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr("gosales.features.engine.create_feature_matrix", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_all, "subprocess", types.SimpleNamespace(run=lambda *args, **kwargs: None))
    monkeypatch.setattr(score_all, "_prune_legacy_model_dirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(score_all, "_derive_targets", lambda: [])
    monkeypatch.setattr(score_all, "default_manifest", lambda **kwargs: {"run_id": "stub", "cutoff": None, "window_months": None})
    monkeypatch.setattr(score_all, "emit_manifest", lambda *args, **kwargs: None)

    cfg = types.SimpleNamespace(
        database=types.SimpleNamespace(strict_db=False, source_tables={}),
    )
    monkeypatch.setattr(score_all, "load_config", lambda: cfg)

    @contextmanager
    def fake_run_context(_phase):
        yield {"write_manifest": lambda *_args, **_kwargs: None, "append_registry": lambda *_args, **_kwargs: None}

    monkeypatch.setattr(score_all, "run_context", fake_run_context)

    def raise_scoring(*_args, **_kwargs):
        raise RuntimeError("No models were available for scoring.")

    monkeypatch.setattr(score_all, "generate_scoring_outputs", raise_scoring)

    with pytest.raises(RuntimeError, match="No models were available for scoring"):
        score_all.score_all()
