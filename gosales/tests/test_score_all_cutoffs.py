import logging
import types

from gosales.pipeline.score_all import (
    _resolve_training_cutoffs,
    _DEFAULT_TRAINING_CUTOFFS,
)


def _run_cfg(**kwargs):
    defaults = {
        "training_cutoffs": [],
        "training_frequency_months": 0,
        "training_cutoff_count": 0,
        "cutoff_date": None,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def test_resolve_training_cutoffs_prefers_override():
    cfg = _run_cfg(cutoff_date="2024-06-30")
    log = logging.getLogger("score_all_override")
    resolved = _resolve_training_cutoffs(
        cfg,
        cutoff_date="2025-03-31",
        override_cutoffs=["2025-03-31", "invalid-date"],
        log=log,
    )
    assert resolved == ["2025-03-31"]


def test_resolve_training_cutoffs_uses_config_list():
    cfg = _run_cfg(training_cutoffs=["2023-12-31", "2024-06-30"])
    log = logging.getLogger("score_all_config")
    resolved = _resolve_training_cutoffs(cfg, cutoff_date=None, override_cutoffs=None, log=log)
    assert resolved == ["2023-12-31", "2024-06-30"]


def test_resolve_training_cutoffs_generates_from_frequency():
    cfg = _run_cfg(cutoff_date="2024-12-31", training_frequency_months=6, training_cutoff_count=3)
    log = logging.getLogger("score_all_generate")
    resolved = _resolve_training_cutoffs(cfg, cutoff_date=None, override_cutoffs=None, log=log)
    assert resolved == ["2023-12-31", "2024-06-30", "2024-12-31"]


def test_resolve_training_cutoffs_falls_back_to_primary_when_generation_invalid():
    cfg = _run_cfg(cutoff_date="2024-12-31", training_frequency_months=0, training_cutoff_count=0)
    log = logging.getLogger("score_all_fallback_primary")
    resolved = _resolve_training_cutoffs(cfg, cutoff_date=None, override_cutoffs=None, log=log)
    assert resolved == ["2024-12-31"]


def test_resolve_training_cutoffs_uses_legacy_defaults_when_no_cutoff_available():
    cfg = _run_cfg()
    log = logging.getLogger("score_all_legacy")
    resolved = _resolve_training_cutoffs(cfg, cutoff_date=None, override_cutoffs=None, log=log)
    assert resolved == list(_DEFAULT_TRAINING_CUTOFFS)
