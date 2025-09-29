"""Tests for the monitoring data collector type consistency score."""

from typing import Iterable, Sequence

import pytest

from gosales.monitoring.data_collector import MonitoringDataCollector


class _FakeResult:
    def __init__(self, rows: Sequence[Sequence[object]]):
        self._rows = list(rows)

    def fetchall(self):
        return list(self._rows)


class _FakeConnection:
    def __init__(self, rows: Sequence[Sequence[object]]):
        self._rows = list(rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *_args, **_kwargs):
        return _FakeResult(self._rows)


class _FakeEngine:
    def __init__(self, rows: Sequence[Sequence[object]]):
        self._rows = list(rows)

    def connect(self):
        return _FakeConnection(self._rows)


def _patch_engine(monkeypatch: pytest.MonkeyPatch, rows: Iterable[Sequence[object]]) -> None:
    engine = _FakeEngine(list(rows))
    monkeypatch.setattr(
        "gosales.monitoring.data_collector.get_db_connection",
        lambda: engine,
    )


def test_type_consistency_scores_high_for_uniform_types(monkeypatch: pytest.MonkeyPatch):
    """Uniform sample types should yield a high score."""

    _patch_engine(monkeypatch, [(1,), (2,), (3,), (4,)])
    collector = MonitoringDataCollector()

    score = collector._calculate_type_consistency_score()

    assert score >= 99.0


def test_type_consistency_penalizes_mixed_types(monkeypatch: pytest.MonkeyPatch):
    """Mixed-type samples should be penalized relative to uniform samples."""

    _patch_engine(monkeypatch, [(1,), ("2",), (3,), (None,)])
    collector = MonitoringDataCollector()

    score = collector._calculate_type_consistency_score()

    assert score < 95.0
