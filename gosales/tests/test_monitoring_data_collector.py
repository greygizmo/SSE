import json
import pytest

from gosales.monitoring import data_collector as dc_module
from gosales.monitoring.data_collector import MonitoringDataCollector


class _DummyResult:
    def __init__(self, value: int):
        self._value = value

    def scalar_one(self):
        return self._value

    def scalar(self):
        return self._value

    def first(self):
        return (self._value,)

    def fetchone(self):
        return (self._value,)


class _DummyConnection:
    def __init__(self, value: int):
        self._value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, *args, **kwargs):
        return _DummyResult(self._value)


class _DummyEngine:
    def __init__(self, value: int):
        self._value = value

    def connect(self):
        return _DummyConnection(self._value)


def test_collect_recent_alerts_reads_base_validation_file(tmp_path, monkeypatch):
    validation_data = {
        "alerts": [
            {
                "severity": "WARN",
                "message": "Base validation alert",
                "component": "Validation",
            }
        ],
        "timestamp": "2024-01-01T00:00:00Z",
    }

    (tmp_path / "validation_metrics.json").write_text(
        json.dumps(validation_data), encoding="utf-8"
    )

    monkeypatch.setattr(dc_module, "OUTPUTS_DIR", tmp_path)

    collector = MonitoringDataCollector()

    alerts = collector._collect_recent_alerts()
    assert any(alert["message"] == "Base validation alert" for alert in alerts)

    score = collector._calculate_data_quality_score()
    assert 30.0 <= score <= 60.0


def test_collect_recent_alerts_success(tmp_path, monkeypatch):
    payload = {
        "status": "ok",
        "gates": {"auc": 0.70, "lift_at_10": 2.0, "cal_mae": 0.10},
        "divisions": [
            {
                "division_name": "Widgets",
                "auc": 0.82,
                "lift_at_10": 2.4,
                "cal_mae": 0.06,
            }
        ],
    }
    metrics_path = tmp_path / "validation_metrics_2024.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(dc_module, "OUTPUTS_DIR", tmp_path)

    collector = MonitoringDataCollector()
    alerts = collector._collect_recent_alerts()

    assert alerts == [
        {
            "level": "INFO",
            "message": "Pipeline completed successfully",
            "timestamp": alerts[0]["timestamp"],
            "component": "Pipeline",
        }
    ]
    score = collector._calculate_data_quality_score()
    assert score == pytest.approx(100.0)


def test_collect_recent_alerts_validation_failure(tmp_path, monkeypatch):
    payload = {
        "status": "fail",
        "gates": {"auc": 0.70, "lift_at_10": 2.0, "cal_mae": 0.10},
        "divisions": [
            {
                "division_name": "Widgets",
                "auc": 0.55,
                "lift_at_10": 1.8,
                "cal_mae": 0.22,
            }
        ],
    }
    metrics_path = tmp_path / "validation_metrics_2024.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(dc_module, "OUTPUTS_DIR", tmp_path)

    collector = MonitoringDataCollector()
    alerts = collector._collect_recent_alerts()

    error_messages = [a["message"] for a in alerts if a["level"] == "ERROR"]

    assert any("status=fail" in msg for msg in error_messages)
    assert any("auc" in msg for msg in error_messages)
    assert not any("successfully" in a["message"] for a in alerts)

    score = collector._calculate_data_quality_score()
    assert score == 0.0


def test_collect_performance_metrics_from_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(dc_module, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(dc_module, "sql_text", None)
    monkeypatch.setattr(dc_module, "get_db_connection", lambda: _DummyEngine(1234))
    monkeypatch.setattr(MonitoringDataCollector, "_get_memory_usage", lambda self: 0.5)

    run_manifest = {
        "divisions_scored": ["NA", "EU"],
        "steps": [
            {"name": "features", "records": 1000, "duration_seconds": 20.0},
            {"name": "scoring", "records_processed": 500, "duration_seconds": 10.0},
        ],
    }
    (tmp_path / "run_context_20240101.json").write_text(json.dumps(run_manifest), encoding="utf-8")

    metrics = MonitoringDataCollector()._collect_performance_metrics()

    assert metrics["processing_rate"] == pytest.approx(50.0)
    assert metrics["memory_usage"] == 0.5
    assert metrics["active_divisions"] == 2
    assert metrics["division_labels"] == ["NA", "EU"]
    assert metrics["total_customers"] == 1234
    assert "fallbacks" not in metrics


def test_collect_performance_metrics_with_fallbacks(monkeypatch, tmp_path):
    monkeypatch.setattr(dc_module, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(dc_module, "sql_text", None)
    monkeypatch.setattr(dc_module, "get_db_connection", lambda: None)
    monkeypatch.setattr(MonitoringDataCollector, "_get_memory_usage", lambda self: 0.25)

    metrics = MonitoringDataCollector()._collect_performance_metrics()

    assert metrics["processing_rate"] == 0.0
    assert metrics["active_divisions"] == 0
    assert metrics["total_customers"] == 0
    assert metrics["fallbacks"]["processing_rate"]
    assert metrics["fallbacks"]["active_divisions"]
    assert metrics["fallbacks"]["total_customers"]


def test_nested_validation_metrics_are_discovered(tmp_path, monkeypatch):
    monkeypatch.setattr(dc_module, "OUTPUTS_DIR", tmp_path)
    collector = MonitoringDataCollector()

    nested_metrics = tmp_path / "validation" / "widgets" / "2024-06-30" / "metrics.json"
    nested_metrics.parent.mkdir(parents=True, exist_ok=True)
    nested_metrics.write_text(json.dumps({"timestamp": "2024-06-30"}), encoding="utf-8")

    files = collector._find_validation_metric_files()
    assert nested_metrics in files

    score = collector._calculate_data_quality_score()
    assert score == pytest.approx(55.0)


def test_data_quality_score_no_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(dc_module, "OUTPUTS_DIR", tmp_path)
    collector = MonitoringDataCollector()
    score = collector._calculate_data_quality_score()
    assert score == pytest.approx(40.0)


def test_data_lineage_from_run_context(tmp_path, monkeypatch):
    run_manifest = {
        "steps": [
            {
                "name": "features",
                "success": True,
                "records": 1000,
                "duration_seconds": 20.1234,
                "source": "feature_builder",
            },
            {
                "name": "scoring",
                "success": False,
                "records_processed": 500,
                "duration": "00:05",
                "data_source": "scoring_engine",
            },
        ]
    }
    (tmp_path / "run_context_20240101.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    monkeypatch.setattr(dc_module, "OUTPUTS_DIR", tmp_path)

    lineage = MonitoringDataCollector()._collect_data_lineage()

    assert lineage == [
        {
            "step": "features",
            "status": "success",
            "records_processed": 1000,
            "execution_time": 20.123,
            "data_source": "feature_builder",
        },
        {
            "step": "scoring",
            "status": "failed",
            "records_processed": 500,
            "execution_time": "00:05",
            "data_source": "scoring_engine",
        },
    ]


def test_data_lineage_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dc_module, "OUTPUTS_DIR", tmp_path)
    assert MonitoringDataCollector()._collect_data_lineage() == []
