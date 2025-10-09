import json
from pathlib import Path
import os
from types import SimpleNamespace

import pandas as pd

from gosales.pipeline import validate_holdout
from gosales.validation.holdout_data import HoldoutData


def test_validate_against_holdout_skips_without_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_holdout, "OUTPUTS_DIR", tmp_path)

    result = validate_holdout.validate_against_holdout()

    assert result["status"] == "skipped"
    assert "icp_scores" in result["message"]


def test_validate_against_holdout_uses_primary_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_holdout, "OUTPUTS_DIR", tmp_path)
    scores_path = tmp_path / "icp_scores.csv"
    scores_path.write_text("division_name,icp_score\nA,0.5\n", encoding="utf-8")

    captured = {}

    def _fake_validate_holdout(icp_scores_csv: str, **kwargs):
        captured["path"] = icp_scores_csv
        out = tmp_path / "validation_metrics.json"
        out.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        return out

    monkeypatch.setattr(validate_holdout, "validate_holdout", _fake_validate_holdout)

    result = validate_holdout.validate_against_holdout(strict=False)

    assert result["status"] == "ok"
    assert Path(captured["path"]) == scores_path
    assert Path(result["metrics_path"]).name == "validation_metrics.json"


def test_validate_against_holdout_falls_back_to_timestamped_file(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_holdout, "OUTPUTS_DIR", tmp_path)
    fallback_old = tmp_path / "icp_scores_20230101.csv"
    fallback_new = tmp_path / "icp_scores_20240201.csv"
    fallback_old.write_text("division_name,icp_score\nA,0.1\n", encoding="utf-8")
    fallback_new.write_text("division_name,icp_score\nB,0.9\n", encoding="utf-8")
    os.utime(
        fallback_old, (fallback_old.stat().st_atime, fallback_old.stat().st_mtime - 100)
    )
    os.utime(fallback_new, None)

    captured = {}

    def _fake_validate_holdout(icp_scores_csv: str, **kwargs):
        captured["path"] = icp_scores_csv
        out = tmp_path / "validation_metrics.json"
        out.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        return out

    monkeypatch.setattr(validate_holdout, "validate_holdout", _fake_validate_holdout)

    result = validate_holdout.validate_against_holdout()

    assert result["status"] == "ok"
    assert Path(captured["path"]) == fallback_new
    assert result["scores_path"] == str(fallback_new)


def test_validate_holdout_writes_metrics(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "division_name": ["Solidworks", "Solidworks"],
            "icp_score": [0.9, 0.1],
            "bought_in_division": [0, 0],
            "customer_id": [1, 2],
            "cutoff_date": ["2024-12-31", "2024-12-31"],
            "prediction_window_months": [6, 6],
        }
    )
    scores_path = tmp_path / "icp_scores.csv"
    data.to_csv(scores_path, index=False)

    monkeypatch.setattr(validate_holdout, "OUTPUTS_DIR", tmp_path)
    cfg = SimpleNamespace(
        validation=SimpleNamespace(holdout_source="csv"),
        run=SimpleNamespace(cutoff_date="2024-12-31", prediction_window_months=6),
        database=SimpleNamespace(source_tables={}),
        etl=SimpleNamespace(source_columns={}),
    )
    monkeypatch.setattr(validate_holdout, "load_config", lambda: cfg)
    monkeypatch.setattr(
        validate_holdout,
        "load_holdout_buyers",
        lambda *args, **kwargs: HoldoutData(pd.Series([1], dtype="Int64"), None, "csv"),
    )

    out = validate_holdout.validate_holdout(scores_path, year_tag="2025")
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["status"] in {"ok", "fail"}
    assert payload["gates"]["auc"] == 0.70
    assert payload["divisions"][0]["division_name"] == "Solidworks"
    assert payload["divisions"][0].get("holdout_applied") is True
