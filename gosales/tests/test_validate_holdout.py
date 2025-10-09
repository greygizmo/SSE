import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from gosales.pipeline import validate_holdout
from gosales.validation.holdout_data import HoldoutData


def test_validate_against_holdout_deprecated():
    result = validate_holdout.validate_against_holdout()
    assert result["status"] == "deprecated"
    assert "gosales.validation.forward" in result["message"]


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
