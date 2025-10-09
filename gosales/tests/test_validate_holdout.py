import json
from pathlib import Path

import pandas as pd

from gosales.pipeline import validate_holdout


def test_validate_against_holdout_deprecated():
    result = validate_holdout.validate_against_holdout()
    assert result["status"] == "deprecated"
    assert "gosales.validation.forward" in result["message"]


def test_validate_holdout_writes_metrics(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "division_name": ["Solidworks", "Solidworks"],
            "icp_score": [0.9, 0.1],
            "bought_in_division": [1, 0],
        }
    )
    scores_path = tmp_path / "icp_scores.csv"
    data.to_csv(scores_path, index=False)

    monkeypatch.setattr(validate_holdout, "OUTPUTS_DIR", tmp_path)

    out = validate_holdout.validate_holdout(scores_path, year_tag="2025")
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["status"] in {"ok", "fail"}
    assert payload["gates"]["auc"] == 0.70
    assert payload["divisions"][0]["division_name"] == "Solidworks"
