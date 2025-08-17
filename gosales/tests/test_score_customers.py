from __future__ import annotations

import json

import polars as pl
import numpy as np

from gosales.pipeline import score_customers


def test_score_customers_no_features(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    meta = {
        "cutoff_date": "2024-01-01",
        "prediction_window_months": 3,
        "class_balance": {"positives": 0},
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta))

    class DummyModel:
        def predict_proba(self, X):  # pragma: no cover - not executed
            return np.zeros((len(X), 2))

    monkeypatch.setattr(score_customers.mlflow.sklearn, "load_model", lambda path: DummyModel())

    feature_df = pl.DataFrame({"customer_id": [1, 2], "bought_in_division": [0, 0]})
    monkeypatch.setattr(score_customers, "create_feature_matrix", lambda *a, **k: feature_df)

    run_manifest = {}
    result = score_customers.score_customers_for_division(
        None, "division", model_dir, run_manifest=run_manifest
    )
    assert result.is_empty()
    assert run_manifest["alerts"][0]["code"] == "NO_FEATURE_COLUMNS"

