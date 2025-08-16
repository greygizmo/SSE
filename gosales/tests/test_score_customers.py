import json
import numpy as np
import pandas as pd
import polars as pl

from gosales.pipeline import score_customers as sc


def test_score_without_bought_in_division(tmp_path, monkeypatch):
    feature_df = pl.DataFrame({"customer_id": [1, 2], "feat": [0.1, 0.2]})

    def fake_create_feature_matrix(engine, division_name, cutoff, window):
        return feature_df

    monkeypatch.setattr(sc, "create_feature_matrix", fake_create_feature_matrix)

    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8]] * len(X))

    monkeypatch.setattr(sc.mlflow.sklearn, "load_model", lambda path: DummyModel())
    monkeypatch.setattr(
        pd,
        "read_sql",
        lambda query, engine, params=None: pd.DataFrame(
            {"customer_id": [1, 2], "customer_name": ["A", "B"]}
        ),
    )

    model_dir = tmp_path / "dummy_model"
    model_dir.mkdir()
    metadata = {
        "cutoff_date": "2024-01-01",
        "prediction_window_months": 1,
        "feature_names": ["feat"],
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata))

    result = sc.score_customers_for_division(None, "Test", model_dir)
    assert not result.is_empty()
    assert "icp_score" in result.columns
