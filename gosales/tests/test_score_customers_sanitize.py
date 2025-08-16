import json
import numpy as np
import pandas as pd
import polars as pl
import joblib
import mlflow.sklearn

from gosales.pipeline.score_customers import score_customers_for_division


class _DummyModel:
    def predict_proba(self, X: pd.DataFrame):
        # Expect fully numeric DataFrame with no missing values
        assert X.isnull().sum().sum() == 0
        assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


def test_score_handles_strings_and_nans(monkeypatch, tmp_path):
    model_dir = tmp_path / "dummy_model"
    model_dir.mkdir()
    meta = {
        "division": "Dummy",
        "cutoff_date": "2024-01-01",
        "prediction_window_months": 6,
        "feature_names": ["f1", "f2"],
        "feature_dtypes": {"f1": "float64", "f2": "float64"},
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta))
    joblib.dump(_DummyModel(), model_dir / "model.pkl")

    # Force joblib loader by making mlflow loader fail
    monkeypatch.setattr(
        mlflow.sklearn,
        "load_model",
        lambda path: (_ for _ in ()).throw(RuntimeError("mlflow unavailable")),
    )

    def _mock_feature_matrix(engine, division_name, cutoff, window_months):
        df = pd.DataFrame(
            {
                "customer_id": [1, 2],
                "bought_in_division": [0, 1],
                "f1": ["1.5", "bad"],
                "f2": [np.nan, "3"],
            }
        )
        return pl.from_pandas(df)

    monkeypatch.setattr(
        "gosales.pipeline.score_customers.create_feature_matrix",
        _mock_feature_matrix,
    )

    monkeypatch.setattr(
        "gosales.pipeline.score_customers.pd.read_sql",
        lambda *args, **kwargs: pd.DataFrame(
            {"customer_id": [1, 2], "customer_name": ["A", "B"]}
        ),
    )

    result = score_customers_for_division(None, "Dummy", model_dir)
    assert result.height == 2
    assert "icp_score" in result.columns

