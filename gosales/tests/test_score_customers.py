from __future__ import annotations

import json
import pandas as pd
import polars as pl
from sqlalchemy import create_engine
from sklearn.dummy import DummyClassifier
import joblib

from gosales.pipeline.score_customers import score_customers_for_division


def test_score_customers_dedupes_names(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/db.sqlite")

    # dim_customer with duplicate customer_id entries
    pd.DataFrame(
        [
            {"customer_id": 1, "customer_name": "Acme"},
            {"customer_id": 1, "customer_name": "Acme"},
            {"customer_id": 2, "customer_name": "Globex"},
        ]
    ).to_sql("dim_customer", engine, index=False, if_exists="replace")

    # patch feature matrix creation to return simple dataframe
    def fake_create_feature_matrix(engine, division_name, cutoff, window):
        return pl.DataFrame(
            {
                "customer_id": [1, 2],
                "bought_in_division": [0, 1],
                "f": [0.1, 0.9],
            }
        )

    monkeypatch.setattr(
        "gosales.pipeline.score_customers.create_feature_matrix", fake_create_feature_matrix
    )

    # ensure mlflow path load fails so joblib fallback is used
    def _raise(*args, **kwargs):
        raise RuntimeError("mlflow not available")

    monkeypatch.setattr(
        "gosales.pipeline.score_customers.mlflow.sklearn.load_model", _raise
    )

    # simple model that outputs constant probabilities
    model = DummyClassifier(strategy="prior")
    model.fit([[0], [1]], [0, 1])

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    joblib.dump(model, model_dir / "model.pkl")
    metadata = {
        "division": "TestDivision",
        "cutoff_date": "2024-01-01",
        "prediction_window_months": 1,
        "feature_names": ["f"],
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    scores = score_customers_for_division(engine, "TestDivision", model_dir)
    pdf = scores.to_pandas()

    assert len(pdf) == 2
    assert pdf["customer_id"].nunique() == 2
