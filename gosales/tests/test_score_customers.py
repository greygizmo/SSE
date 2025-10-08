from __future__ import annotations

import json
import pandas as pd
import polars as pl
import pytest
from sqlalchemy import create_engine
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

from gosales.pipeline import score_customers as score_module
from gosales.pipeline.score_customers import score_customers_for_division
from gosales.models.shap_utils import compute_shap_reasons


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


def test_compute_shap_reasons_returns_non_empty_columns_when_shap():
    pytest.importorskip("shap")

    X = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1],
            "f2": [1, 0, 1, 0],
        }
    )
    y = [0, 1, 0, 1]

    model = DecisionTreeClassifier(random_state=0)
    model.fit(X, y)

    reasons = compute_shap_reasons(model, X, feature_cols=["f1", "f2"], top_k=2)

    assert list(reasons.columns) == ["reason_1", "reason_2"]
    assert not reasons.empty
    assert all(reasons[col].notna().any() for col in reasons.columns)


def test_dim_customer_cache_isolation_across_engines(tmp_path):
    score_module._DIM_CUSTOMER_CACHE.clear()

    db1 = tmp_path / "tenant_a.sqlite"
    db2 = tmp_path / "tenant_b.sqlite"
    engine_a = create_engine(f"sqlite:///{db1}")
    engine_b = create_engine(f"sqlite:///{db2}")

    try:
        pd.DataFrame([{"customer_id": 1, "customer_name": "Alpha"}]).to_sql(
            "dim_customer", engine_a, index=False, if_exists="replace"
        )
        pd.DataFrame([{"customer_id": 2, "customer_name": "Beta"}]).to_sql(
            "dim_customer", engine_b, index=False, if_exists="replace"
        )

        df_a = score_module._get_dim_customer(engine_a)
        df_b = score_module._get_dim_customer(engine_b)

        assert df_a["customer_name"].tolist() == ["Alpha"]
        assert df_b["customer_name"].tolist() == ["Beta"]

        # Cached responses must remain stable even if a caller mutates the returned frame.
        df_a.loc[:, "customer_name"] = ["Corrupted"]
        df_b.loc[:, "customer_name"] = ["Corrupted"]

        df_a_cached = score_module._get_dim_customer(engine_a)
        df_b_cached = score_module._get_dim_customer(engine_b)

        assert df_a_cached["customer_name"].tolist() == ["Alpha"]
        assert df_b_cached["customer_name"].tolist() == ["Beta"]
    finally:
        engine_a.dispose()
        engine_b.dispose()
