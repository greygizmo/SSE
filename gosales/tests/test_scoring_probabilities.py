import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import gosales.pipeline.score_customers as sc


def _setup_engine(customer_ids):
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE dim_customer (customer_id INTEGER PRIMARY KEY, customer_name TEXT)"
    )
    for cid in customer_ids:
        cur.execute(
            "INSERT INTO dim_customer VALUES (?, ?)",
            (int(cid), f"Customer {cid}")
        )
    conn.commit()
    return conn


def _write_model(tmp_path: Path, model) -> Path:
    joblib.dump(model, tmp_path / "model.pkl")
    meta = {
        "cutoff_date": "2021-01-01",
        "prediction_window_months": 3,
        "division": "Test",
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    return tmp_path


def test_score_customers_with_predict_proba(monkeypatch, tmp_path):
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)
    model_dir = _write_model(tmp_path, model)

    feature_pl = pl.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "bought_in_division": y.tolist(),
        "x0": X.flatten().tolist(),
    })

    monkeypatch.setattr(sc, "create_feature_matrix", lambda *args, **kwargs: feature_pl)
    monkeypatch.setattr(sc.mlflow.sklearn, "load_model", lambda *a, **k: (_ for _ in ()).throw(Exception("no mlflow")))

    engine = _setup_engine([1, 2, 3, 4])
    result = sc.score_customers_for_division(engine, "Test", model_dir)

    expected = model.predict_proba(X)[:, 1]
    assert np.allclose(result.sort("customer_id")["icp_score"].to_numpy(), expected)


def test_score_customers_with_decision_function(monkeypatch, tmp_path):
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = LinearSVC().fit(X, y)
    model_dir = _write_model(tmp_path, model)

    feature_pl = pl.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "bought_in_division": y.tolist(),
        "x0": X.flatten().tolist(),
    })

    monkeypatch.setattr(sc, "create_feature_matrix", lambda *args, **kwargs: feature_pl)
    monkeypatch.setattr(sc.mlflow.sklearn, "load_model", lambda *a, **k: (_ for _ in ()).throw(Exception("no mlflow")))

    engine = _setup_engine([1, 2, 3, 4])
    result = sc.score_customers_for_division(engine, "Test", model_dir)

    margins = model.decision_function(X)
    expected = 1 / (1 + np.exp(-margins))
    assert np.allclose(result.sort("customer_id")["icp_score"].to_numpy(), expected)
