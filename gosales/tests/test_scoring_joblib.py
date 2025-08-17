import json
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

from gosales.pipeline.score_customers import score_customers_for_division


class DummyModel:
    """Simple model that returns a fixed probability for the positive class."""

    def predict_proba(self, X):  # pragma: no cover - trivial
        return np.tile([0.4, 0.6], (len(X), 1))


def _seed(engine):
    fact = pd.DataFrame(
        [
            {
                "customer_id": 1,
                "order_date": "2024-01-01",
                "product_division": "Solidworks",
                "product_sku": "SWX_Core",
                "gross_profit": 100,
            },
            {
                "customer_id": 1,
                "order_date": "2024-02-01",
                "product_division": "Services",
                "product_sku": "Training",
                "gross_profit": 5,
            },
            {
                "customer_id": 2,
                "order_date": "2023-12-15",
                "product_division": "Simulation",
                "product_sku": "Simulation",
                "gross_profit": 50,
            },
        ]
    )
    fact.to_sql("fact_transactions", engine, if_exists="replace", index=False)
    pd.DataFrame(
        {
            "customer_id": [1, 2],
            "customer_name": ["Acme", "Globex"],
        }
    ).to_sql("dim_customer", engine, if_exists="replace", index=False)


def test_scoring_with_joblib(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/score.db")
    _seed(eng)

    model_dir = Path(tmp_path) / "dummy_model"
    model_dir.mkdir()
    joblib.dump(DummyModel(), model_dir / "model.pkl")
    meta = {
        "division": "Solidworks",
        "cutoff_date": "2024-01-31",
        "prediction_window_months": 1,
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    scores = score_customers_for_division(eng, "Solidworks", model_dir)
    assert not scores.is_empty()
    assert "icp_score" in scores.columns
