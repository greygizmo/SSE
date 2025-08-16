import pandas as pd
import polars as pl
import numpy as np
from sqlalchemy import create_engine, inspect

from gosales.pipeline import validate_holdout


def test_validate_holdout_restores_fact_table(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{tmp_path}/test.db")

    # Seed original fact_transactions and related tables
    orig_fact = pd.DataFrame(
        {
            "customer_id": [1],
            "order_date": ["2024-01-01"],
            "product_sku": ["SWX_Core"],
            "product_division": ["Solidworks"],
            "gross_profit": [100.0],
            "quantity": [1],
        }
    )
    orig_fact.to_sql("fact_transactions", eng, index=False, if_exists="replace")
    pd.DataFrame({"customer_id": [1]}).to_sql("dim_customer", eng, index=False, if_exists="replace")

    sales_log = pd.DataFrame(
        {
            "CustomerId": [1],
            "Rec Date": ["2024-01-01"],
            "Division": ["Solidworks"],
            "SWX_Core": [100],
            "SWX_Core_Qty": [1],
        }
    )
    sales_log.to_sql("sales_log", eng, index=False, if_exists="replace")

    data_dir = tmp_path / "data"
    holdout_dir = data_dir / "holdout"
    holdout_dir.mkdir(parents=True)
    holdout_df = pd.DataFrame(
        {
            "CustomerId": [1],
            "Rec Date": ["2025-02-01"],
            "Division": ["Solidworks"],
            "SWX_Core": [200],
            "SWX_Core_Qty": [1],
        }
    )
    holdout_path = holdout_dir / "Sales Log 2025 YTD.csv"
    holdout_df.to_csv(holdout_path, index=False)

    # Patch external dependencies
    monkeypatch.setattr(validate_holdout, "DATA_DIR", data_dir)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(validate_holdout, "OUTPUTS_DIR", out_dir)
    monkeypatch.setattr(validate_holdout, "get_db_connection", lambda: eng)

    def fake_load_csv(path, table, engine):
        pd.read_csv(path).to_sql(table, engine, index=False, if_exists="replace")

    monkeypatch.setattr(validate_holdout, "load_csv_to_db", fake_load_csv)
    monkeypatch.setattr(
        validate_holdout,
        "get_sku_mapping",
        lambda: {"SWX_Core": {"qty_col": "SWX_Core_Qty", "division": "Solidworks"}},
    )
    monkeypatch.setattr(
        validate_holdout,
        "create_feature_matrix",
        lambda engine, div, cutoff_date=None, prediction_window_months=6: pl.DataFrame(
            {"customer_id": [1], "bought_in_division": [1], "feat": [0.1]}
        ),
    )

    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.0, 1.0] for _ in range(len(X))])

    monkeypatch.setattr(validate_holdout.mlflow.sklearn, "load_model", lambda p: DummyModel())
    monkeypatch.setattr(validate_holdout, "roc_auc_score", lambda y, p: 1.0)
    monkeypatch.setattr(
        validate_holdout,
        "classification_report",
        lambda y, yp, output_dict: {"1": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0}},
    )
    monkeypatch.setattr(validate_holdout, "confusion_matrix", lambda y, yp: np.array([[1, 0], [0, 1]]))

    validate_holdout.validate_against_holdout()

    restored = pd.read_sql("fact_transactions", eng)
    pd.testing.assert_frame_equal(restored.sort_index(axis=1), orig_fact.sort_index(axis=1))
    assert not inspect(eng).has_table("fact_transactions_temp")
