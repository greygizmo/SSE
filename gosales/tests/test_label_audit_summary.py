from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from gosales.pipeline import label_audit


def _seed(engine) -> None:
    fact = pd.DataFrame(
        [
            # Pre-cutoff activity for cohort sizing
            {
                "customer_id": 1,
                "order_date": "2024-05-01",
                "product_division": "Solidworks",
                "product_sku": "SWX_Core",
            },
            {
                "customer_id": 2,
                "order_date": "2024-05-10",
                "product_division": "Services",
                "product_sku": "Services",
            },
            # Window positives for each division
            {
                "customer_id": 1,
                "order_date": "2024-07-02",
                "product_division": "Solidworks",
                "product_sku": "SWX_Core",
            },
            {
                "customer_id": 2,
                "order_date": "2024-07-03",
                "product_division": "Services",
                "product_sku": "Services",
            },
        ]
    )
    fact.to_sql("fact_transactions", engine, if_exists="replace", index=False)
    dim = pd.DataFrame({"customer_id": [1, 2, 3]})
    dim.to_sql("dim_customer", engine, if_exists="replace", index=False)


def test_label_audit_appends_summaries(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "outputs"
    monkeypatch.setattr(label_audit, "OUTPUTS_DIR", outputs_dir)

    engine = create_engine(f"sqlite:///{tmp_path}/audit.db")
    _seed(engine)

    label_audit.compute_label_audit(engine, "Solidworks", "2024-06-30", 1)
    label_audit.compute_label_audit(engine, "Services", "2024-06-30", 1)

    summary_path = outputs_dir / label_audit.SUMMARY_FILENAME
    assert summary_path.exists()

    summaries = pd.read_csv(summary_path)
    assert set(summaries["division"]) == {"Solidworks", "Services"}
    assert len(summaries) == 2

