import pandas as pd
from sqlalchemy import create_engine, inspect


def test_fact_transactions_table_exists(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/fact.db")
    sample = pd.DataFrame(
        {
            "customer_id": [1],
            "order_date": ["2024-01-01"],
            "product_sku": ["SWX_Core"],
            "product_division": ["Solidworks"],
            "gross_profit": [100.0],
            "quantity": [1],
        }
    )
    sample.to_sql("fact_transactions", engine, index=False, if_exists="replace")

    inspector = inspect(engine)
    assert "fact_transactions" in inspector.get_table_names()
