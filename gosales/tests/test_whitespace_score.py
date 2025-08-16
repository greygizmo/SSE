from sqlalchemy import create_engine
import pandas as pd
from gosales.pipeline.score_customers import generate_whitespace_opportunities


def _seed(engine):
    transactions = pd.DataFrame([
        {"customer_id": 1, "order_date": "2024-01-01", "product_division": "A", "gross_profit": 100},
        {"customer_id": 1, "order_date": "2024-02-15", "product_division": "B", "gross_profit": 50},
        {"customer_id": 2, "order_date": "2023-12-01", "product_division": "A", "gross_profit": 20},
    ])
    transactions.to_sql("fact_transactions", engine, if_exists="replace", index=False)
    pd.DataFrame({"customer_id": [1, 2]}).to_sql("dim_customer", engine, if_exists="replace", index=False)


def test_whitespace_score_is_continuous(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/ws.db")
    _seed(eng)
    df = generate_whitespace_opportunities(eng)
    assert not df.is_empty()
    scores = df["whitespace_score"].to_list()
    assert min(scores) >= 0.0 and max(scores) <= 1.0
    assert not all(round(s, 1) in {0.5, 0.6, 0.8} for s in scores)
