from sqlalchemy import create_engine
import pandas as pd

from gosales.features.als_embed import customer_als_embeddings


def test_customer_als_embeddings_respects_lookback(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/als.db")
    data = [
        {"customer_id": 1, "order_date": "2023-06-01", "product_sku": "A", "quantity": 1},
        {"customer_id": 2, "order_date": "2023-07-01", "product_sku": "B", "quantity": 1},
        {"customer_id": 3, "order_date": "2021-01-01", "product_sku": "A", "quantity": 1},
    ]
    pd.DataFrame(data).to_sql("fact_transactions", eng, index=False, if_exists="replace")

    all_emb = customer_als_embeddings(eng, "2024-01-01", factors=2, lookback_months=None)
    recent_emb = customer_als_embeddings(eng, "2024-01-01", factors=2, lookback_months=12)

    all_ids = set(all_emb["customer_id"].to_list())
    recent_ids = set(recent_emb["customer_id"].to_list())
    assert 3 in all_ids
    assert 3 not in recent_ids
