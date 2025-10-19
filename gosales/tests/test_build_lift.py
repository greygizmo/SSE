import pandas as pd
import polars as pl
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori, association_rules

def test_product_indicators_and_rules_non_empty():
    engine = create_engine("sqlite:///:memory:")
    data = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3],
            "product_sku": ["A", "B", "A", "B", "C"],
        }
    )
    data.to_sql("fact_transactions", engine, index=False)

    fact_transactions = pl.read_database("select * from fact_transactions", engine)
    basket = (
        fact_transactions.lazy()
        .group_by(["customer_id", "product_sku"])
        .agg(pl.len().alias("count"))
        .collect()
    )

    basket_plus = (
        basket.to_dummies(columns=["product_sku"])
        .group_by("customer_id")
        .agg(pl.all().exclude(["customer_id", "count"]).sum())
    )

    assert any(col.startswith("product_sku_") for col in basket_plus.columns)

    frequent_itemsets = apriori(
        basket_plus.drop("customer_id").to_pandas().astype(bool),
        min_support=0.001,
        use_colnames=True,
    )
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    assert not rules.empty
