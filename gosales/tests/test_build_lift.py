import pandas as pd
import polars as pl
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori, association_rules

def test_product_indicators_and_rules_non_empty():
    engine = create_engine("sqlite:///:memory:")
    data = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3],
            "product_name": ["A", "B", "A", "B", "C"],
        }
    )
    data.to_sql("fact_orders", engine, index=False)

    fact_orders = pl.read_database("select * from fact_orders", engine)
    basket = (
        fact_orders.lazy()
        .group_by(["customer_id", "product_name"])
        .agg(pl.len().alias("count"))
        .collect()
    )

    basket_plus = (
        basket.to_dummies(columns=["product_name"])
        .group_by("customer_id")
        .agg(pl.all().exclude(["customer_id", "count"]).sum())
    )

    assert any(col.startswith("product_name_") for col in basket_plus.columns)

    frequent_itemsets = apriori(
        basket_plus.drop("customer_id").to_pandas().astype(bool),
        min_support=0.001,
        use_colnames=True,
    )
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    assert not rules.empty
