"""Compute market-basket lift tables that power whitespace prioritization."""

import polars as pl
from mlxtend.frequent_patterns import apriori, association_rules
from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR

logger = get_logger(__name__)


def build_lift(engine, output_path):
    """Calculates the market basket lift for each product combination.

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
        output_path (str): The path to the output CSV file.
    """
    logger.info("Building lift...")

    # Read transactions; prefer fact_transactions(product_sku), fallback to legacy fact_orders(product_name)
    try:
        fact_transactions = pl.read_database(
            "SELECT customer_id, product_sku FROM fact_transactions",
            engine,
        )
        item_col = "product_sku"
        src = fact_transactions
    except Exception:
        fact_orders = pl.read_database(
            "SELECT customer_id, product_name FROM fact_orders", engine
        )
        fact_orders = fact_orders.rename({"product_name": "product_sku"})
        item_col = "product_sku"
        src = fact_orders

    # Create a basket for each customer
    basket = (
        src.lazy()
        .group_by(["customer_id", item_col])
        .agg(pl.len().alias("count"))
        .collect()
    )

    # Create a one-hot encoded boolean matrix per customer (avoid mlxtend deprecation on non-bool)
    basket_plus = (
        basket.to_dummies(columns=[item_col])
        .group_by("customer_id")
        .agg(pl.all().exclude(["customer_id"]).sum())
    )

    count_columns = [col for col in basket_plus.columns if col.startswith("count")]
    if count_columns:
        basket_plus = basket_plus.drop(count_columns)

    basket_plus = basket_plus.with_columns(
        (pl.all().exclude("customer_id") > 0).cast(pl.Boolean)
    )

    # Perform market basket analysis
    frequent_itemsets = apriori(
        basket_plus.drop("customer_id").to_pandas().astype(bool),
        min_support=0.001,
        use_colnames=True,
    )
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Save the rules to a CSV file
    rules.to_csv(output_path, index=False)

    logger.info(f"Successfully built lift and saved to {output_path}")


if __name__ == "__main__":
    # Get database connection
    db_engine = get_db_connection()

    # Define the output path for the lift CSV file
    output_path = OUTPUTS_DIR / "lift.csv"

    # Create the outputs directory if it doesn't exist
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Build the lift
    build_lift(db_engine, output_path)
