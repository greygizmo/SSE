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

    # Read the fact_transactions table from the database
    fact_transactions = pl.read_database(
        "SELECT customer_id, product_sku FROM fact_transactions",
        engine,
    )

    # Create a basket for each customer
    basket = (
        fact_transactions.lazy()
        .group_by(["customer_id", "product_sku"])
        .agg(pl.count().alias("count"))
        .collect()
    )

    # Create a one-hot encoded matrix
    basket_plus = (
        basket.to_dummies(columns=["product_sku"])
        .group_by("customer_id")
        .agg(pl.sum(col) for col in basket.columns if col.startswith("product_sku_"))
    )

    # Perform market basket analysis
    frequent_itemsets = apriori(basket_plus.drop("customer_id").to_pandas(), min_support=0.001, use_colnames=True)
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
