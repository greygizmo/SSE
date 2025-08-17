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

    # Read the fact_orders table from the database
    fact_orders = pl.read_database("select * from fact_orders", engine)

    # Create a basket for each customer
    basket = (
        fact_orders.lazy()
        .group_by(["customer_id", "product_name"])
        .agg(pl.count().alias("count"))
        .collect()
    )

    # Create a one-hot encoded matrix
    basket_plus = (
        basket.to_dummies(columns=["product_name"])
        .group_by("customer_id")
        .agg(pl.all().exclude(["customer_id", "count"]).sum())
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
