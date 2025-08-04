import polars as pl
from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger

logger = get_logger(__name__)

def create_feature_matrix(engine, product_name: str):
    """Creates a feature matrix for a given product.

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
        product_name (str): The name of the product to create the feature matrix for.

    Returns:
        polars.DataFrame: The feature matrix.
    """
    logger.info(f"Creating feature matrix for product: {product_name}...")

    # Read the fact_orders table from the database
    fact_orders = pl.read_database("select * from fact_orders", engine)

    # Filter the orders for the given product
    product_orders = fact_orders.filter(pl.col("product_name") == product_name)

    # Create the feature matrix
    feature_matrix = (
        product_orders.lazy()
        .group_by("customer_id")
        .agg(
            [
                pl.sum("revenue").alias("total_spend"),
                pl.count().alias("total_orders"),
                (pl.max("order_date") - pl.min("order_date")).alias(
                    "days_since_first_order"
                ),
            ]
        )
        .collect()
    )

    logger.info(f"Successfully created feature matrix for product: {product_name}.")

    return feature_matrix


if __name__ == "__main__":
    # Get database connection
    db_engine = get_db_connection()

    # Create the feature matrix for the "Simulation" product
    feature_matrix = create_feature_matrix(db_engine, "Simulation")

    # Print the feature matrix
    print(feature_matrix.to_pandas().to_string().encode("utf-8"))
