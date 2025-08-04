import polars as pl
from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger

logger = get_logger(__name__)

def create_feature_matrix(engine, product_name: str):
    """Creates a feature matrix for ML training with proper binary target.

    FIXED: Now creates features for ALL customers with binary target:
    - 1 if customer bought the product, 0 if they didn't
    - Includes features for all customers, not just buyers

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
        product_name (str): The name of the product to create the feature matrix for.

    Returns:
        polars.DataFrame: The feature matrix with 'bought_product' target column.
    """
    logger.info(f"Creating feature matrix for product: {product_name}...")

    # Read the fact_orders table from the database
    fact_orders = pl.read_database("select * from fact_orders", engine)

    # Get all unique customers
    all_customers = fact_orders.select("customer_id").unique()
    
    # Create features for all customers (overall spending behavior)
    customer_features = (
        fact_orders.lazy()
        .group_by("customer_id")
        .agg(
            [
                pl.sum("revenue").alias("total_spend_all_products"),
                pl.len().alias("total_orders_all_products"),  # Fixed: use pl.len() instead of pl.count()
                pl.n_unique("product_name").alias("unique_products_bought"),
            ]
        )
        .collect()
    )

    # Identify customers who bought the specific product
    product_buyers = (
        fact_orders.filter(pl.col("product_name") == product_name)
        .select("customer_id")
        .unique()
        .with_columns(pl.lit(1).alias("bought_product"))
    )

    # Create the final feature matrix with binary target
    feature_matrix = (
        customer_features.lazy()
        .join(product_buyers.lazy(), on="customer_id", how="left")
        .with_columns(
            pl.col("bought_product").fill_null(0)  # 0 for customers who didn't buy
        )
        .collect()
    )

    logger.info(f"Successfully created feature matrix for product: {product_name}.")
    logger.info(f"Total customers: {feature_matrix.height}")
    logger.info(f"Customers who bought {product_name}: {feature_matrix.filter(pl.col('bought_product') == 1).height}")
    logger.info(f"Customers who didn't buy {product_name}: {feature_matrix.filter(pl.col('bought_product') == 0).height}")

    return feature_matrix


if __name__ == "__main__":
    # Get database connection
    db_engine = get_db_connection()

    # Create the feature matrix for the "Simulation" product
    feature_matrix = create_feature_matrix(db_engine, "Simulation")

    # Print the feature matrix
    print(feature_matrix.to_pandas().to_string().encode("utf-8"))
