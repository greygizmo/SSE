import polars as pl
from sqlalchemy import text
from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger

logger = get_logger(__name__)

def build_star_schema(engine):
    """Builds the star schema from the raw data.

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
    """
    logger.info("Building star schema...")

    # Read the raw data from the database
    sales_log = pl.read_database("select * from sales_log", engine)
    analytics_sales_logs = pl.read_database(
        "select * from analytics_sales_logs", engine
    )

    # Create the dim_customer table
    logger.info("Creating dim_customer table...")
    dim_customer = (
        sales_log.lazy()
        .select(["Sales Log[Customer]", "Sales Log[CustomerId]"])
        .unique()
        .rename({"Sales Log[Customer]": "customer_name", "Sales Log[CustomerId]": "customer_id"})
        .collect()
    )
    dim_customer.write_database(
        "dim_customer", engine, if_table_exists="replace"
    )
    logger.info("Successfully created dim_customer table.")

    # Create the dim_product table
    logger.info("Creating dim_product table...")
    dim_product = (
        analytics_sales_logs.lazy()
        .select(["Analytics_SalesLogs[Description]"])
        .unique()
        .rename({"Analytics_SalesLogs[Description]": "product_name"})
        .collect()
    )
    dim_product.write_database(
        "dim_product", engine, if_table_exists="replace"
    )
    logger.info("Successfully created dim_product table.")

    # Create the fact_orders table
    logger.info("Creating fact_orders table...")
    fact_orders = (
        sales_log.lazy()
        .join(
            analytics_sales_logs.lazy(),
            left_on="Sales Log[Id]",
            right_on="Analytics_SalesLogs[Id]",
            how="inner",
        )
        .select(
            [
                "Sales Log[Id]",
                "Sales Log[CustomerId]",
                "Analytics_SalesLogs[Description]",
                "Sales Log[Revenue]",
                "Sales Log[Rec Date]",
            ]
        )
        .rename(
            {
                "Sales Log[Id]": "order_id",
                "Sales Log[CustomerId]": "customer_id",
                "Analytics_SalesLogs[Description]": "product_name",
                "Sales Log[Revenue]": "revenue",
                "Sales Log[Rec Date]": "order_date",
            }
        )
        .collect()
    )
    fact_orders.write_database(
        "fact_orders", engine, if_table_exists="replace"
    )
    logger.info("Successfully created fact_orders table.")


if __name__ == "__main__":
    # Get database connection
    db_engine = get_db_connection()

    # Build the star schema
    build_star_schema(db_engine)
