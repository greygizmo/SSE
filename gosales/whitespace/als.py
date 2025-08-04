import polars as pl
import implicit
from scipy.sparse import coo_matrix
from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR

logger = get_logger(__name__)

def build_als(engine, output_path):
    """Uses alternating least squares to find whitespace opportunities.

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
        output_path (str): The path to the output CSV file.
    """
    logger.info("Building ALS model...")

    # Read the fact_orders table from the database
    fact_orders = pl.read_database("select * from fact_orders", engine)

    # Create a user-item matrix
    user_item = (
        fact_orders.lazy()
        .group_by(["customer_id", "product_name"])
        .agg(pl.len().alias("count"))
        .collect()
    )

    # Create a sparse matrix
    sparse_matrix = coo_matrix(
        (
            user_item["count"],
            (
                user_item["customer_id"].cast(pl.Categorical).to_physical(),
                user_item["product_name"].cast(pl.Categorical).to_physical(),
            ),
        )
    )

    # Train the ALS model
    model = implicit.als.AlternatingLeastSquares(factors=50)
    model.fit(sparse_matrix)

    # Get the user and item factors
    user_factors = model.user_factors
    item_factors = model.item_factors

    # Get the recommendations
    recommendations = model.recommend_all(sparse_matrix)

    # Save the recommendations to a CSV file
    pl.DataFrame(recommendations).write_csv(output_path)

    logger.info(f"Successfully built ALS model and saved to {output_path}")


if __name__ == "__main__":
    # Get database connection
    db_engine = get_db_connection()

    # Define the output path for the ALS recommendations CSV file
    output_path = OUTPUTS_DIR / "als_recommendations.csv"

    # Create the outputs directory if it doesn't exist
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Build the ALS model
    build_als(db_engine, output_path)
