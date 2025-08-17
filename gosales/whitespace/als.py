import polars as pl
import implicit
from scipy.sparse import coo_matrix
from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR

logger = get_logger(__name__)


def build_als(engine, output_path, top_n: int = 10):
    """Uses alternating least squares to find whitespace opportunities.

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
        output_path (str): The path to the output CSV file.
        top_n (int, optional): Number of recommendations to generate per user.
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

    # Build mappings between ids and indices
    user_ids = user_item["customer_id"].unique().to_list()
    product_names = user_item["product_name"].unique().to_list()
    user_mapping = {uid: idx for idx, uid in enumerate(user_ids)}
    product_mapping = {pname: idx for idx, pname in enumerate(product_names)}

    user_item = user_item.with_columns(
        pl.col("customer_id").map_elements(user_mapping.get).alias("user_idx"),
        pl.col("product_name").map_elements(product_mapping.get).alias("item_idx"),
    )

    # Create a sparse matrix (users x items)
    sparse_matrix = coo_matrix(
        (
            user_item["count"],
            (user_item["user_idx"], user_item["item_idx"]),
        ),
        shape=(len(user_ids), len(product_names)),
    ).tocsr()

    # Train the ALS model
    model = implicit.als.AlternatingLeastSquares(factors=50)
    model.fit(sparse_matrix)

    # Generate top-N recommendations per user
    recommendations = []
    for user_idx, user_id in enumerate(user_ids):
        item_indices, scores = model.recommend(user_idx, sparse_matrix[user_idx], N=top_n)
        for item_idx, score in zip(item_indices, scores):
            recommendations.append(
                {
                    "customer_id": user_id,
                    "product_name": product_names[item_idx],
                    "score": float(score),
                }
            )

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
