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

    # Read the fact_transactions table from the database
    fact_transactions = pl.read_database(
        "SELECT customer_id, product_sku FROM fact_transactions",
        engine,
    )

    # Create a user-item matrix
    user_item = (
        fact_transactions.lazy()
        .group_by(["customer_id", "product_sku"])
        .agg(pl.len().alias("count"))
        .collect()
    )

    # Create a sparse matrix directly from categorical encodings
    sparse_matrix = coo_matrix(
        (
            user_item["count"],
            (
                user_item["customer_id"].cast(pl.Categorical).to_physical(),
                user_item["product_sku"].cast(pl.Categorical).to_physical(),
            ),
        )
    ).tocsr()

    # Train the ALS model (deterministic)
    model = implicit.als.AlternatingLeastSquares(factors=50, random_state=42)
    model.fit(sparse_matrix)

    # Generate top-N recommendations per user index
    records = []
    n_users = sparse_matrix.shape[0]
    for user_idx in range(n_users):
        item_indices, scores = model.recommend(user_idx, sparse_matrix[user_idx], N=top_n)
        for item_idx, score in zip(item_indices, scores):
            records.append(
                {
                    "customer_id": int(user_idx),
                    "product_name": int(item_idx),
                    "score": float(score),
                }
            )

    # Save the recommendations to a CSV file
    pl.DataFrame(records).write_csv(output_path)

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
