"""Train ALS recommenders to surface whitespace opportunities per customer."""

import polars as pl
import implicit
from scipy.sparse import coo_matrix, csr_matrix
from threadpoolctl import threadpool_limits
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

    # Read transactions: prefer fact_transactions (product_sku), fallback to legacy fact_orders (product_name)
    try:
        tx = pl.read_database("SELECT customer_id, product_sku AS item FROM fact_transactions", engine)
    except Exception:
        tx = pl.read_database("SELECT customer_id, product_name AS item FROM fact_orders", engine)

    # Create a user-item interaction table (counts)
    user_item = (
        tx.lazy()
        .group_by(["customer_id", "item"])
        .agg(pl.len().alias("count"))
        .collect()
    )

    # Build explicit, deterministic mappings for readable outputs
    raw_customer_ids = user_item["customer_id"].to_list()
    user_ids = sorted(set(raw_customer_ids), key=lambda uid: str(uid))
    item_names = sorted(set(user_item["item"].to_list()))
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_name_to_idx = {name: idx for idx, name in enumerate(item_names)}

    user_codes = [user_id_to_idx[uid] for uid in raw_customer_ids]
    item_codes = [item_name_to_idx[str(name)] for name in user_item["item"].to_list()]
    counts = user_item["count"].to_list()

    # Create a sparse matrix (users x items) in CSR to satisfy implicit's expectations
    sparse_matrix = csr_matrix(
        coo_matrix((counts, (user_codes, item_codes)), shape=(len(user_ids), len(item_names)))
    )

    # Train the ALS model (deterministic)
    model = implicit.als.AlternatingLeastSquares(factors=50, random_state=42)
    with threadpool_limits(1, "blas"):
        model.fit(sparse_matrix)

    # Generate top-N recommendations per user (map indices back to readable ids/names)
    records = []
    for user_idx, user_id in enumerate(user_ids):
        item_indices, scores = model.recommend(user_idx, sparse_matrix[user_idx], N=top_n)
        for item_idx, score in zip(item_indices, scores):
            records.append({
                "customer_id": user_id,
                "product_name": item_names[int(item_idx)],
                "score": float(score),
            })

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
