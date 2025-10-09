"""Train ALS recommenders to surface whitespace opportunities per customer."""

from __future__ import annotations

import math
from typing import Any, Tuple

import implicit
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix
from threadpoolctl import threadpool_limits

from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR

logger = get_logger(__name__)

_MISSING_ITEM_SENTINEL = "[missing-item]"
_NAN_ITEM_SENTINEL = "[nan-item]"


def _is_nan(value: Any) -> bool:
    """Return True if value behaves like NaN."""
    try:
        return math.isnan(value)
    except TypeError:
        return False


def _item_display(value: Any) -> str:
    """Generate a readable representation for ALS item identifiers."""
    if value is None:
        return _MISSING_ITEM_SENTINEL
    if _is_nan(value):
        return _NAN_ITEM_SENTINEL
    return str(value)


def _item_key(value: Any) -> Tuple[str, str]:
    """Construct a deterministic key for deduplicating ALS items."""
    display = _item_display(value)
    return (type(value).__name__, display)


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
        tx = pl.read_database(
            "SELECT customer_id, product_sku AS item FROM fact_transactions", engine
        )
    except Exception:
        tx = pl.read_database(
            "SELECT customer_id, product_name AS item FROM fact_orders", engine
        )

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
    raw_items = user_item["item"].to_list()

    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_keys = [_item_key(item) for item in raw_items]
    # Sorting tuples keeps ordering deterministic even when original values differ in type.
    unique_item_keys = sorted(set(item_keys))
    item_key_to_idx = {key: idx for idx, key in enumerate(unique_item_keys)}
    readable_item_names = [display for _, display in unique_item_keys]

    user_codes = [user_id_to_idx[uid] for uid in raw_customer_ids]
    item_codes = [item_key_to_idx[key] for key in item_keys]
    counts = user_item["count"].to_list()

    # Create a sparse matrix (users x items) in CSR to satisfy implicit's expectations
    sparse_matrix = csr_matrix(
        coo_matrix(
            (counts, (user_codes, item_codes)),
            shape=(len(user_ids), len(unique_item_keys)),
        )
    )

    # Train the ALS model (deterministic)
    model = implicit.als.AlternatingLeastSquares(factors=50, random_state=42)
    with threadpool_limits(1, "blas"):
        model.fit(sparse_matrix)

    # Generate top-N recommendations per user (map indices back to readable ids/names)
    records = []
    for user_idx, user_id in enumerate(user_ids):
        item_indices, scores = model.recommend(
            user_idx, sparse_matrix[user_idx], N=top_n
        )
        for item_idx, score in zip(item_indices, scores):
            records.append(
                {
                    "customer_id": user_id,
                    "product_name": readable_item_names[int(item_idx)],
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
