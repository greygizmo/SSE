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

    # Encode users and items to indices while retaining mappings
    user_ids = user_item["customer_id"].to_list()
    product_names = user_item["product_name"].to_list()

    user_encoding = {u: i for i, u in enumerate(sorted(set(user_ids)))}
    item_encoding = {p: i for i, p in enumerate(sorted(set(product_names)))}

    user_mapping = {i: u for u, i in user_encoding.items()}
    item_mapping = {i: p for p, i in item_encoding.items()}

    user_codes = pl.Series([user_encoding[u] for u in user_ids])
    item_codes = pl.Series([item_encoding[p] for p in product_names])

    # Create a sparse matrix and convert to CSR for implicit
    counts = user_item["count"].to_list()
    sparse_matrix = coo_matrix((counts, (user_codes.to_list(), item_codes.to_list()))).tocsr()

    # Train the ALS model
    model = implicit.als.AlternatingLeastSquares(factors=50)
    model.fit(sparse_matrix)

    # Get the user and item factors
    user_factors = model.user_factors
    item_factors = model.item_factors

    # Get the recommendations
    recommendations = model.recommend_all(sparse_matrix)

    # Map recommendations back to IDs and names
    records = []
    for user_idx, item_indices in enumerate(recommendations):
        cid = user_mapping.get(user_idx)
        for item_idx in item_indices:
            pname = item_mapping.get(item_idx)
            records.append({"customer_id": cid, "product_name": pname})

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
