from gosales.utils.db import get_db_connection
from gosales.etl.load_csv import load_csv_to_db
from gosales.etl.build_star import build_star_schema
from gosales.features.engine import create_feature_matrix
from gosales.models.train_simulation import train_supplies_model
from gosales.pipeline.score_customers import generate_scoring_outputs
from gosales.whitespace.build_lift import build_lift
from gosales.whitespace.als import build_als
from gosales.utils.logger import get_logger
from gosales.utils.paths import DATA_DIR, OUTPUTS_DIR

logger = get_logger(__name__)

def score_all():
    """Orchestrates the entire pipeline, from data ingestion to scoring."""
    logger.info("Starting the scoring pipeline...")

    # Get database connection
    db_engine = get_db_connection()

    # Load the CSV files into the database
    csv_files = {
        "Analytics_order_tags.csv": "analytics_order_tags",
        "Analytics_SalesLogs.csv": "analytics_sales_logs",
        "Sales_Log.csv": "sales_log",
    }
    for file_name, table_name in csv_files.items():
        file_path = DATA_DIR / "database_samples" / file_name
        load_csv_to_db(file_path, table_name, db_engine)

    # Build the star schema
    build_star_schema(db_engine)

    # Create the feature matrix for the "Supplies" product (most common in our data)
    feature_matrix = create_feature_matrix(db_engine, "Supplies")

    # Train the Supplies model (renamed from Simulation for our data)
    train_supplies_model(db_engine)

    # ADDED: Generate customer ICP scores and whitespace opportunities
    generate_scoring_outputs(db_engine)

    # Build the lift (legacy whitespace analysis)
    try:
        lift_output_path = OUTPUTS_DIR / "lift.csv"
        build_lift(db_engine, lift_output_path)
    except Exception as e:
        logger.warning(f"Lift analysis failed: {e}")

    # Build the ALS model (legacy recommendation system)
    try:
        als_output_path = OUTPUTS_DIR / "als_recommendations.csv"
        build_als(db_engine, als_output_path)
    except Exception as e:
        logger.warning(f"ALS analysis failed: {e}")

    logger.info("Complete scoring pipeline finished successfully!")


if __name__ == "__main__":
    score_all()
