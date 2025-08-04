from gosales.utils.db import get_db_connection
from gosales.etl.load_csv import load_csv_to_db
from gosales.etl.build_star import build_star_schema
from gosales.features.engine import create_feature_matrix
from gosales.models.train_simulation import train_simulation_model
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

    # Create the feature matrix for the "Simulation" product
    feature_matrix = create_feature_matrix(db_engine, "Simulation")

    # Train the Simulation model
    train_simulation_model(db_engine)

    # Build the lift
    lift_output_path = OUTPUTS_DIR / "lift.csv"
    build_lift(db_engine, lift_output_path)

    # Build the ALS model
    als_output_path = OUTPUTS_DIR / "als_recommendations.csv"
    build_als(db_engine, als_output_path)

    logger.info("Scoring pipeline finished.")


if __name__ == "__main__":
    score_all()
