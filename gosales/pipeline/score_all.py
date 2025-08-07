from gosales.utils.db import get_db_connection
from gosales.etl.load_csv import load_csv_to_db
from gosales.etl.build_star import build_star_schema
from gosales.models.train_division_model import train_division_model
from gosales.pipeline.score_customers import generate_scoring_outputs
from gosales.utils.logger import get_logger
from gosales.utils.paths import DATA_DIR

logger = get_logger(__name__)

def score_all():
    """
    Orchestrates the entire GoSales pipeline from data ingestion to final scoring.
    
    This master script executes the following steps in order:
    1.  Loads all raw CSV data into a staging table in the database.
    2.  Builds the clean, tidy star schema (`dim_customer`, `fact_transactions`).
    3.  Trains a new machine learning model for the 'Solidworks' division.
    4.  Generates and saves the final ICP scores and whitespace opportunities.
    """
    logger.info("Starting the full GoSales scoring pipeline...")

    # --- 1. Setup ---
    db_engine = get_db_connection()
    target_division = "Solidworks"

    # --- 2. ETL Phase ---
    logger.info("--- Phase 1: ETL ---")
    # Define the CSV files and their corresponding table names
    csv_files = {
        "Sales_Log.csv": "sales_log",
    }
    for file_name, table_name in csv_files.items():
        file_path = DATA_DIR / "database_samples" / file_name
        load_csv_to_db(file_path, table_name, db_engine)
    
    build_star_schema(db_engine)
    logger.info("--- ETL Phase Complete ---")

    # --- 3. Model Training Phase ---
    logger.info(f"--- Phase 2: Training model for {target_division} division ---")
    # Use 2024-12-31 as cutoff, predict 6 months into 2025
    train_division_model(db_engine, target_division, cutoff_date="2024-12-31", prediction_window_months=6)
    logger.info("--- Model Training Phase Complete ---")

    # --- 4. Scoring Phase ---
    logger.info("--- Phase 3: Generating Scores and Whitespace ---")
    generate_scoring_outputs(db_engine)
    logger.info("--- Scoring Phase Complete ---")

    logger.info("GoSales scoring pipeline finished successfully!")


if __name__ == "__main__":
    score_all()
