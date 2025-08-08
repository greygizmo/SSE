from gosales.utils.db import get_db_connection
from gosales.etl.load_csv import load_csv_to_db
from gosales.etl.build_star import build_star_schema
from gosales.models.train_division_model import train_division_model
from gosales.pipeline.label_audit import compute_label_audit
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
        "TR - Industry Enrichment.csv": "industry_enrichment",
    }
    for file_name, table_name in csv_files.items():
        file_path = DATA_DIR / "database_samples" / file_name
        load_csv_to_db(file_path, table_name, db_engine)
    
    build_star_schema(db_engine)
    logger.info("--- ETL Phase Complete ---")

    # --- 3. Label Audit (Phase 2) ---
    logger.info("--- Phase 2: Label audit (leakage-safe targets) ---")
    cutoff_date = "2024-09-30"
    prediction_window_months = 6
    compute_label_audit(db_engine, target_division, cutoff_date, prediction_window_months)
    logger.info("--- Label audit complete ---")

    # --- 4. Feature Library emission (catalog) ---
    # Build a feature matrix once to emit the feature catalog before training
    try:
        from gosales.features.engine import create_feature_matrix
        create_feature_matrix(db_engine, target_division, cutoff_date, prediction_window_months)
        logger.info("--- Feature catalog emitted ---")
    except Exception as e:
        logger.warning(f"Feature catalog emission failed (non-blocking): {e}")

    # --- 5. Model Training Phase ---
    logger.info(f"--- Phase 3: Training model for {target_division} division ---")
    train_division_model(db_engine, target_division, cutoff_date=cutoff_date, prediction_window_months=prediction_window_months)
    logger.info("--- Model Training Phase Complete ---")

    # --- 6. Scoring Phase ---
    logger.info("--- Phase 4: Generating Scores and Whitespace ---")
    generate_scoring_outputs(db_engine)
    logger.info("--- Scoring Phase Complete ---")

    logger.info("GoSales scoring pipeline finished successfully!")


if __name__ == "__main__":
    score_all()
