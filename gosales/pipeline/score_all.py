from gosales.utils.db import get_db_connection
from gosales.etl.load_csv import load_csv_to_db
from gosales.etl.build_star import build_star_schema
from gosales.models.train_division_model import train_division_model
from gosales.pipeline.label_audit import compute_label_audit
from gosales.pipeline.score_customers import generate_scoring_outputs
from gosales.utils.logger import get_logger
from gosales.utils.paths import DATA_DIR, OUTPUTS_DIR
from gosales.etl.sku_map import division_set
from gosales.utils.run_context import default_manifest, emit_manifest
from gosales.pipeline.validate_holdout import validate_holdout

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
    try:
        divisions = list(division_set())
    except Exception:
        divisions = ["Solidworks"]

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
    # Training cutoff chosen so the 6-month target window is within training data (Jul–Dec 2024)
    cutoff_date = "2024-06-30"
    prediction_window_months = 6
    for div in divisions:
        try:
            compute_label_audit(db_engine, div, cutoff_date, prediction_window_months)
        except Exception as e:
            logger.warning(f"Label audit failed for {div}: {e}")
    logger.info("--- Label audit complete ---")

    # --- 4. Feature Library emission (catalog) ---
    # Build a feature matrix per division to emit the feature catalog before training
    try:
        from gosales.features.engine import create_feature_matrix
        for div in divisions:
            try:
                create_feature_matrix(db_engine, div, cutoff_date, prediction_window_months)
            except Exception as e:
                logger.warning(f"Feature catalog emission failed for {div} (non-blocking): {e}")
        logger.info("--- Feature catalogs emitted ---")
    except Exception as e:
        logger.warning(f"Feature catalog emission failed (non-blocking): {e}")

    # --- 5. Model Training Phase ---
    logger.info("--- Phase 3: Training models for all divisions ---")
    for div in divisions:
        try:
            logger.info(f"Training model for division: {div}")
            train_division_model(db_engine, div, cutoff_date=cutoff_date, prediction_window_months=prediction_window_months)
        except Exception as e:
            logger.warning(f"Training failed for {div}: {e}")
    logger.info("--- Model Training Phase Complete ---")

    # --- 6. Scoring Phase ---
    logger.info("--- Phase 4: Generating Scores and Whitespace ---")
    # Create run manifest and record high-level context
    run_manifest = default_manifest(pipeline_version="0.1.0")
    run_manifest["cutoff"] = cutoff_date
    run_manifest["window_months"] = int(prediction_window_months)

    # Generate outputs; function will update manifest details (divisions scored, alerts)
    generate_scoring_outputs(db_engine, run_manifest=run_manifest)

    # Persist manifest alongside outputs
    try:
        manifest_path = emit_manifest(OUTPUTS_DIR, run_manifest["run_id"], run_manifest)
        logger.info(f"Wrote run manifest to {manifest_path}")
    except Exception as e:
        logger.warning(f"Failed to write run manifest: {e}")
    logger.info("--- Scoring Phase Complete ---")

    # --- 7. Hold-out validation & gates (Phase 5) ---
    try:
        from pathlib import Path as _Path
        icp_path = OUTPUTS_DIR / "icp_scores.csv"
        if icp_path.exists():
            # Derive a year tag from cutoff (simple heuristic: cutoff year + 1)
            year_tag = None
            try:
                y = int(str(run_manifest.get("cutoff", "")).split("-")[0]) if isinstance(run_manifest, dict) else None
                if y:
                    year_tag = str(y + 1)
            except Exception:
                year_tag = None
            validate_holdout(icp_scores_csv=str(icp_path), year_tag=year_tag)
    except Exception as e:
        logger.warning(f"Hold-out validation step failed (non-blocking): {e}")

    logger.info("GoSales scoring pipeline finished successfully!")


if __name__ == "__main__":
    score_all()
