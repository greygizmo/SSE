#!/usr/bin/env python3
"""
Customer scoring pipeline that generates ICP scores and whitespace analysis for specific divisions.
"""
import polars as pl
import pandas as pd
import mlflow.sklearn
import json
from pathlib import Path

from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import MODELS_DIR, OUTPUTS_DIR
from gosales.features.engine import create_feature_matrix

logger = get_logger(__name__)

def score_customers_for_division(engine, division_name: str, model_path: Path):
    """
    Score all customers for a specific division using a trained ML model.
    """
    logger.info(f"Scoring customers for division: {division_name}")
    
    try:
        model = mlflow.sklearn.load_model(str(model_path))
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return pl.DataFrame()
    
    # Get feature matrix for all customers for the specified division
    cutoff = None
    window_months = None
    try:
        with open(model_path / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            cutoff = meta.get("cutoff_date")
            window_months = int(meta.get("prediction_window_months")) if meta.get("prediction_window_months") is not None else None
    except Exception:
        pass

    if cutoff and window_months:
        feature_matrix = create_feature_matrix(engine, division_name, cutoff, window_months)
    else:
        feature_matrix = create_feature_matrix(engine, division_name)
    
    if feature_matrix.is_empty():
        logger.warning(f"No feature matrix for {division_name}")
        return pl.DataFrame()
    
    # Prepare features for scoring (must match training)
    X = feature_matrix.drop(["customer_id", "bought_in_division"]).to_pandas()
    # Align columns to training feature order using saved metadata if present
    try:
        with open(model_path / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        train_cols = meta.get("feature_names", [])
        if train_cols:
            # Add any missing columns with zeros, drop extras, and reorder
            for col in train_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[train_cols]
    except Exception:
        # If metadata missing, proceed with current X
        pass
    
    try:
        # Get the probability of buying from the division
        probabilities = model.predict_proba(X)[:, 1]
        
        scores_df = feature_matrix.select(["customer_id", "bought_in_division"]).to_pandas()
        scores_df['division_name'] = division_name
        scores_df['icp_score'] = probabilities
        
        customer_names = pd.read_sql("select customer_id, customer_name from dim_customer", engine)
        scores_df = scores_df.merge(customer_names, on='customer_id', how='left')
        
        logger.info(f"Successfully scored {len(scores_df)} customers for {division_name}")
        return pl.from_pandas(scores_df)
        
    except Exception as e:
        logger.error(f"Failed to score customers for {division_name}: {e}")
        return pl.DataFrame()

def generate_whitespace_opportunities(engine):
    """
    Generate whitespace opportunities based on division purchase patterns.
    This is a simplified version and can be enhanced.
    """
    logger.info("Generating whitespace opportunities...")
    try:
        transactions = pl.from_pandas(pd.read_sql("SELECT * FROM fact_transactions", engine))
        customers = pl.from_pandas(pd.read_sql("SELECT * FROM dim_customer", engine))
        # Align join key dtypes
        if "customer_id" in transactions.columns:
            transactions = transactions.with_columns(pl.col("customer_id").cast(pl.Int64, strict=False))
        if "customer_id" in customers.columns:
            customers = customers.with_columns(pl.col("customer_id").cast(pl.Int64, strict=False))
        
        customer_summary = (
            transactions
            .group_by("customer_id")
            .agg([
                pl.col("product_division").unique().alias("divisions_bought"),
                pl.sum("gross_profit").alias("total_gp"),
            ])
        )
        
        all_divisions = transactions.select("product_division").unique()["product_division"].to_list()
        
        opportunities = []
        for row in customer_summary.iter_rows(named=True):
            not_bought = [div for div in all_divisions if div not in row["divisions_bought"]]
            for division in not_bought:
                score = 0.5 # Placeholder logic
                if row["total_gp"] > 10000: score = 0.8
                elif row["total_gp"] > 1000: score = 0.6
                
                opportunities.append({
                    "customer_id": row["customer_id"],
                    "whitespace_division": division,
                    "whitespace_score": score,
                    "reason": f"Customer has high engagement but has not bought from the {division} division."
                })
        
        if not opportunities:
            return pl.DataFrame()

        whitespace_df = pl.DataFrame(opportunities).with_columns(pl.col("customer_id").cast(pl.Int64, strict=False))\
            .join(customers, on="customer_id", how="left")
        logger.info(f"Generated {len(whitespace_df)} whitespace opportunities")
        return whitespace_df

    except Exception as e:
        logger.error(f"Failed to generate whitespace opportunities: {e}")
        return pl.DataFrame()

def generate_scoring_outputs(engine):
    """
    Generate and save ICP scores and whitespace analysis.
    """
    logger.info("Starting customer scoring and whitespace analysis...")
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    available_models = {
        "Solidworks": MODELS_DIR / "solidworks_model"
    }
    
    all_scores = []
    for division_name, model_path in available_models.items():
        if model_path.exists():
            scores = score_customers_for_division(engine, division_name, model_path)
            if not scores.is_empty():
                all_scores.append(scores)
        else:
            logger.warning(f"Model not found for {division_name}: {model_path}")
            
    if all_scores:
        combined_scores = pl.concat(all_scores, how="vertical")
        icp_scores_path = OUTPUTS_DIR / "icp_scores.csv"
        combined_scores.write_csv(str(icp_scores_path))
        logger.info(f"Saved ICP scores for {len(combined_scores)} customer-division combinations to {icp_scores_path}")
    else:
        logger.warning("No models were available for scoring.")
    
    whitespace = generate_whitespace_opportunities(engine)
    if not whitespace.is_empty():
        whitespace_path = OUTPUTS_DIR / "whitespace.csv"
        whitespace.write_csv(str(whitespace_path))
        logger.info(f"Saved whitespace opportunities to {whitespace_path}")

    logger.info("Scoring pipeline completed successfully!")

if __name__ == "__main__":
    db_engine = get_db_connection()
    generate_scoring_outputs(db_engine)
