#!/usr/bin/env python3
"""
Customer scoring pipeline that generates ICP scores and whitespace analysis.

This module loads trained models and scores all customers, outputting:
- icp_scores.csv: Customer ICP scores for each product
- whitespace.csv: Whitespace opportunities for customers
"""

import polars as pl
import pandas as pd
import mlflow.sklearn
from pathlib import Path

from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import MODELS_DIR, OUTPUTS_DIR
from gosales.features.engine import create_feature_matrix

logger = get_logger(__name__)


def score_customers_for_product(engine, product_name: str, model_path: Path):
    """Score all customers for a specific product.
    
    Args:
        engine: Database engine
        product_name: Name of the product to score
        model_path: Path to the trained model
        
    Returns:
        polars.DataFrame: Customers with their scores
    """
    logger.info(f"Scoring customers for product: {product_name}")
    
    try:
        # Load the trained model
        model = mlflow.sklearn.load_model(str(model_path))
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return pl.DataFrame()
    
    # Get feature matrix for all customers
    feature_matrix = create_feature_matrix(engine, product_name)
    
    if feature_matrix.is_empty():
        logger.warning(f"No feature matrix for {product_name}")
        return pl.DataFrame()
    
    # Prepare features for scoring (same as training)
    X = feature_matrix.drop(["customer_id", "bought_product"]).to_pandas()
    
    # Score all customers
    try:
        # Get probability of buying the product
        probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1 (buying)
        
        # Create scoring results
        scores_df = feature_matrix.select(["customer_id", "bought_product"]).to_pandas()
        scores_df['product_name'] = product_name
        scores_df['icp_score'] = probabilities
        scores_df['confidence'] = 'medium'  # Could be enhanced based on probability ranges
        
        # Add customer names if available
        try:
            customer_names = pl.read_database(
                "select customer_id, customer_name from dim_customer", engine
            ).to_pandas()
            scores_df = scores_df.merge(customer_names, on='customer_id', how='left')
        except:
            scores_df['customer_name'] = 'Unknown'
        
        logger.info(f"Successfully scored {len(scores_df)} customers for {product_name}")
        return pl.from_pandas(scores_df)
        
    except Exception as e:
        logger.error(f"Failed to score customers for {product_name}: {e}")
        return pl.DataFrame()


def generate_whitespace_opportunities(engine):
    """Generate whitespace opportunities based on product purchase patterns.
    
    Args:
        engine: Database engine
        
    Returns:
        polars.DataFrame: Whitespace opportunities
    """
    logger.info("Generating whitespace opportunities...")
    
    try:
        # Get all customer-product combinations
        fact_orders = pl.read_database("select * from fact_orders", engine)
        
        # Get customers and their purchased products
        customer_products = (
            fact_orders
            .group_by("customer_id")
            .agg([
                pl.col("product_name").unique().alias("products_bought"),
                pl.sum("revenue").alias("total_revenue"),
                pl.len().alias("total_orders")
            ])
        )
        
        # Get all available products
        all_products = fact_orders.select("product_name").unique()["product_name"].to_list()
        
        whitespace_opportunities = []
        
        for row in customer_products.iter_rows(named=True):
            customer_id = row["customer_id"]
            products_bought = row["products_bought"]
            total_revenue = row["total_revenue"]
            
            # Find products not bought by this customer
            not_bought = [p for p in all_products if p not in products_bought]
            
            for product in not_bought:
                # Simple whitespace scoring based on customer value
                if total_revenue > 5000:
                    whitespace_score = 0.8
                    priority = "High"
                elif total_revenue > 1000:
                    whitespace_score = 0.6
                    priority = "Medium"
                else:
                    whitespace_score = 0.3
                    priority = "Low"
                
                whitespace_opportunities.append({
                    "customer_id": customer_id,
                    "product_name": product,
                    "whitespace_score": whitespace_score,
                    "priority": priority,
                    "reason": f"Customer spent ${total_revenue:.0f}, likely to buy {product}"
                })
        
        whitespace_df = pl.DataFrame(whitespace_opportunities)
        
        # Add customer names
        try:
            customer_names = pl.read_database(
                "select customer_id, customer_name from dim_customer", engine
            )
            whitespace_df = whitespace_df.join(customer_names, on="customer_id", how="left")
        except:
            whitespace_df = whitespace_df.with_columns(pl.lit("Unknown").alias("customer_name"))
        
        logger.info(f"Generated {len(whitespace_df)} whitespace opportunities")
        return whitespace_df
        
    except Exception as e:
        logger.error(f"Failed to generate whitespace opportunities: {e}")
        return pl.DataFrame()


def generate_scoring_outputs(engine):
    """Generate ICP scores and whitespace analysis outputs.
    
    Args:
        engine: Database engine
    """
    logger.info("Starting customer scoring pipeline...")
    
    # Ensure outputs directory exists
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # Initialize scoring results
    all_scores = []
    
    # Score for available products with trained models
    available_models = {
        "Supplies": MODELS_DIR / "supplies_model"
    }
    
    for product_name, model_path in available_models.items():
        if model_path.exists():
            scores = score_customers_for_product(engine, product_name, model_path)
            if not scores.is_empty():
                all_scores.append(scores)
        else:
            logger.warning(f"Model not found for {product_name}: {model_path}")
    
    # Combine all scores and save to CSV
    if all_scores:
        combined_scores = pl.concat(all_scores, how="vertical")
        
        # Save ICP scores
        icp_scores_path = OUTPUTS_DIR / "icp_scores.csv"
        combined_scores.write_csv(str(icp_scores_path))
        logger.info(f"Saved ICP scores to {icp_scores_path}")
        
        # Display summary
        logger.info(f"Scored {combined_scores.height} customer-product combinations")
        for product in combined_scores["product_name"].unique():
            product_scores = combined_scores.filter(pl.col("product_name") == product)
            avg_score = product_scores["icp_score"].mean()
            logger.info(f"  {product}: {len(product_scores)} customers, avg score: {avg_score:.3f}")
    else:
        logger.warning("No models available for scoring")
    
    # Generate whitespace opportunities
    whitespace = generate_whitespace_opportunities(engine)
    if not whitespace.is_empty():
        whitespace_path = OUTPUTS_DIR / "whitespace.csv"
        whitespace.write_csv(str(whitespace_path))
        logger.info(f"Saved whitespace opportunities to {whitespace_path}")
        
        # Display summary
        high_priority = whitespace.filter(pl.col("priority") == "High").height
        medium_priority = whitespace.filter(pl.col("priority") == "Medium").height
        low_priority = whitespace.filter(pl.col("priority") == "Low").height
        logger.info(f"Whitespace: {high_priority} high, {medium_priority} medium, {low_priority} low priority")
    
    logger.info("Customer scoring pipeline completed successfully!")


if __name__ == "__main__":
    # Get database connection
    db_engine = get_db_connection()
    
    # Generate scoring outputs
    generate_scoring_outputs(db_engine)