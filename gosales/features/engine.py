import polars as pl
import pandas as pd
from datetime import datetime
from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger

logger = get_logger(__name__)

def create_feature_matrix(engine, division_name: str, cutoff_date: str = None, prediction_window_months: int = 6):
    """
    Creates a rich feature matrix for a specific division for ML training with proper time-based splitting.

    This function reads from the clean `fact_transactions` and `dim_customer` tables
    and engineers a wide range of behavioral features, including recency, monetary value,
    customer growth, and ecosystem engagement.

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
        division_name (str): The name of the division to create the feature matrix for (e.g., 'Solidworks').
        cutoff_date (str, optional): Date string (YYYY-MM-DD) to use as feature cutoff. If None, uses all historical data.
        prediction_window_months (int): Number of months after cutoff_date to define the prediction target.

    Returns:
        polars.DataFrame: The feature matrix with a binary target column `bought_in_division`.
    """
    logger.info(f"Creating feature matrix for division: {division_name}...")
    if cutoff_date:
        logger.info(f"Using cutoff date: {cutoff_date} (features from data <= cutoff)")
        logger.info(f"Target: purchases in {prediction_window_months} months after cutoff")

    # --- 1. Load Base Data ---
    try:
        transactions_pd = pd.read_sql("SELECT * FROM fact_transactions", engine)
        # Ensure order_date is properly converted to datetime
        transactions_pd['order_date'] = pd.to_datetime(transactions_pd['order_date'])
        
        # Filter data for time-based split if cutoff_date is provided
        if cutoff_date:
            cutoff_dt = pd.to_datetime(cutoff_date)
            # Split data into feature period (<=cutoff) and prediction period (after cutoff)
            feature_data = transactions_pd[transactions_pd['order_date'] <= cutoff_dt].copy()
            prediction_data = transactions_pd[transactions_pd['order_date'] > cutoff_dt].copy()
            
            # Calculate prediction window end date
            from dateutil.relativedelta import relativedelta
            prediction_end = cutoff_dt + relativedelta(months=prediction_window_months)
            prediction_data = prediction_data[prediction_data['order_date'] <= prediction_end]
            
            logger.info(f"Feature data: {len(feature_data)} transactions <= {cutoff_date}")
            logger.info(f"Prediction data: {len(prediction_data)} transactions in {prediction_window_months}-month window")
        else:
            # Use all data for features and target (original behavior)
            feature_data = transactions_pd.copy()
            prediction_data = transactions_pd.copy()
        
        transactions = pl.from_pandas(feature_data)
        customers = pl.from_pandas(pd.read_sql("SELECT customer_id FROM dim_customer", engine))
    except Exception as e:
        logger.error(f"Failed to read necessary tables from the database: {e}")
        return pl.DataFrame()

    if transactions.is_empty() or customers.is_empty():
        logger.warning("Transactions or customers data is empty. Cannot build feature matrix.")
        return pl.DataFrame()

    # --- 2. Create the Binary Target Variable ---
    # Target: 1 if the customer bought any product in the target division in the prediction window, 0 otherwise.
    if cutoff_date:
        # Use prediction window data for target labels
        prediction_buyers_df = prediction_data[prediction_data['product_division'] == division_name]['customer_id'].unique()
        division_buyers_pd = pd.DataFrame({'customer_id': prediction_buyers_df, 'bought_in_division': 1})
        division_buyers = pl.from_pandas(division_buyers_pd).lazy()
        logger.info(f"Target: {len(prediction_buyers_df)} customers bought {division_name} in prediction window")
    else:
        # Original behavior: ever bought in historical data
        division_buyers = (
            transactions.lazy()
            .filter(pl.col("product_division") == division_name)
            .select("customer_id")
            .unique()
            .with_columns(pl.lit(1).cast(pl.Int8).alias("bought_in_division"))
        )

    # --- 3. Engineer Behavioral Features ---
    # Get the current date for recency calculations
    current_date = datetime.now().date()

    features = (
        transactions.lazy()
        .group_by("customer_id")
        .agg([
            # Recency Features
            pl.col("order_date").max().alias("last_order_date"),
            pl.col("order_date").filter(pl.col("product_division") == division_name).max().alias(f"last_{division_name}_order_date"),
            
            # Frequency Features
            pl.len().alias("total_transactions_all_time"),
            pl.col("order_date").filter(pl.col("order_date").dt.year().is_in([2023, 2024])).len().alias("transactions_last_2y"),

            # Monetary Features
            pl.sum("gross_profit").alias("total_gp_all_time"),
            pl.col("gross_profit").filter(pl.col("order_date").dt.year().is_in([2023, 2024])).sum().alias("total_gp_last_2y"),
            pl.mean("gross_profit").alias("avg_transaction_gp"),

            # Cross-division behavioral patterns (non-leaky features)
            pl.col("product_division").filter(pl.col("product_division") == "Services").len().alias("services_transaction_count"),
            pl.col("product_division").filter(pl.col("product_division") == "Simulation").len().alias("simulation_transaction_count"),
            pl.col("product_division").filter(pl.col("product_division") == "Hardware").len().alias("hardware_transaction_count"),

            # Services engagement (proxy for technical sophistication)
            pl.col("gross_profit").filter(pl.col("product_division") == "Services").sum().alias("total_services_gp"),
            pl.col("gross_profit").filter(pl.col("product_sku") == "Training").sum().alias("total_training_gp"),

            # Growth trajectory features
            pl.col("gross_profit").filter(pl.col("order_date").dt.year() == 2024).sum().alias("gp_2024"),
            pl.col("gross_profit").filter(pl.col("order_date").dt.year() == 2023).sum().alias("gp_2023"),
            
            # General engagement features
            pl.n_unique("product_division").alias("product_diversity_score"),
        ])
        .collect()
    )
    
    # Calculate recency features in pandas for easier date arithmetic
    features_pd = features.to_pandas()
    
    # Handle date columns properly
    if 'last_order_date' in features_pd.columns:
        # Convert string dates to datetime and calculate days difference
        last_order_dates = pd.to_datetime(features_pd['last_order_date'], errors='coerce')
        # Calculate days difference safely
        days_diff = []
        for date in last_order_dates:
            if pd.isna(date):
                days_diff.append(999)
            else:
                days_diff.append((current_date - date.date()).days)
        features_pd['days_since_last_order'] = days_diff
    else:
        features_pd['days_since_last_order'] = 999  # Default for customers with no orders
        
    division_date_col = f'last_{division_name}_order_date'
    if division_date_col in features_pd.columns:
        # Convert string dates to datetime and calculate days difference
        last_division_dates = pd.to_datetime(features_pd[division_date_col], errors='coerce')
        # Calculate days difference safely
        days_diff = []
        for date in last_division_dates:
            if pd.isna(date):
                days_diff.append(999)
            else:
                days_diff.append((current_date - date.date()).days)
        features_pd[f'days_since_last_{division_name}_order'] = days_diff
    else:
        features_pd[f'days_since_last_{division_name}_order'] = 999  # Default for customers with no orders in division
    
    # Convert back to polars
    features = pl.from_pandas(features_pd)

    # --- 4. Combine Features and Target ---
    # Start with all customers, then left-join the features and the target variable.
    feature_matrix = (
        customers.lazy()
        .join(features.lazy(), on="customer_id", how="left")
        .join(division_buyers, on="customer_id", how="left")
        .with_columns([
            pl.col("bought_in_division").fill_null(0).cast(pl.Int8), # Customers who never bought get 0
        ])
        .collect()
    )
    
    # Fill nulls for all other columns in pandas for easier handling
    feature_matrix_pd = feature_matrix.to_pandas()
    
    # Drop the date columns that are no longer needed for ML
    date_columns = [col for col in feature_matrix_pd.columns if 'date' in col.lower()]
    feature_matrix_pd = feature_matrix_pd.drop(columns=date_columns)
    
    # Fill nulls and ensure proper data types
    feature_matrix_pd = feature_matrix_pd.fillna(0)
    
    # Convert back to polars
    feature_matrix = pl.from_pandas(feature_matrix_pd)

    # Join with customer industry data
    try:
        logger.info("Joining industry data with feature matrix...")
        customers_with_industry_pd = pd.read_sql("""
            SELECT customer_id, industry, industry_sub 
            FROM dim_customer 
            WHERE industry IS NOT NULL
        """, engine)
        
        if not customers_with_industry_pd.empty:
            # Normalise text
            customers_with_industry_pd['industry'] = customers_with_industry_pd['industry'].astype(str).str.strip()
            customers_with_industry_pd['industry_sub'] = customers_with_industry_pd['industry_sub'].astype(str).str.strip()

            # Top-N categories
            top_industries = customers_with_industry_pd['industry'].value_counts().head(20).index.tolist()
            top_subs = customers_with_industry_pd['industry_sub'].value_counts().head(30).index.tolist()

            # Helper to sanitize feature names for LightGBM (alnum + underscore only)
            import re
            def sanitize_key(text: str) -> str:
                if text is None:
                    return "unknown"
                key = str(text).lower()
                key = key.replace("&", " and ")
                key = re.sub(r"[^0-9a-zA-Z]+", "_", key)
                key = re.sub(r"_+", "_", key).strip("_")
                if not key:
                    key = "unknown"
                return key

            # Industry dummies
            industry_key_map = {industry: sanitize_key(industry) for industry in top_industries}
            for industry, key in industry_key_map.items():
                customers_with_industry_pd[f"is_{key}"] = (customers_with_industry_pd['industry'] == industry).astype(int)

            # Sub-industry dummies
            sub_key_map = {sub: sanitize_key(sub) for sub in top_subs}
            for sub, key in sub_key_map.items():
                customers_with_industry_pd[f"is_sub_{key}"] = (customers_with_industry_pd['industry_sub'] == sub).astype(int)

            # Interaction examples: industry × services engagement will be created post-join

            # Convert to polars and join
            industry_features = pl.from_pandas(customers_with_industry_pd)
            feature_columns = ["customer_id"] + \
                [f"is_{industry_key_map[i]}" for i in top_industries] + \
                [f"is_sub_{sub_key_map[s]}" for s in top_subs]

            feature_matrix = feature_matrix.join(
                industry_features.select(feature_columns),
                on="customer_id",
                how="left"
            ).fill_null(0)
            
            logger.info(f"Successfully joined industry data. Added {len(top_industries)} industry and {len(top_subs)} sub-industry dummies.")
        else:
            logger.warning("No industry data available for joining.")
            
    except Exception as e:
        logger.warning(f"Could not join industry data: {e}")

    # Example interaction features (post-join) if present
    try:
        interaction_cols = [c for c in feature_matrix.columns if c.startswith("is_") or c.startswith("is_sub_")]
        if "total_services_gp" in feature_matrix.columns and interaction_cols:
            # Multiply normalized services GP by industry flags (simple scaled interaction)
            max_services = float(feature_matrix["total_services_gp"].max()) or 1.0
            svc_norm = feature_matrix["total_services_gp"].cast(pl.Float64) / max_services
            for c in interaction_cols[:10]:  # limit to top few to avoid explosion
                feature_matrix = feature_matrix.with_columns((svc_norm * feature_matrix[c].cast(pl.Float64)).alias(f"{c}_x_services"))
    except Exception:
        pass

    logger.info(f"Successfully created feature matrix for division: {division_name}.")
    logger.info(f"Total customers processed: {feature_matrix.height}")
    positive_cases = feature_matrix.filter(pl.col('bought_in_division') == 1).height
    logger.info(f"Customers who bought in {division_name}: {positive_cases}")
    
    return feature_matrix


if __name__ == "__main__":
    db_engine = get_db_connection()
    # Example: Build the feature matrix for the 'Solidworks' division
    feature_matrix = create_feature_matrix(db_engine, "Solidworks")
    if not feature_matrix.is_empty():
        print("Feature Matrix Head:")
        print(feature_matrix.head().to_pandas().to_string())
        print("\nFeature Matrix Shape:")
        print(feature_matrix.shape)
