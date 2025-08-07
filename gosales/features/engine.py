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

            # Growth & Scale Features
            pl.col("quantity").filter(pl.col("product_sku") == "SWX_Core").sum().alias("total_core_seats"),
            pl.col("quantity").filter(pl.col("product_sku") == "SWX_Pro_Prem").sum().alias("total_pro_prem_seats"),
            
            # Ecosystem Engagement Features
            pl.col("product_sku").filter(pl.col("product_sku").is_in(["Core_New_UAP", "Pro_Prem_New_UAP"])).is_not_null().any().alias("has_uap_support"),
            pl.col("product_sku").filter(pl.col("product_sku") == "Success Plan GP").is_not_null().any().alias("has_success_plan"),
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
    
    # Convert boolean columns properly
    boolean_columns = ['has_uap_support', 'has_success_plan']
    for col in boolean_columns:
        if col in feature_matrix_pd.columns:
            feature_matrix_pd[col] = feature_matrix_pd[col].astype(int)
    
    feature_matrix = pl.from_pandas(feature_matrix_pd)

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
