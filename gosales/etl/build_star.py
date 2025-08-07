import polars as pl
import pandas as pd
from sqlalchemy import text
from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger

logger = get_logger(__name__)

def build_star_schema(engine):
    """
    Builds a tidy, analytics-ready star schema from the raw sales_log data.
    
    This function performs the following transformations:
    1.  Reads the raw `sales_log` table.
    2.  Creates a clean `dim_customer` dimension table.
    3.  Defines a comprehensive mapping of raw columns to standardized SKUs and Divisions.
    4.  "Unpivots" the wide `sales_log` table into a tidy `fact_transactions` table,
        where each row represents a single product line item within a transaction.
    5.  Cleans and standardizes data types for key columns.
    
    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
    """
    logger.info("Building star schema with new tidy transaction model...")

    # Read the raw data from the database using pandas first, then convert to polars
    logger.info("Reading sales_log table...")
    try:
        sales_log_pd = pd.read_sql("SELECT * FROM sales_log", engine)
        sales_log = pl.from_pandas(sales_log_pd)
    except Exception as e:
        logger.error(f"Failed to read sales_log table: {e}")
        return

    # --- 1. Create dim_customer ---
    logger.info("Creating dim_customer table...")
    dim_customer = (
        sales_log.lazy()
        .select(["Customer", "CustomerId"])
        .unique(subset=["CustomerId"])
        .rename({"Customer": "customer_name", "CustomerId": "customer_id"})
        .filter(pl.col("customer_id").is_not_null())
        .collect()
    )
    dim_customer.write_database("dim_customer", engine, if_table_exists="replace")
    logger.info(f"Successfully created dim_customer table with {len(dim_customer)} unique customers.")

    # --- 2. Define SKU and Division Mapping ---
    # This mapping is the core logic for the unpivot operation.
    # It links GP columns to their corresponding Qty columns and assigns a Division.
    sku_mapping = {
        'SWX_Core': {'qty_col': 'SWX_Core_Qty', 'division': 'Solidworks'},
        'SWX_Pro_Prem': {'qty_col': 'SWX_Pro_Prem_Qty', 'division': 'Solidworks'},
        'Core_New_UAP': {'qty_col': 'Core_New_UAP_Qty', 'division': 'Solidworks'},
        'Pro_Prem_New_UAP': {'qty_col': 'Pro_Prem_New_UAP_Qty', 'division': 'Solidworks'},
        'PDM': {'qty_col': 'PDM_Qty', 'division': 'Solidworks'},
        'Simulation': {'qty_col': 'Simulation_Qty', 'division': 'Simulation'},
        'Services': {'qty_col': 'Services_Qty', 'division': 'Services'},
        'Training': {'qty_col': 'Training_Qty', 'division': 'Services'},
        'Success Plan GP': {'qty_col': 'Success_Plan_Qty', 'division': 'Services'},
        'Supplies': {'qty_col': 'Consumables_Qty', 'division': 'Hardware'}, # Assuming supplies = consumables
    }

    # --- 3. Unpivot the data to create fact_transactions ---
    logger.info("Unpivoting sales_log to create fact_transactions table...")
    
    all_transactions = []
    
    # Base columns to keep for every transaction line item
    id_vars = ["CustomerId", "Rec Date", "Division"]

    for gp_col, details in sku_mapping.items():
        qty_col = details['qty_col']
        division = details['division']

        if gp_col in sales_log.columns and qty_col in sales_log.columns:
            # Melt for the current SKU
            melted_df = (
                sales_log.lazy()
                .select(id_vars + [gp_col, qty_col])
                .filter(pl.col(gp_col).is_not_null() | pl.col(qty_col).is_not_null())
                .with_columns([
                    pl.lit(gp_col).alias("product_sku"),
                    pl.lit(division).alias("product_division")
                ])
                .rename({gp_col: "gross_profit", qty_col: "quantity"})
                .collect()
            )
            all_transactions.append(melted_df)

    if not all_transactions:
        logger.error("No transactions could be processed from the sku_mapping. Aborting.")
        return

    # Combine all the melted DataFrames into one large table
    fact_transactions = pl.concat(all_transactions, how="vertical_relaxed")
    
    # --- 4. Clean and Finalize the Table ---
    logger.info("Cleaning and finalizing fact_transactions table...")
    
    # Convert to pandas for easier data cleaning
    fact_transactions_pd = fact_transactions.to_pandas()
    
    # Clean the data
    fact_transactions_pd['customer_id'] = pd.to_numeric(fact_transactions_pd['CustomerId'], errors='coerce')
    fact_transactions_pd['order_date'] = pd.to_datetime(fact_transactions_pd['Rec Date'])
    
    # Clean currency columns
    def clean_currency(value):
        if pd.isna(value):
            return 0.0
        if isinstance(value, str):
            # Handle negative values in parentheses
            if '(' in value and ')' in value:
                value = '-' + value.replace('(', '').replace(')', '')
            return float(value.replace('$', '').replace(',', ''))
        return float(value)
    
    fact_transactions_pd['gross_profit'] = fact_transactions_pd['gross_profit'].apply(clean_currency)
    fact_transactions_pd['quantity'] = pd.to_numeric(fact_transactions_pd['quantity'], errors='coerce').fillna(0)
    
    # Filter out rows with no meaningful transaction data
    fact_transactions_pd = fact_transactions_pd[
        (fact_transactions_pd['gross_profit'] != 0) | (fact_transactions_pd['quantity'] != 0)
    ]
    
    # Select final columns
    fact_transactions_pd = fact_transactions_pd[[
        'customer_id', 'order_date', 'product_sku', 'product_division', 'gross_profit', 'quantity'
    ]]
    
    # Convert back to polars
    fact_transactions = pl.from_pandas(fact_transactions_pd)

    fact_transactions.write_database("fact_transactions", engine, if_table_exists="replace")
    logger.info(f"Successfully created fact_transactions table with {len(fact_transactions)} total line items.")
    
    # --- Deprecate old tables ---
    logger.info("Dropping old fact_orders and dim_product tables...")
    with engine.connect() as connection:
        connection.execute(text("DROP TABLE IF EXISTS fact_orders;"))
        connection.execute(text("DROP TABLE IF EXISTS dim_product;"))
        connection.commit()
    logger.info("Old tables dropped successfully.")


if __name__ == "__main__":
    db_engine = get_db_connection()
    build_star_schema(db_engine)
