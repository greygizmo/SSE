import polars as pl
import pandas as pd
import json
try:
    from rapidfuzz import process, fuzz
    FUZZY_AVAILABLE = True
except Exception:
    FUZZY_AVAILABLE = False
from sqlalchemy import text
from gosales.utils.db import get_db_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger
from gosales.etl.sku_map import get_sku_mapping
from gosales.etl.cleaners import clean_currency_value, coerce_datetime, summarise_dataframe_schema
from gosales.etl.contracts import (
    check_required_columns,
    check_primary_key_not_null,
    check_no_duplicate_pk,
    violations_to_dataframe,
)

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
        # Data contracts: required columns and PK/null checks
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        contracts_dir = OUTPUTS_DIR / "contracts"
        contracts_dir.mkdir(parents=True, exist_ok=True)

        # Persist schema snapshot
        schema_json = summarise_dataframe_schema(sales_log_pd)
        with open(contracts_dir / "schema.json", "w", encoding="utf-8") as f:
            json.dump(schema_json, f, indent=2)

        required_cols = list({"CustomerId", "Rec Date", "Division"})
        # Extend with GP/Qty pairs from mapping for visibility (not all required to exist)
        mapping = get_sku_mapping()
        for gp_col, meta in mapping.items():
            required_cols.extend([gp_col, meta["qty_col"]])

        violations = []
        violations += check_required_columns(sales_log_pd, "sales_log", required_cols)

        # PK checks (blockers)
        pk_cols = tuple(c for c in ("CustomerId", "Rec Date") if c in sales_log_pd.columns)
        violations += check_primary_key_not_null(sales_log_pd, "sales_log", pk_cols)
        # Duplicate check only if both cols available (best-effort)
        if len(pk_cols) >= 2:
            violations += check_no_duplicate_pk(sales_log_pd, "sales_log", pk_cols)

        vdf = violations_to_dataframe(violations)
        vdf.to_csv(contracts_dir / "violations.csv", index=False)

        # Block on PK/null/duplicate violations
        if any(v.violation_type in {"null_in_pk", "missing_pk_column", "duplicate_pk"} for v in violations):
            logger.error("Data contract violations detected. See outputs/contracts/violations.csv")
            return

        sales_log = pl.from_pandas(sales_log_pd)
    except Exception as e:
        logger.error(f"Failed to read sales_log table: {e}")
        return

    # --- 1. Create dim_customer with industry enrichment ---
    logger.info("Creating dim_customer table with industry enrichment...")

    # Read industry enrichment data if available
    try:
        industry_enrichment_pd = pd.read_sql("SELECT * FROM industry_enrichment", engine)
        industry_enrichment = pl.from_pandas(industry_enrichment_pd)
        logger.info(f"Successfully loaded industry enrichment data with {len(industry_enrichment)} records.")
    except Exception as e:
        logger.warning(f"Industry enrichment table not found: {e}")
        industry_enrichment = None

    # Start with basic customer data from sales_log
    # Keep both the legacy CustomerId and the ERP internal Id (ns_id) for robust joining
    # Build per-customer record; keep exact and normalised customer name for joining with enrichment
    dim_customer_base = (
        sales_log.lazy()
        .select(["Customer", "CustomerId"]) 
        .rename({
            "Customer": "customer_name",
            "CustomerId": "customer_id",
        })
        .group_by("customer_id")
        .agg([
            pl.col("customer_name").first().cast(pl.Utf8).str.strip_chars().alias("customer_name"),
        ])
        .with_columns([
            # Keep primary customer_id as integer to match fact tables and downstream joins
            pl.col("customer_id").cast(pl.Int64, strict=False),
            # Normalised name and numeric prefix for robust matching
            pl.col("customer_name").cast(pl.Utf8).str.to_lowercase().str.replace_all(r"\s+", " ").alias("customer_name_norm"),
            pl.col("customer_name")
                .cast(pl.Utf8)
                .str.extract(r"^\s*(\d+)")
                .cast(pl.Int64, strict=False)
                .alias("customer_prefix_id"),
        ])
        .filter(pl.col("customer_id").is_not_null())
    )

    if industry_enrichment is not None:
        # Join with industry enrichment data
        # Note: Mapping CustomerId from sales_log to ID from industry enrichment
        industry_clean = (
            industry_enrichment.lazy()
            .select([
                "Customer",
                "Cleaned Customer Name",
                "Web Address",
                "Industry",
                "Industry Sub List",
                "Reasoning",
                "ID",
            ])
            .rename({
                "Customer": "customer_name",
                "Cleaned Customer Name": "cleaned_customer_name",
                "Web Address": "web_address",
                "Industry": "industry",
                "Industry Sub List": "industry_sub",
                "Reasoning": "industry_reasoning",
                "ID": "industry_id_raw",
            })
            .with_columns([
                # Normalise customer name for exact match join
                pl.col("customer_name").cast(pl.Utf8).str.strip_chars().alias("customer_name"),
                pl.col("customer_name").cast(pl.Utf8).str.to_lowercase().str.replace_all(r"\s+", " ").alias("customer_name_norm"),
                pl.col("industry_id_raw").cast(pl.Utf8).str.strip_chars().cast(pl.Int64, strict=False).alias("industry_id_int"),
            ])
            .filter(pl.col("customer_name").is_not_null())
        )
        
        # Left join to preserve all customers, even those without industry data
        # A) Join on normalised customer name
        dc_name = (
            dim_customer_base
            .join(industry_clean.select(["customer_name_norm", "industry", "industry_sub", "web_address", "cleaned_customer_name", "industry_reasoning"]).unique(),
                  on="customer_name_norm", how="left")
        )

        # B) Fallback join on numeric prefix id -> enrichment ID
        dc_prefix = (
            dim_customer_base
            .join(industry_clean.select(["industry_id_int", "industry", "industry_sub", "web_address", "cleaned_customer_name", "industry_reasoning"]).unique(),
                  left_on="customer_prefix_id", right_on="industry_id_int", how="left")
            .rename({
                "industry": "industry_fallback",
                "industry_sub": "industry_sub_fallback",
                "web_address": "web_address_fallback",
                "cleaned_customer_name": "cleaned_customer_name_fallback",
                "industry_reasoning": "industry_reasoning_fallback",
            })
        )

        dim_customer = (
            dc_name
            .join(dc_prefix.select(["customer_id", "industry_fallback", "industry_sub_fallback", "web_address_fallback", "cleaned_customer_name_fallback", "industry_reasoning_fallback"]), on="customer_id", how="left")
            .with_columns([
                pl.coalesce([pl.col("industry"), pl.col("industry_fallback")]).alias("industry"),
                pl.coalesce([pl.col("industry_sub"), pl.col("industry_sub_fallback")]).alias("industry_sub"),
                pl.coalesce([pl.col("web_address"), pl.col("web_address_fallback")]).alias("web_address"),
                pl.coalesce([pl.col("cleaned_customer_name"), pl.col("cleaned_customer_name_fallback")]).alias("cleaned_customer_name"),
                pl.coalesce([pl.col("industry_reasoning"), pl.col("industry_reasoning_fallback")]).alias("industry_reasoning"),
            ])
            .drop(["industry_fallback", "industry_sub_fallback", "web_address_fallback", "cleaned_customer_name_fallback", "industry_reasoning_fallback"])
            .collect()
        )

        # C) Optional fuzzy-match fallback for remaining unmatched names
        try:
            if FUZZY_AVAILABLE:
                unmatched = dim_customer.filter(pl.col("industry").is_null())
                if len(unmatched) > 0:
                    logger.info(f"Attempting fuzzy match for {len(unmatched)} customers without industry data...")
                    # Collect eager frames to pandas
                    unmatched_pd = unmatched.select(["customer_id", "customer_name_norm"]).to_pandas()
                    choices_pd = industry_clean.select(["customer_name_norm"]).unique().collect().to_pandas()
                    choices = choices_pd["customer_name_norm"].dropna().tolist()

                    # Two-pass threshold: strict then slightly relaxed
                    def run_fuzzy(names: pd.Series, threshold: int) -> pd.DataFrame:
                        local_matches = []
                        for name in names.fillna(""):
                            if not name:
                                local_matches.append((name, None, 0))
                                continue
                            match = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
                            if match and match[1] >= threshold:
                                local_matches.append((name, match[0], match[1]))
                            else:
                                local_matches.append((name, None, 0))
                        return pd.DataFrame(local_matches, columns=["customer_name_norm", "matched_name_norm", "score"])\
                            .dropna(subset=["matched_name_norm"]) 

                    match_df = run_fuzzy(unmatched_pd["customer_name_norm"], 97)
                    # For names still unmatched, try threshold 94
                    if len(match_df) < len(unmatched_pd):
                        remaining = unmatched_pd.merge(match_df[["customer_name_norm"]], on="customer_name_norm", how="left", indicator=True)
                        remaining = remaining[remaining['_merge'] == 'left_only']
                        if not remaining.empty:
                            match_df_relaxed = run_fuzzy(remaining["customer_name_norm"], 94)
                            match_df = pd.concat([match_df, match_df_relaxed], ignore_index=True)

                    if not match_df.empty:
                        # Join back to enrichment to fetch attributes
                        enrich_pd = industry_clean.select(["customer_name_norm", "industry", "industry_sub", "web_address", "cleaned_customer_name", "industry_reasoning"]).unique().collect().to_pandas()
                        fuzz_join = match_df.merge(enrich_pd, left_on="matched_name_norm", right_on="customer_name_norm", how="left")
                        fuzz_join = fuzz_join[["customer_name_norm_x", "matched_name_norm", "score", "industry", "industry_sub", "web_address", "cleaned_customer_name", "industry_reasoning"]].rename(columns={"customer_name_norm_x": "customer_name_norm"})

                        # Persist fuzzy pairs for audit
                        try:
                            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                            fuzz_join.to_csv(OUTPUTS_DIR / 'industry_fuzzy_matches.csv', index=False)
                        except Exception:
                            pass

                        # Merge into dim_customer
                        dim_pd = dim_customer.to_pandas()
                        dim_pd = dim_pd.merge(fuzz_join.drop(columns=["matched_name_norm", "score"]), on="customer_name_norm", how="left", suffixes=("", "_fuzzy"))
                        for col in ["industry", "industry_sub", "web_address", "cleaned_customer_name", "industry_reasoning"]:
                            dim_pd[col] = dim_pd[col].where(dim_pd[col].notna(), dim_pd[f"{col}_fuzzy"])  # fill nulls with fuzzy
                            fcol = f"{col}_fuzzy"
                            if fcol in dim_pd.columns:
                                dim_pd.drop(columns=[fcol], inplace=True)
                        dim_customer = pl.from_pandas(dim_pd)
                        logger.info("Fuzzy matching applied to fill remaining industry data.")
            else:
                logger.info("rapidfuzz not available; skipping fuzzy fallback.")
        except Exception as e:
            logger.warning(f"Fuzzy match fallback failed: {e}")
        logger.info(f"Successfully created dim_customer table with {len(dim_customer)} customers, industry data available for {dim_customer.filter(pl.col('industry').is_not_null()).height} customers.")
    else:
        dim_customer = dim_customer_base.collect()
        logger.info(f"Successfully created dim_customer table with {len(dim_customer)} unique customers (no industry data).")

    dim_customer.write_database("dim_customer", engine, if_table_exists="replace")

    # Write industry coverage report
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        dim_pd = dim_customer.to_pandas()
        total = len(dim_pd)
        with_industry = int(dim_pd['industry'].notna().sum())
        summary_rows = [
            {"metric": "total_customers", "value": total},
            {"metric": "with_industry", "value": with_industry},
            {"metric": "coverage_pct", "value": round(with_industry/total*100, 2) if total else 0},
        ]
        top_ind = (
            dim_pd[['industry']]
            .dropna()
            .value_counts()
            .reset_index()
            .rename(columns={0: 'count'})
            .head(50)
        )
        top_sub = (
            dim_pd[['industry_sub']]
            .dropna()
            .value_counts()
            .reset_index()
            .rename(columns={0: 'count'})
            .head(50)
        )
        pd.DataFrame(summary_rows).to_csv(OUTPUTS_DIR / 'industry_coverage_summary.csv', index=False)
        top_ind.to_csv(OUTPUTS_DIR / 'industry_top50.csv', index=False)
        top_sub.to_csv(OUTPUTS_DIR / 'sub_industry_top50.csv', index=False)
        logger.info("Wrote industry coverage reports to outputs directory.")
    except Exception as e:
        logger.warning(f"Failed to write industry coverage reports: {e}")

    # --- 2. Define SKU and Division Mapping ---
    # Centralized mapping for unpivot operation.
    sku_mapping = get_sku_mapping()

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
    fact_transactions_pd['order_date'] = coerce_datetime(fact_transactions_pd['Rec Date'])

    # Clean currency columns using shared cleaner
    fact_transactions_pd['gross_profit'] = fact_transactions_pd['gross_profit'].apply(clean_currency_value)
    fact_transactions_pd['quantity'] = pd.to_numeric(fact_transactions_pd['quantity'], errors='coerce').fillna(0)
    
    # Filter out rows with no meaningful transaction data
    fact_transactions_pd = fact_transactions_pd[
        (fact_transactions_pd['gross_profit'] != 0) | (fact_transactions_pd['quantity'] != 0)
    ]
    
    # Select and deterministically sort final columns
    fact_transactions_pd = fact_transactions_pd[[
        'customer_id', 'order_date', 'product_sku', 'product_division', 'gross_profit', 'quantity'
    ]].sort_values(by=['customer_id', 'order_date', 'product_sku'], kind='mergesort').reset_index(drop=True)
    
    # Convert back to polars
    fact_transactions = pl.from_pandas(fact_transactions_pd)

    fact_transactions.write_database("fact_transactions", engine, if_table_exists="replace")
    logger.info(f"Successfully created fact_transactions table with {len(fact_transactions)} total line items.")

    # Row count audit
    try:
        row_counts = pd.DataFrame([
            {"table": "sales_log", "rows": int(len(sales_log_pd))},
            {"table": "dim_customer", "rows": int(len(dim_customer))},
            {"table": "fact_transactions", "rows": int(len(fact_transactions))},
        ])
        (OUTPUTS_DIR / "contracts").mkdir(parents=True, exist_ok=True)
        row_counts.to_csv(OUTPUTS_DIR / "contracts" / "row_counts.csv", index=False)
    except Exception:
        pass

    # Sample heads for quick inspection
    try:
        samples_dir = OUTPUTS_DIR / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        dim_customer.head(20).to_pandas().to_csv(samples_dir / "dim_customer_head.csv", index=False)
        fact_transactions.head(50).to_pandas().to_csv(samples_dir / "fact_transactions_head.csv", index=False)
    except Exception:
        pass
    
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
