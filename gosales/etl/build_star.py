import polars as pl
import pandas as pd
import json
try:
    from rapidfuzz import process, fuzz
    FUZZY_AVAILABLE = True
except Exception:
    FUZZY_AVAILABLE = False
from sqlalchemy import text
from pathlib import Path
import hashlib
from gosales.utils.db import get_db_connection, get_curated_connection
from gosales.utils.sql import validate_identifier, ensure_allowed_identifier
from gosales.sql.queries import select_all
from gosales.utils.paths import OUTPUTS_DIR, DATA_DIR
from gosales.utils.config import load_config
from gosales.ops.run import run_context
from gosales.etl.ingest import robust_read_csv
from gosales.utils.logger import get_logger
from gosales.etl.sku_map import get_sku_mapping
from gosales.etl.cleaners import clean_currency_value, coerce_datetime, summarise_dataframe_schema
from gosales.etl.contracts import (
    check_required_columns,
    check_primary_key_not_null,
    check_no_duplicate_pk,
    violations_to_dataframe,
    check_date_parse_and_bounds,
)

logger = get_logger(__name__)

def _checksum_parquet(parquet_path: Path) -> str:
    h = hashlib.sha256()
    with open(parquet_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _ensure_parquet_dirs(curated_dir: Path) -> None:
    (curated_dir / "fact").mkdir(parents=True, exist_ok=True)
    (curated_dir / "dim").mkdir(parents=True, exist_ok=True)


def build_star_schema(engine, config_path: str | Path | None = None, rebuild: bool = False, staging_only: bool = False, fail_soft: bool = False):
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
    cfg = load_config(config_path)
    curated_dir = Path(cfg.paths.curated)
    _ensure_parquet_dirs(curated_dir)

    # Destructive rebuild (curated layer only)
    if rebuild:
        try:
            logger.info("Rebuild requested: dropping curated tables and removing parquet snapshots")
            # Use curated engine and transactional DDL to allow rollback on failure
            with get_curated_connection().begin() as connection:
                connection.execute(text("DROP TABLE IF EXISTS fact_transactions;"))
                connection.execute(text("DROP TABLE IF EXISTS dim_customer;"))
                connection.execute(text("DROP TABLE IF EXISTS dim_product;"))
                connection.execute(text("DROP TABLE IF EXISTS dim_date;"))
        except Exception as e:
            logger.warning(f"Failed to drop curated tables: {e}")
        # Remove parquet files (keep directory)
        try:
            for sub in [curated_dir / "fact", curated_dir / "dim"]:
                for p in sub.glob("*.parquet"):
                    p.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean curated parquet: {e}")

    # Resolve source table/view names from config (supports schema-qualified names)
    db_sources = {}
    try:
        db_sources = dict(getattr(getattr(cfg, 'database', object()), 'source_tables', {}) or {})
    except Exception:
        db_sources = {}
    sales_log_src = db_sources.get('sales_log', 'sales_log')
    ind_src = db_sources.get('industry_enrichment', 'industry_enrichment')

    # Engines: source (raw) and curated (write target)
    curated_engine = get_curated_connection()

    # Read the raw data from the database using pandas first, then convert to polars
    logger.info("Reading sales_log table...")
    try:
        if isinstance(sales_log_src, str) and sales_log_src.lower() != "csv":
            allow = set(getattr(getattr(cfg, 'database', object()), 'allowed_identifiers', []) or [])
            if allow:
                ensure_allowed_identifier(sales_log_src, allow)
            else:
                validate_identifier(sales_log_src)
    except Exception as e:
        raise
    try:
        # Chunked read for large sources
        def _read_sql_chunks(sql: str, params: dict | None = None, chunksize: int = 200_000) -> pd.DataFrame:
            try:
                it = pd.read_sql_query(sql, engine, params=params, chunksize=chunksize)
                frames = [chunk for chunk in it]
                if not frames:
                    return pd.DataFrame()
                return pd.concat(frames, ignore_index=True)
            except Exception:
                return pd.read_sql(sql, engine)

        # Centralized query template
        allow = set(getattr(getattr(cfg, 'database', object()), 'allowed_identifiers', []) or [])
        q_sales = select_all(sales_log_src, allowlist=allow if allow else None)
        sales_log_pd = _read_sql_chunks(q_sales)

        # --- Normalize schema to canonical wide format ---
        def normalize_sales_log_schema(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            # Trim column names
            df.columns = [str(c).strip() for c in df.columns]

            # Create canonical columns from exact DB headers provided via config
            for col in ["CustomerId", "Rec Date", "Division", "Customer", "InvoiceId"]:
                if col not in df.columns:
                    df[col] = None
            src_map = {}
            try:
                src_map = dict(getattr(getattr(cfg, 'etl', object()), 'source_columns', {}) or {})
            except Exception:
                src_map = {}
            cust_col = src_map.get('customer_id')
            date_col = src_map.get('order_date')
            div_col = src_map.get('division')
            name_col = src_map.get('customer_name')
            inv_col = src_map.get('invoice_id')
            if cust_col and cust_col in df.columns:
                df['CustomerId'] = df[cust_col]
            if date_col and date_col in df.columns:
                df['Rec Date'] = df[date_col]
            if inv_col and inv_col in df.columns:
                df['InvoiceId'] = df[inv_col]
            if div_col and div_col in df.columns:
                df['Division'] = df[div_col]
            if name_col and name_col in df.columns:
                df['Customer'] = df[name_col]
            # Final fallback for Customer name
            if df['Customer'].isna().all():
                try:
                    df['Customer'] = df['CustomerId'].astype(str)
                except Exception:
                    df['Customer'] = ""

            # Ensure all mapped GP/Qty columns exist
            mapping_local = get_sku_mapping()
            for gp_col, meta in mapping_local.items():
                qty_col = meta["qty_col"]
                if gp_col not in df.columns:
                    df[gp_col] = 0
                if qty_col not in df.columns:
                    df[qty_col] = 0

            # Keep only rows with valid identifiers
            def _is_filled(x: pd.Series) -> pd.Series:
                return (~x.isna()) & (x.astype(str).str.strip() != "")

            df = df[_is_filled(df["CustomerId"]) & _is_filled(df["Rec Date"])].copy()

            # Drop exact duplicate PKs (keep first)
            df = df.drop_duplicates(subset=["CustomerId", "Rec Date"], keep="first")
            return df

        sales_log_pd = normalize_sales_log_schema(sales_log_pd)

        # Data contracts: required columns and PK/null checks
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        contracts_dir = OUTPUTS_DIR / "contracts"
        contracts_dir.mkdir(parents=True, exist_ok=True)

        # Persist schema snapshot & staging parquet and profile
        schema_json = summarise_dataframe_schema(sales_log_pd)
        with open(contracts_dir / "schema.json", "w", encoding="utf-8") as f:
            json.dump(schema_json, f, indent=2)

        # Write normalized staging parquet (lower snake case headers for a clean copy)
        staging_dir = Path(cfg.paths.staging)
        staging_dir.mkdir(parents=True, exist_ok=True)
        try:
            norm_cols = [
                (
                    str(c)
                    .strip()
                    .lower()
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("__", "_")
                )
                for c in sales_log_pd.columns
            ]
            sales_log_pd_norm = sales_log_pd.copy()
            # Ensure unique normalized columns by de-duplicating with suffixes
            seen: dict[str,int] = {}
            uniq_cols: list[str] = []
            for c in norm_cols:
                if c in seen:
                    seen[c] += 1
                    uniq_cols.append(f"{c}__{seen[c]}")
                else:
                    seen[c] = 0
                    uniq_cols.append(c)
            sales_log_pd_norm.columns = uniq_cols
            pl.from_pandas(sales_log_pd_norm).write_parquet(staging_dir / "sales_log_normalized.parquet")
        except Exception as e:
            logger.warning(f"Failed to write staging parquet: {e}")

        # Column profile
        try:
            prof = []
            for col in sales_log_pd.columns:
                series = sales_log_pd[col]
                null_pct = float(series.isna().mean()) if hasattr(series, 'isna') else 0.0
                card = int(series.nunique(dropna=True)) if hasattr(series, 'nunique') else 0
                prof.append({"column": col, "null_pct": round(null_pct, 6), "cardinality": card, "dtype": str(series.dtype)})
            with open(contracts_dir / "column_profile.json", "w", encoding="utf-8") as f:
                json.dump(prof, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write column profile: {e}")

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
        # Date bounds
        try:
            from pandas import Timestamp
            maxd = pd.to_datetime(cfg.run.cutoff_date, errors="coerce") if cfg.run.cutoff_date else None
            violations += check_date_parse_and_bounds(sales_log_pd, "sales_log", "Rec Date", maxd)
        except Exception:
            pass

        vdf = violations_to_dataframe(violations)
        vdf.to_csv(contracts_dir / "violations.csv", index=False)

        # Block on PK/null/duplicate violations unless fail_soft
        if any(v.violation_type in {"null_in_pk", "missing_pk_column", "duplicate_pk"} for v in violations):
            if fail_soft:
                logger.warning("Contract violations present, continuing due to fail_soft=True")
            else:
                logger.error("Data contract violations detected. See outputs/contracts/violations.csv")
                return

        if staging_only:
            logger.info("Staging-only flag set; stopping after contracts.")
            return

        sales_log = pl.from_pandas(sales_log_pd)
    except Exception as e:
        logger.error(f"Failed to read sales_log table: {e}")
        return

    # --- 1. Create dim_customer with industry enrichment ---
    logger.info("Creating dim_customer table with industry enrichment...")

    # Read industry enrichment data if available
    try:
        if isinstance(ind_src, str) and ind_src.lower() == 'csv':
            # Read from CSV path in config
            try:
                csv_path = getattr(getattr(cfg, 'etl', object()), 'industry_enrichment_csv', None)
                if not csv_path:
                    # Default fallback path
                    csv_path = DATA_DIR / 'database_samples' / 'TR - Industry Enrichment.csv'
                else:
                    csv_path = Path(csv_path)
                ie_pd = robust_read_csv(csv_path)
                industry_enrichment = pl.from_pandas(ie_pd)
                logger.info(f"Loaded industry enrichment CSV with {len(industry_enrichment)} records from {csv_path}.")
            except Exception as ee:
                logger.warning(f"Failed reading industry enrichment CSV: {ee}. Falling back to DB if available...")
                industry_enrichment_pd = pd.read_sql("SELECT * FROM industry_enrichment", engine)
                industry_enrichment = pl.from_pandas(industry_enrichment_pd)
                logger.info(f"Successfully loaded industry enrichment data with {len(industry_enrichment)} records.")
        else:
            # Validate identifier for DB-backed enrichment source
            try:
                allow = set(getattr(getattr(cfg, 'database', object()), 'allowed_identifiers', []) or [])
                if allow:
                    ensure_allowed_identifier(ind_src, allow)
                else:
                    validate_identifier(ind_src)
            except Exception as ve:
                raise
            q_ind = select_all(ind_src, allowlist=allow if allow else None)
            industry_enrichment_pd = _read_sql_chunks(q_ind)
            industry_enrichment = pl.from_pandas(industry_enrichment_pd)
            logger.info(f"Successfully loaded industry enrichment data with {len(industry_enrichment)} records.")
    except Exception as e:
        logger.warning(f"Industry enrichment source not found/loaded: {e}")
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
            # Keep customer_id as string (GUID-safe) for downstream joins
            pl.col("customer_id").cast(pl.Utf8),
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

    # Preserve raw sales_log data for Branch/Rep features
    # This ensures we have the original Branch and Rep information available
    fact_sales_log_raw = (
        sales_log.lazy()
        .select([
            "CustomerId", "Rec Date", "branch", "rep", "Customer",
            "Division", "invoice_date", "InvoiceId"
        ])
        .rename({
            "CustomerId": "customer_id",
            "Rec Date": "order_date",
            "Customer": "customer_name",
            "Division": "division"
        })
        .with_columns([
            # Ensure consistent string typing for customer_id
            pl.col("customer_id").cast(pl.Utf8),
            # Preserve original columns for feature engineering (already lowercase)
            pl.col("branch").cast(pl.Utf8),
            pl.col("rep").cast(pl.Utf8),
            pl.col("order_date").cast(pl.Date),
            pl.col("division").cast(pl.Utf8),
            pl.col("invoice_date").cast(pl.Date),
            pl.col("InvoiceId").alias("invoice_id").cast(pl.Utf8),
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
        
        # C) Optional fuzzy-match fallback for remaining unmatched names (configurable)
        try:
            if FUZZY_AVAILABLE:
                unmatched = dim_customer.filter(pl.col("industry").is_null())
                total = int(len(dim_customer)) if dim_customer is not None else 0
                with_ind = int(dim_customer.filter(pl.col('industry').is_not_null()).height)
                cov = (with_ind / total) if total else 0.0
                do_fuzzy = bool(getattr(getattr(cfg, 'etl', object()), 'enable_industry_fuzzy', True))
                min_unmatched = int(getattr(getattr(cfg, 'etl', object()), 'fuzzy_min_unmatched', 50))
                skip_cov_ge = float(getattr(getattr(cfg, 'etl', object()), 'fuzzy_skip_if_coverage_ge', 0.95))
                if not do_fuzzy:
                    logger.info("Industry fuzzy matching disabled via config; skipping fuzzy fallback.")
                elif cov >= skip_cov_ge:
                    logger.info(f"Industry coverage {cov:.2%} >= {skip_cov_ge:.0%} threshold; skipping fuzzy fallback.")
                elif len(unmatched) < min_unmatched:
                    logger.info(f"Only {len(unmatched)} unmatched (< {min_unmatched}); skipping fuzzy fallback.")
                else:
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

    dim_customer.write_database("dim_customer", curated_engine, if_table_exists="replace")
    # Write Parquet snapshot
    dim_customer_parquet = curated_dir / "dim" / "dim_customer.parquet"
    dim_customer.write_parquet(dim_customer_parquet)

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
    if "InvoiceId" in sales_log.columns:
        id_vars.append("InvoiceId")

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
    fact_transactions_pd['customer_id'] = fact_transactions_pd['CustomerId'].astype(str)
    fact_transactions_pd['order_date'] = coerce_datetime(fact_transactions_pd['Rec Date'])
    if 'InvoiceId' in fact_transactions_pd.columns:
        fact_transactions_pd['invoice_id'] = fact_transactions_pd['InvoiceId'].astype(str)

    # Clean currency columns using shared cleaner
    fact_transactions_pd['gross_profit'] = fact_transactions_pd['gross_profit'].apply(clean_currency_value)
    fact_transactions_pd['quantity'] = pd.to_numeric(fact_transactions_pd['quantity'], errors='coerce').fillna(0)
    
    # Route product_division for SKUs that depend on source DB Division (e.g., AM_Support)
    try:
        _map = get_sku_mapping()
        routed = [
            (k, v.get('db_division_routes', {}))
            for k, v in _map.items()
            if isinstance(v, dict) and 'db_division_routes' in v
        ]
        if routed and 'Division' in fact_transactions_pd.columns:
            for sku_key, route in routed:
                if not route:
                    continue
                mask = fact_transactions_pd['product_sku'] == sku_key
                if mask.any():
                    # Map row-wise using source Division column
                    fact_transactions_pd.loc[mask, 'product_division'] = (
                        fact_transactions_pd.loc[mask, 'Division']
                        .map(lambda x: route.get(str(x).strip(), fact_transactions_pd.loc[mask, 'product_division'].iloc[0]))
                    )
    except Exception as e:
        logger.warning(f"Division routing step skipped/failed: {e}")

    # Filter out rows with no meaningful transaction data
    fact_transactions_pd = fact_transactions_pd[
        (fact_transactions_pd['gross_profit'] != 0) | (fact_transactions_pd['quantity'] != 0)
    ]
    
    # Normalize division strings (trim) and select/sort final columns
    fact_transactions_pd['product_division'] = fact_transactions_pd['product_division'].astype(str).str.strip()
    final_cols = ['customer_id', 'order_date', 'product_sku', 'product_division', 'gross_profit', 'quantity']
    if 'invoice_id' in fact_transactions_pd.columns:
        final_cols.insert(1, 'invoice_id')
    fact_transactions_pd = fact_transactions_pd[final_cols] \
        .sort_values(by=[c for c in final_cols if c in fact_transactions_pd.columns], kind='mergesort') \
        .reset_index(drop=True)
    
    # Round monetary and quantities for idempotency
    if not fact_transactions_pd.empty:
        fact_transactions_pd['gross_profit'] = fact_transactions_pd['gross_profit'].round(2)
        fact_transactions_pd['quantity'] = fact_transactions_pd['quantity'].round(0)

    # Convert back to polars
    fact_transactions = pl.from_pandas(fact_transactions_pd)

    fact_transactions.write_database("fact_transactions", curated_engine, if_table_exists="replace")

    # Write raw sales log data for Branch/Rep features
    fact_sales_log_raw = fact_sales_log_raw.collect()
    fact_sales_log_raw.write_database("fact_sales_log_raw", curated_engine, if_table_exists="replace")
    # Write Parquet snapshot and checksum
    fact_parquet = curated_dir / "fact" / "fact_transactions.parquet"
    fact_transactions.write_parquet(fact_parquet)
    logger.info(f"Successfully created fact_transactions table with {len(fact_transactions)} total line items.")

    # Create indexes on common join/filter keys for performance
    try:
        with curated_engine.connect() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_customer ON fact_transactions(customer_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_order_date ON fact_transactions(order_date)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fact_division ON fact_transactions(product_division)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_dim_customer_id ON dim_customer(customer_id)"))
            conn.commit()
        logger.info("Created indexes on fact_transactions and dim_customer")
    except Exception as e:
        logger.warning(f"Failed to create indexes: {e}")

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

    # --- QA summary & report ---
    try:
        qa_dir = OUTPUTS_DIR / 'qa'
        qa_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "tables": {
                "fact_transactions": {
                    "rows": int(len(fact_transactions)),
                    "sum_gp": float(fact_transactions.select(pl.col("gross_profit").sum()).item()),
                    "checksum": _checksum_parquet(fact_parquet) if fact_parquet.exists() else None,
                },
                "dim_customer": {
                    "rows": int(len(dim_customer)),
                    "checksum": _checksum_parquet(dim_customer_parquet) if dim_customer_parquet.exists() else None,
                },
            }
        }
        with open(qa_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        # Minimal markdown report
        with open(qa_dir / 'phase0_report.md', 'w', encoding='utf-8') as f:
            f.write("# Phase 0 QA Report\n\n")
            f.write(f"Fact rows: {summary['tables']['fact_transactions']['rows']}\\n\n")
            f.write(f"Dim customer rows: {summary['tables']['dim_customer']['rows']}\\n")
    except Exception as e:
        logger.warning(f"Failed to write QA summary/report: {e}")

    # Sample heads for quick inspection
    try:
        samples_dir = OUTPUTS_DIR / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        dim_customer.head(20).to_pandas().to_csv(samples_dir / "dim_customer_head.csv", index=False)
        fact_transactions.head(50).to_pandas().to_csv(samples_dir / "fact_transactions_head.csv", index=False)
    except Exception:
        pass
    
    # --- FK integrity (fact → dim_customer, dim_product, dim_date) ---
    try:
        # Prepare keys
        dim_c = dim_customer.select([pl.col("customer_id")]).unique().with_columns(pl.col("customer_id").cast(pl.Utf8))
        fact_ck = fact_transactions.select([pl.col("customer_id")]).with_columns(pl.col("customer_id").cast(pl.Utf8))
        # Left anti joins to find missing
        missing_customer = fact_ck.join(dim_c, on="customer_id", how="anti")
        # Product and date integrity (best effort)
        dim_p = fact_transactions.select(["product_sku", "product_division"]).unique()
        missing_product = fact_transactions.select(["product_sku", "product_division"]).join(dim_p, on=["product_sku", "product_division"], how="anti")
        # Date: ensure not null
        missing_date = fact_transactions.filter(pl.col("order_date").is_null())

        quarantine_dir = curated_dir / "fact" / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        if len(missing_customer) > 0:
            missing_customer.write_parquet(quarantine_dir / "missing_customer.parquet")
        if len(missing_product) > 0:
            missing_product.write_parquet(quarantine_dir / "missing_product.parquet")
        if len(missing_date) > 0:
            missing_date.write_parquet(quarantine_dir / "missing_date.parquet")
    except Exception as e:
        logger.warning(f"FK integrity/quarantine step failed: {e}")

    # --- dim_date and dim_product (minimal) ---
    try:
        logger.info("Building dim_date and dim_product")
        # dim_date: derive from min/max order_date in facts
        fpd = fact_transactions.to_pandas()
        if not fpd.empty:
            mind = fpd["order_date"].min()
            maxd = fpd["order_date"].max()
            if pd.notna(mind) and pd.notna(maxd):
                rng = pd.date_range(mind, maxd, freq="D")
                dim_date = pl.DataFrame({
                    "date_key": [int(d.strftime("%Y%m%d")) for d in rng],
                    "date": [d.date().isoformat() for d in rng],
                    "year": [d.year for d in rng],
                    "quarter": [int((d.month - 1) / 3) + 1 for d in rng],
                    "month": [d.month for d in rng],
                })
                dim_date.write_database("dim_date", curated_engine, if_table_exists="replace")
                dim_date.write_parquet(curated_dir / "dim" / "dim_date.parquet")
        # dim_product: unique SKUs/divisions
        dp = (
            fact_transactions.select(["product_sku", "product_division"])\
            .unique()
        )
        dp.write_database("dim_product", curated_engine, if_table_exists="replace")
        dp.write_parquet(curated_dir / "dim" / "dim_product.parquet")
    except Exception as e:
        logger.warning(f"Failed to build dim_date/dim_product: {e}")

    # --- Deprecate old tables ---
    try:
        logger.info("Dropping old fact_orders and dim_product tables if legacy names exist...")
        with curated_engine.connect() as connection:
            connection.execute(text("DROP TABLE IF EXISTS fact_orders;"))
            connection.execute(text("DROP TABLE IF EXISTS dim_product_legacy;"))
            connection.commit()
    except Exception:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build curated star schema")
    parser.add_argument("--config", type=str, default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--staging-only", action="store_true")
    parser.add_argument("--fail-soft", action="store_true")
    args = parser.parse_args()

    with run_context("PHASE0") as rc:
        # Persist resolved config
        cfg = load_config(args.config)
        runs_dir = Path(rc["run_dir"])
        runs_dir.mkdir(parents=True, exist_ok=True)
        resolved_path = runs_dir / "config_resolved.yaml"
        try:
            import yaml

            with open(resolved_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
        except Exception:
            pass

        db_engine = get_db_connection()
        build_star_schema(db_engine, config_path=args.config, rebuild=args.rebuild, staging_only=args.staging_only, fail_soft=args.fail_soft)

