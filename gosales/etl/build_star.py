"""Transform raw NetSuite extracts into a curated star schema.

The GoSales modeling stack relies on a tidy set of dimension and fact tables.
This module owns the heavy lifting: pulling raw sales logs, cleaning identifiers
and currencies, mapping SKUs to business divisions, and persisting parquet
snapshots.  Orchestrators and ad-hoc scripts import these helpers when they need
to rebuild or inspect the curated layer.
"""

import json
import re
import hashlib
from pathlib import Path

import pandas as pd
import polars as pl
try:
    from rapidfuzz import process, fuzz
    FUZZY_AVAILABLE = True
except Exception:
    FUZZY_AVAILABLE = False
from sqlalchemy import text
from gosales.utils.db import get_db_connection, get_curated_connection
from gosales.utils.sql import validate_identifier, ensure_allowed_identifier
from gosales.sql.queries import select_all
from gosales.utils.paths import OUTPUTS_DIR, DATA_DIR
from gosales.utils.config import load_config
from gosales.ops.run import run_context
from gosales.etl.ingest import robust_read_csv
from gosales.utils.logger import get_logger
# sku mapping handled within line-item ETL; no direct use here
from gosales.etl.cleaners import clean_currency_value, coerce_datetime, summarise_dataframe_schema
from gosales.utils.identifiers import normalize_identifier_expr, normalize_identifier_series
from gosales.etl.contracts import (
    check_required_columns,
    check_primary_key_not_null,
    check_no_duplicate_pk,
    violations_to_dataframe,
    check_date_parse_and_bounds,
)
from gosales.etl import build_fact_sales_line, summarise_sales_header
# No direct use of rollup/goal mapping in this module; handled in sales_line ETL

logger = get_logger(__name__)

_ROLLUP_NORMALIZE_RE = re.compile(r"[^0-9a-z]+")


def _normalize_rollup_key(label: str | None) -> str:
    if not label:
        return ""
    return " ".join(_ROLLUP_NORMALIZE_RE.sub(" ", str(label).strip().lower()).split())

def _checksum_parquet(parquet_path: Path) -> str:
    h = hashlib.sha256()
    with open(parquet_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _ensure_parquet_dirs(curated_dir: Path) -> None:
    (curated_dir / "fact").mkdir(parents=True, exist_ok=True)
    (curated_dir / "dim").mkdir(parents=True, exist_ok=True)


def build_star_schema(
    engine,
    config_path: str | Path | None = None,
    rebuild: bool = False,
    staging_only: bool = False,
    fail_soft: bool = False,
    use_line_item_facts: bool | None = None,
):
    """
    Build the curated star schema using the authoritative line-item sales fact.

    This implementation no longer reads or depends on the legacy `sales_log`
    header view. It performs the following steps deterministically:
    1.  Build `fact_sales_line` from `dbo.table_saleslog_detail` (or configured
        equivalent), including USD-normalized monetary columns and division/rollup
        metadata.
    2.  Create a clean `dim_customer` roster, blending NetSuite and Assets where
        available, with optional industry enrichment and fuzzy matching.
    3.  Project the line-item fact into tidy `fact_transactions` for downstream
        features, preserving invoice ids and canonical product divisions.
    4.  Persist curated snapshots and minimal dimensions (`dim_date`, `dim_product`).

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
    """
    logger.info("Building star schema with new tidy transaction model...")
    cfg = load_config(config_path)
    if use_line_item_facts is not None:
        try:
            cfg.etl.line_items.use_line_item_facts = bool(use_line_item_facts)
        except Exception as override_error:  # pragma: no cover - defensive log
            logger.warning(
                "Unable to override line-item toggle via parameter (use_line_item_facts=%s): %s",
                use_line_item_facts,
                override_error,
            )
    curated_dir = Path(cfg.paths.curated)
    _ensure_parquet_dirs(curated_dir)

    # Destructive rebuild (curated layer only)
    if rebuild:
        try:
            logger.info("Rebuild requested: dropping curated tables and removing parquet snapshots")
            # Use curated engine and transactional DDL to allow rollback on failure
            with get_curated_connection().begin() as connection:
                connection.execute(text("DROP TABLE IF EXISTS fact_transactions;"))
                connection.execute(text("DROP TABLE IF EXISTS fact_sales_line;"))
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

    # Resolve enrichment table name (legacy Sales Log intentionally unsupported)
    db_sources = {}
    try:
        db_sources = dict(getattr(getattr(cfg, 'database', object()), 'source_tables', {}) or {})
    except Exception:
        db_sources = {}
    ind_src = db_sources.get('industry_enrichment', 'industry_enrichment')

    backend = getattr(getattr(engine, 'dialect', None), 'name', '')
    is_azure_like = backend in {'mssql'}
    if not is_azure_like:
        if isinstance(ind_src, str) and ind_src.lower() != 'csv' and ind_src != 'industry_enrichment':
            logger.info(
                "Local engine '%s' detected; using fallback industry enrichment table 'industry_enrichment' instead of '%s'.",
                backend or 'unknown',
                ind_src,
            )
            ind_src = 'industry_enrichment'

    # Engines: source (raw) and curated (write target)
    curated_engine = get_curated_connection()

    # Build the canonical line-item fact (authoritative source)
    line_items_cfg = getattr(getattr(cfg, "etl", object()), "line_items", None)
    logger.info("Building fact_sales_line from line-item sales detail source...")
    try:
        fact_sales_line = build_fact_sales_line(engine=engine, config_path=config_path, cfg=cfg)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to build fact_sales_line: %s", exc)
        fact_sales_line = pl.DataFrame()

    if fact_sales_line.is_empty():
        logger.error("fact_sales_line is empty; cannot proceed without legacy Sales Log (disabled).")
        return
    else:
        sort_cols = [col for col in ["Sales_Order", "Rec_Date", "Item_internalid"] if col in fact_sales_line.columns]
        if sort_cols:
            fact_sales_line = fact_sales_line.sort(sort_cols, nulls_last=True)

        fact_sales_line.write_database("fact_sales_line", curated_engine, if_table_exists="replace")
        fact_line_parquet = curated_dir / "fact" / "fact_sales_line.parquet"
        fact_sales_line.write_parquet(fact_line_parquet)
        logger.info("fact_sales_line ready: %d rows", len(fact_sales_line))

        try:
            from gosales.etl import summarise_sales_header

            fact_sales_header = summarise_sales_header(fact_sales_line)
            fact_sales_header.write_database("fact_sales_header", curated_engine, if_table_exists="replace")
            fact_header_parquet = curated_dir / "fact" / "fact_sales_header.parquet"
            fact_sales_header.write_parquet(fact_header_parquet)
            logger.info("fact_sales_header ready: %d rows", len(fact_sales_header))
        except Exception as exc:
            logger.warning("Unable to build fact_sales_header from line items: %s", exc)

    # No legacy Sales Log: skip staging/contracts for that source entirely
    if staging_only:
        logger.info("Staging-only flag set; nothing to stage in curated without raw staging. Exiting.")
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

    # Base roster from line items (minimal; names supplied by NetSuite/Assets if available)
    dc_sales = (
        fact_sales_line.lazy()
        .select(["CompanyId"]) if "CompanyId" in fact_sales_line.columns else pl.LazyFrame(
            pl.DataFrame({"CompanyId": []})
        )
    )
    if not isinstance(dc_sales, pl.LazyFrame):  # defensive
        dc_sales = pl.DataFrame(dc_sales)
        dc_sales = dc_sales.lazy()
    dc_sales = (
        dc_sales
        .rename({"CompanyId": "customer_id"})
        .group_by("customer_id")
        .agg([
            pl.lit(None).cast(pl.Utf8).alias("customer_name"),
        ])
        .with_columns([
            normalize_identifier_expr(pl.col("customer_id")).alias("customer_id"),
            pl.lit(None).cast(pl.Utf8).alias("customer_name_norm"),
            pl.lit(None).cast(pl.Int64).alias("customer_prefix_id"),
            pl.lit(1).alias("source_sales_log"),
            pl.lit(0).alias("source_ns"),
            pl.lit(0).alias("source_assets"),
        ])
        .filter(pl.col("customer_id").is_not_null())
    )

    # Optional: union in NetSuite roster and Assets owners to expand inference roster
    try:
        sources_cfg = getattr(getattr(getattr(cfg, 'etl', object()), 'dim_customer', object()), 'sources', None)
        if not sources_cfg:
            # Default precedence: ns > assets (line-derived roster always included)
            sources = ["ns", "assets"]
        else:
            sources = [str(s).strip().lower() for s in sources_cfg]
    except Exception:
        sources = ["ns", "assets"]

    frames: list[pl.LazyFrame] = []
    rank_map = {name: idx for idx, name in enumerate([s for s in sources if s in {"ns","sales_log","assets"}], start=0)}

    # Sales base always present; assume CompanyId aligns to NetSuite internalid
    _order_cols = [
        "customer_id",
        "customer_name",
        "customer_name_norm",
        "customer_prefix_id",
        "internalid",
        "source_sales_log",
        "source_ns",
        "source_assets",
        "source_rank",
        "source",
    ]

    frames.append(
        dc_sales.with_columns([
            pl.lit(rank_map.get("sales_log", 1)).alias("source_rank"),
            pl.lit("line").alias("source"),
            # For line-derived roster, treat customer_id as the NetSuite internalid bridge
            pl.col("customer_id").alias("internalid"),
        ]).select(_order_cols)
    )

    if "ns" in sources:
        try:
            try:
                q = "SELECT internalid, entityid AS customer_name FROM dim_ns_customer"
                ns_pd = pd.read_sql(q, curated_engine)
            except Exception:
                q = "SELECT internalid, ns_companyname AS customer_name FROM dim_ns_customer"
                ns_pd = pd.read_sql(q, curated_engine)
            if not ns_pd.empty:
                ns_pd["internalid"] = normalize_identifier_series(ns_pd["internalid"]).astype(str)
                ns_pd["customer_name"] = ns_pd["customer_name"].astype(str).str.strip()
                ns_pd["customer_name_norm"] = ns_pd["customer_name"].str.lower().str.replace(r"\s+"," ", regex=True)
                ns_pd["customer_prefix_id"] = ns_pd["customer_name"].str.extract(r"^\s*(\d+)").astype("Int64")
                # For NS-only records (no line counterpart), set customer_id to internalid for a stable key,
                # and also carry the internalid bridge column explicitly
                ns_pd["customer_id"] = ns_pd["internalid"].astype(str)
                ns_pl = pl.from_pandas(ns_pd)[["customer_id","customer_name","customer_name_norm","customer_prefix_id","internalid"]]
                ns_pl = ns_pl.lazy().with_columns([
                    pl.lit(0).alias("source_sales_log"),
                    pl.lit(1).alias("source_ns"),
                    pl.lit(0).alias("source_assets"),
                    pl.lit(rank_map.get("ns", 0)).alias("source_rank"),
                    pl.lit("ns").alias("source"),
                ]).select(_order_cols)
                frames.append(ns_pl)
        except Exception as e:
            logger.warning(f"dim_ns_customer unavailable for roster union: {e}")

    if "assets" in sources:
        try:
            fa_pd = pd.read_sql("SELECT DISTINCT customer_id, customer_name FROM fact_assets", curated_engine)
            if not fa_pd.empty:
                fa_pd["customer_id"] = normalize_identifier_series(fa_pd["customer_id"]).astype(str)
                fa_pd["customer_name"] = fa_pd["customer_name"].astype(str).str.strip()
                fa_pd["customer_name_norm"] = fa_pd["customer_name"].str.lower().str.replace(r"\s+"," ", regex=True)
                fa_pd["customer_prefix_id"] = fa_pd["customer_name"].str.extract(r"^\s*(\d+)").astype("Int64")
                # No internalid for assets-only rows; include null bridge
                fa_pd["internalid"] = None
                fa_pl = pl.from_pandas(fa_pd)[["customer_id","customer_name","customer_name_norm","customer_prefix_id","internalid"]]
                fa_pl = fa_pl.lazy().with_columns([
                    pl.lit(0).alias("source_sales_log"),
                    pl.lit(0).alias("source_ns"),
                    pl.lit(1).alias("source_assets"),
                    pl.lit(rank_map.get("assets", 2)).alias("source_rank"),
                    pl.lit("assets").alias("source"),
                ]).select(_order_cols)
                frames.append(fa_pl)
        except Exception as e:
            logger.warning(f"fact_assets unavailable for roster union: {e}")

    # Union frames and deduplicate by customer_id with precedence
    if frames:
        dim_customer_base = pl.concat(frames, how="vertical_relaxed").with_columns([
            pl.col("customer_id").cast(pl.Utf8),
            pl.col("internalid").cast(pl.Utf8)
        ])
        # Keep best source per customer_id
        dim_customer_base = (
            dim_customer_base
            .sort(["customer_id","source_rank"])  # lower rank first (ns over sales over assets)
            .group_by("customer_id")
            .agg([
                pl.col("customer_name").first().alias("customer_name"),
                pl.col("customer_name_norm").first().alias("customer_name_norm"),
                pl.col("customer_prefix_id").first().alias("customer_prefix_id"),
                pl.col("source_sales_log").max().alias("source_sales_log"),
                pl.col("source_ns").max().alias("source_ns"),
                pl.col("source_assets").max().alias("source_assets"),
                pl.col("source").first().alias("source"),
                pl.col("source_rank").first().alias("source_rank"),
                pl.col("internalid").first().alias("internalid"),
            ])
        ).lazy()
    else:
        dim_customer_base = dc_sales

    # Curated raw line metadata for Branch/Rep features from line items
    select_exprs = []
    for col in ("CompanyId", "Rec_Date", "Branch", "Rep", "Division", "Invoice_Date", "Sales_Order"):
        if col in fact_sales_line.columns:
            select_exprs.append(pl.col(col))

    fact_sales_log_raw = (
        fact_sales_line.lazy()
        .select(select_exprs)
        .rename({
            "CompanyId": "customer_id",
            "Rec_Date": "order_date",
            "Division": "division",
            "Invoice_Date": "invoice_date",
            "Sales_Order": "invoice_id",
        })
        .with_columns([
            normalize_identifier_expr(pl.col("customer_id")).alias("customer_id"),
            pl.col("order_date").cast(pl.Date, strict=False).alias("order_date"),
            pl.col("division").cast(pl.Utf8),
            (pl.col("Branch").cast(pl.Utf8).alias("branch") if "Branch" in fact_sales_line.columns else pl.lit(None).alias("branch")),
            (pl.col("Rep").cast(pl.Utf8).alias("rep") if "Rep" in fact_sales_line.columns else pl.lit(None).alias("rep")),
            ((pl.col("invoice_date").cast(pl.Date, strict=False).alias("invoice_date")) if "Invoice_Date" in fact_sales_line.columns else pl.lit(None).alias("invoice_date")),
            ((normalize_identifier_expr(pl.col("invoice_id")).alias("invoice_id")) if "Sales_Order" in fact_sales_line.columns else pl.lit(None).alias("invoice_id")),
        ])
        .select(["customer_id","order_date","branch","rep","division","invoice_date","invoice_id"])
        .filter(pl.col("customer_id").is_not_null())
    )

    if industry_enrichment is not None:
        # Join with industry enrichment data
        # Note: Mapping customer roster (from line facts) to industry enrichment IDs
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
            unmatched = dim_customer.filter(pl.col("industry").is_null())
            total = int(len(dim_customer)) if dim_customer is not None else 0
            with_ind = int(dim_customer.filter(pl.col('industry').is_not_null()).height)
            cov = (with_ind / total) if total else 0.0
            do_fuzzy = bool(getattr(getattr(cfg, 'etl', object()), 'enable_industry_fuzzy', False))
            min_unmatched = int(getattr(getattr(cfg, 'etl', object()), 'fuzzy_min_unmatched', 50))
            skip_cov_ge = float(getattr(getattr(cfg, 'etl', object()), 'fuzzy_skip_if_coverage_ge', 0.95))
            max_unmatched = int(getattr(getattr(cfg, 'etl', object()), 'fuzzy_max_unmatched', 20000))

            # Apply cached matches first to shrink problem size
            try:
                cache_path = getattr(getattr(cfg, 'etl', object()), 'fuzzy_cache_path', None) or (OUTPUTS_DIR / 'industry_fuzzy_matches.csv')
                import os
                if cache_path and os.path.exists(cache_path):
                    cache_pd = pd.read_csv(cache_path)
                    if not cache_pd.empty and 'customer_name_norm' in cache_pd.columns:
                        dim_pd = dim_customer.to_pandas()
                        before = dim_pd['industry'].notna().sum()
                        dim_pd = dim_pd.merge(
                            cache_pd[['customer_name_norm','industry','industry_sub','web_address','cleaned_customer_name','industry_reasoning']].drop_duplicates('customer_name_norm'),
                            on='customer_name_norm', how='left', suffixes=('', '_cache')
                        )
                        for col in ['industry','industry_sub','web_address','cleaned_customer_name','industry_reasoning']:
                            dim_pd[col] = dim_pd[col].where(dim_pd[col].notna(), dim_pd.get(f"{col}_cache"))
                            c = f"{col}_cache"
                            if c in dim_pd.columns:
                                dim_pd.drop(columns=[c], inplace=True)
                        dim_customer = pl.from_pandas(dim_pd)
                        after = dim_pd['industry'].notna().sum()
                        logger.info(f"Applied industry cache: +{after-before} customers filled")
                        unmatched = dim_customer.filter(pl.col('industry').is_null())
            except Exception as _e:
                logger.warning(f"Industry cache join failed: {_e}")

            if not do_fuzzy:
                logger.info("Industry fuzzy matching disabled via config; skipping fuzzy fallback.")
            elif cov >= skip_cov_ge:
                logger.info(f"Industry coverage {cov:.2%} >= {skip_cov_ge:.0%} threshold; skipping fuzzy fallback.")
            elif len(unmatched) < min_unmatched:
                logger.info(f"Only {len(unmatched)} unmatched (< {min_unmatched}); skipping fuzzy fallback.")
            elif len(unmatched) > max_unmatched:
                logger.info(f"Skipping fuzzy: unmatched {len(unmatched)} exceeds fuzzy_max_unmatched={max_unmatched}")
            elif FUZZY_AVAILABLE:
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

                    # Persist fuzzy pairs for audit and as cache
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
    # Mapping now occurs within the line-item build; no legacy unpivot required.

    # --- 3. Project line items to fact_transactions (no Sales Log) ---
    logger.info("Projecting fact_sales_line into fact_transactions (sku/goal/division from mapping)")

    tx_exprs = []
    if "CompanyId" in fact_sales_line.columns:
        tx_exprs.append(pl.col("CompanyId").alias("customer_id"))
    if "Rec_Date" in fact_sales_line.columns:
        tx_exprs.append(pl.col("Rec_Date").alias("order_date"))
    if "Sales_Order" in fact_sales_line.columns:
        tx_exprs.append(pl.col("Sales_Order").alias("invoice_id"))
    tx_exprs.append((pl.col("item_rollup").cast(pl.Utf8) if "item_rollup" in fact_sales_line.columns else pl.lit("UNKNOWN")).alias("product_sku"))
    tx_exprs.append((pl.col("division_canonical").cast(pl.Utf8) if "division_canonical" in fact_sales_line.columns else pl.lit("UNKNOWN")).alias("product_division"))
    if "division_goal" in fact_sales_line.columns:
        tx_exprs.append(pl.col("division_goal").cast(pl.Utf8).alias("product_goal"))
    # Monetary and qty
    if "GP_usd" in fact_sales_line.columns:
        gp_expr = pl.col("GP_usd")
    elif "GP" in fact_sales_line.columns:
        gp_expr = pl.col("GP")
    else:
        gp_expr = pl.lit(0.0)
    tx_exprs.append(gp_expr.alias("gross_profit"))
    # Include normalized revenue if present for ALS and composition
    if "Revenue_usd" in fact_sales_line.columns:
        tx_exprs.append(pl.col("Revenue_usd").cast(pl.Float64, strict=False).alias("revenue"))
    elif "Revenue" in fact_sales_line.columns:
        tx_exprs.append(pl.col("Revenue").cast(pl.Float64, strict=False).alias("revenue"))
    else:
        tx_exprs.append(pl.lit(0.0).alias("revenue"))
    # Use normalized per-line quantity when present; default to 1.0
    if "Line_Qty" in fact_sales_line.columns:
        tx_exprs.append(pl.col("Line_Qty").cast(pl.Float64, strict=False).alias("quantity"))
    else:
        tx_exprs.append(pl.lit(1.0).alias("quantity"))

    fact_transactions = (
        fact_sales_line.lazy()
        .select(tx_exprs)
        .with_columns([
            normalize_identifier_expr(pl.col("customer_id")).alias("customer_id"),
            pl.col("order_date").cast(pl.Date, strict=False).alias("order_date"),
            pl.col("product_sku").cast(pl.Utf8).str.strip_chars().alias("product_sku"),
            pl.col("product_division").cast(pl.Utf8).str.strip_chars().alias("product_division"),
            pl.col("product_goal").cast(pl.Utf8) if "division_goal" in fact_sales_line.columns else pl.lit(None).alias("product_goal"),
            pl.col("gross_profit").cast(pl.Float64, strict=False),
            pl.col("quantity").cast(pl.Float64, strict=False),
        ])
        .filter(pl.col("customer_id").is_not_null())
        .collect()
    )

    # Deterministic sort
    sort_cols = [c for c in ["customer_id", "order_date", "product_division", "product_sku"] if c in fact_transactions.columns]
    if sort_cols:
        fact_transactions = fact_transactions.sort(sort_cols, nulls_last=True)

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
    parser.add_argument(
        "--use-line-items",
        dest="use_line_items",
        action="store_true",
        help="Enable line-item fact build (writes fact_sales_line).",
    )
    parser.add_argument(
        "--no-use-line-items",
        dest="use_line_items",
        action="store_false",
        help="Force-disable line-item fact build.",
    )
    parser.set_defaults(use_line_items=None)
    args = parser.parse_args()

    with run_context("PHASE0") as rc:
        # Persist resolved config
        cfg = load_config(args.config)
        if args.use_line_items is not None:
            try:
                cfg.etl.line_items.use_line_item_facts = bool(args.use_line_items)
            except Exception:
                logger.warning("Unable to set line-item toggle for recorded manifest")
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
        build_star_schema(
            db_engine,
            config_path=args.config,
            rebuild=args.rebuild,
            staging_only=args.staging_only,
            fail_soft=args.fail_soft,
            use_line_item_facts=args.use_line_items,
        )

