"""Utilities for building the line-item fact table from the new sales detail source.

The helpers in this module load ``dbo.table_saleslog_detail`` (or the configured
equivalent), collapse duplicate snapshots using ``last_update``, expose
``Amount2`` as COGS, and generate USD-normalised revenue/COGS/GP fields based on
``SalesOrder_Currency`` and ``USD_CAD_Conversion_rate``. The resulting Polars
DataFrame can be consumed by the curated star build or downstream feature
pipelines once the new fact is enabled.
"""

from __future__ import annotations

from typing import Iterable, Sequence
import re

import pandas as pd
import polars as pl

from gosales.utils.config import load_config
from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR
from gosales.sql.queries import select_all
from gosales.utils.sql import ensure_allowed_identifier, validate_identifier
from gosales.etl.sku_map import get_sku_mapping

logger = get_logger(__name__)


# Columns that downstream consumers expect to retain from the sales detail table
KEEP_COLUMNS: list[str] = [
    "Rec_Date",
    "Sales_Order",
    "Branch",
    "Division",
    "New_Business",
    "New",
    "Rep",
    "Referral_NS_Field",
    "Rev_type",
    "Item_internalid",
    "Amount2",
    "Revenue",
    "SWX_Core",
    "SWX_Core_Qty",
    "Core_New_UAP",
    "Core_New_UAP_Qty",
    "SWX_Pro_Prem",
    "SWX_Pro_Prem_Qty",
    "Pro_Prem_New_UAP",
    "Pro_Prem_New_UAP_Qty",
    "Simulation",
    "Simulation_Qty",
    "CAMWorks_Seats",
    "CAMWorks_Seats_Qty",
    "Misc_Seats",
    "Misc_Seats_Qty",
    "EPDM_CAD_Editor_Seats",
    "EPDM_CAD_Editor_Seats_Qty",
    "HV_Simulation",
    "HV_Simulation_Qty",
    "SW_Electrical",
    "SW_Electrical_Qty",
    "SW_Inspection",
    "SW_Inspection_Qty",
    "_3DX_Revenue",
    "_3DX_Revenue_Qty",
    "Training",
    "Training_Qty",
    "Services",
    "Services_Qty",
    "Fortus",
    "Fortus_Qty",
    "_1200_Elite_Fortus250",
    "_1200_Elite_Fortus250_Qty",
    "uPrint",
    "uPrint_Qty",
    "Altium_PCBWorks",
    "Altium_PCBWorks_Qty",
    "SWX_Core_Seats",
    "SWX_Pro_Prem_Seats",
    "Core_New_UAP_Seats",
    "Pro_Prem_New_UAP_Seats",
    "GP",
    "Term_GP",
    "Invoice_Date",
    "Created_date",
    "CompanyId",
    "SW_Plastics",
    "SW_Plastics_Qty",
    "CATIA",
    "CATIA_Qty",
    "FDM",
    "FDM_Qty",
    "AM_Software",
    "AM_Software_Qty",
    "Polyjet",
    "Polyjet_Qty",
    "P3",
    "P3_Qty",
    "SLA",
    "SLA_Qty",
    "SAF",
    "SAF_Qty",
    "Metals",
    "Metals_Qty",
    "_3DP_Software",
    "_3DP_Software_Qty",
    "_3DP_Training",
    "_3DP_Training_Qty",
    "Post_Processing",
    "Post_Processing_Qty",
    "Consumables",
    "Consumables_Qty",
    "FormLabs",
    "FormLabs_Qty",
    "Other_Misc",
    "Other_Misc_Qty",
    "Spare_Parts_Repair_Parts_Time_Materials",
    "Spare_Parts_Repair_Parts_Time_materials_Qty",
    "Creaform",
    "Creaform_Qty",
    "Artec",
    "Artec_Qty",
    "Success_Plan",
    "Success_Plan_Qty",
    "Success_Plan_Level",
    "AM_Support",
    "AM_Support_Qty",
    "Delmia",
    "Delmia_Qty",
    "CPE_YXC_Renewal",
    "CPE_YXC_Renewal_Qty",
    "SalesOrder_Currency",
    "USD_CAD_Conversion_rate",
    "Draftsight",
    "GeoMagic",
    "Draftsight_Qty",
    "GeoMagic_Qty",
]


DEFAULT_KEY_COLUMNS: list[str] = [
    "Sales_Order",
    "Item_internalid",
    "Revenue",
    "Amount2",
    "GP",
    "Term_GP",
]


DEFAULT_CURRENCY_COLUMNS: list[str] = ["Revenue", "Amount2", "GP", "Term_GP"]

DEFAULT_DATE_COLUMNS: list[str] = ["Rec_Date", "Invoice_Date", "Created_date"]

HEADER_GROUP_COLUMNS: list[str] = [
    "Sales_Order",
    "Rec_Date",
    "Branch",
    "Division",
    "New_Business",
    "New",
    "Rep",
    "Referral_NS_Field",
    "Rev_type",
    "CompanyId",
    "SalesOrder_Currency",
    "USD_CAD_Conversion_rate",
    "manual_adjustment_flag",
]


DIVISION_GOAL_COLUMN = "division_goal"
DIVISION_CANONICAL_COLUMN = "division_canonical"
ITEM_ROLLUP_COLUMN = "item_rollup"
UNKNOWN_DIVISION = "UNKNOWN"

_ROLLUP_DIVISION_OVERRIDES: dict[str, str] = {
    "draftsight": "Solidworks",
    "geomagic": "Scanning",
    "altium pcbworks": "Solidworks",
    "swood": "Solidworks",
    "3dx revenue": "CPE",
    "3dp training": "Training",
    "service": "Services",
    "new logo": "Services",
}

_GOAL_FALLBACK_MAP: dict[str, str] = {
    "cad": "Solidworks",
    "printers": "Hardware",
    "printer accessorials": "Hardware",
    "training services": "Services",
    "specialty software": "Solidworks",
    "miscellaneous": "Hardware",
    "cpe": "CPE",
    "scanners": "Scanning",
    "draftsight": "Solidworks",
    "geomagic": "Scanning",
    "new logo": "Services",
}


_NON_ALPHANUM_RE = re.compile(r"[^0-9a-z]+")


def _normalize_rollup_label(value: str | None) -> str:
    """Return a normalized key for rollup/goal comparisons."""

    if not value:
        return ""
    cleaned = _NON_ALPHANUM_RE.sub(" ", str(value).strip().lower())
    return " ".join(cleaned.split())


def _build_rollup_division_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    try:
        base_mapping = get_sku_mapping()
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Failed to load SKU mapping for division lookup: %s", exc)
        base_mapping = {}

    for gp_col, meta in base_mapping.items():
        norm = _normalize_rollup_label(gp_col)
        if not norm:
            continue
        division = (meta or {}).get("division")
        if division:
            lookup[norm] = division
    lookup.update(_ROLLUP_DIVISION_OVERRIDES)
    return lookup


def normalize_line_quantity(
    frame: pl.DataFrame,
    behavior_cfg,
) -> pl.DataFrame:
    """Add a normalized line quantity column `Line_Qty`.

    - Picks the first available candidate column from common names: Quantity, Qty, Line_Qty, SO_Qty.
    - Casts to Float64, fills nulls with 0, and applies absolute value to avoid negative counts.
    - Honors return_treatment semantics: when behavior returns are excluded upstream, quantity simply reflects counts; when `separate_flag`, quantity remains absolute while `is_return_line` encodes sign.
    """
    if frame.is_empty():
        return frame

    # Candidate names in preferred order
    candidates = [
        "Line_Quantity",
        "Quantity",
        "Qty",
        "Line_Qty",
        "SO_Qty",
        "Item_Qty",
    ]
    found = next((c for c in candidates if c in frame.columns), None)
    if found is None:
        # If not found, ensure column exists as 0.0
        return frame.with_columns(pl.lit(0.0).alias("Line_Qty"))

    col = pl.col(found).cast(pl.Float64, strict=False)

    # Always use absolute; sign handling belongs to revenue or explicit flags
    qty_expr = col.fill_null(0.0).abs()

    return frame.with_columns(qty_expr.alias("Line_Qty"))


_ROLLUP_DIVISION_LOOKUP = _build_rollup_division_lookup()


def _resolve_canonical_division(goal: str | None, rollup: str | None, legacy_division: str | None) -> str:
    """Return the canonical division derived from Goal/rollup with legacy fallback."""

    rollup_norm = _normalize_rollup_label(rollup)
    if rollup_norm and rollup_norm in _ROLLUP_DIVISION_LOOKUP:
        return _ROLLUP_DIVISION_LOOKUP[rollup_norm]

    goal_norm = _normalize_rollup_label(goal)
    if goal_norm and goal_norm in _GOAL_FALLBACK_MAP:
        return _GOAL_FALLBACK_MAP[goal_norm]

    legacy = (legacy_division or "").strip()
    if legacy:
        return legacy

    return UNKNOWN_DIVISION


def _resolve_source_table(cfg, logical_name: str, default: str | None = None) -> str | None:
    try:
        return getattr(getattr(cfg, "database", object()), "source_tables", {}).get(logical_name, default)
    except Exception:
        return default


def get_rollup_goal_mappings(engine=None, config_path: str | None = None, cfg=None) -> tuple[dict[str, str], dict[str, str]]:
    """Return dictionaries mapping normalized rollups to Goal and canonical divisions."""

    cfg = cfg or load_config(config_path)
    engine = engine or get_db_connection()

    line_cfg = getattr(getattr(cfg, "etl", object()), "line_items", None)
    sources_cfg = getattr(line_cfg, "sources", None)

    product_mapping_df = _load_product_tag_mapping(engine, cfg, sources_cfg)

    if product_mapping_df.is_empty():
        return {}, {}

    rollup_to_goal: dict[str, str] = {}
    rollup_to_division: dict[str, str] = {}

    for row in product_mapping_df.iter_rows(named=True):
        raw_rollup = row.get(ITEM_ROLLUP_COLUMN)
        norm_rollup = _normalize_rollup_label(raw_rollup)
        goal = row.get(DIVISION_GOAL_COLUMN)
        if not norm_rollup:
            continue
        if goal and norm_rollup not in rollup_to_goal:
            rollup_to_goal[norm_rollup] = str(goal)
        if norm_rollup not in rollup_to_division:
            rollup_to_division[norm_rollup] = _resolve_canonical_division(goal, raw_rollup, None)

    return rollup_to_goal, rollup_to_division


def _read_sales_detail(engine, cfg, table_name: str | None = None) -> pl.DataFrame:
    """Fetch the configured sales detail table into a Polars DataFrame."""

    default_table = "dbo.table_saleslog_detail"
    table = table_name or _resolve_source_table(cfg, "sales_detail", default_table)

    if isinstance(table, str) and table.lower() != "csv":
        allowed = set(getattr(getattr(cfg, "database", object()), "allowed_identifiers", []) or [])
        if allowed:
            ensure_allowed_identifier(table, allowed)
        else:
            validate_identifier(table)

    query = select_all(
        table,
        allowlist=set(getattr(getattr(cfg, "database", object()), "allowed_identifiers", []) or []) or None,
    )

    logger.info("Reading sales detail source `%s`", table)

    def _read_chunks(sql: str, chunk_size: int = 200_000) -> pd.DataFrame:
        try:
            iterator = pd.read_sql_query(sql, engine, chunksize=chunk_size)
            frames = [chunk for chunk in iterator]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)
        except Exception:
            return pd.read_sql(sql, engine)

    pdf = _read_chunks(query)
    if pdf.empty:
        logger.warning("Sales detail source returned no rows")
        return pl.DataFrame()

    # Strip whitespace from headers for safety
    pdf.columns = [str(col).strip() for col in pdf.columns]
    return pl.from_pandas(pdf)


def _read_source_table(engine, cfg, table_name: str | None) -> pl.DataFrame:
    """Generic helper to read a configured table into Polars."""

    if not table_name:
        return pl.DataFrame()
    if isinstance(table_name, str) and table_name.lower() == "csv":
        logger.warning("Source `%s` configured as CSV; skipping database read in sales_line ETL.", table_name)
        return pl.DataFrame()

    allowed = set(getattr(getattr(cfg, "database", object()), "allowed_identifiers", []) or [])
    query = select_all(
        table_name,
        allowlist=allowed or None,
    )

    def _read_chunks(sql: str, chunk_size: int = 200_000) -> pd.DataFrame:
        try:
            iterator = pd.read_sql_query(sql, engine, chunksize=chunk_size)
            frames = [chunk for chunk in iterator]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)
        except Exception:
            return pd.read_sql(sql, engine)

    pdf = _read_chunks(query)
    if pdf.empty:
        return pl.DataFrame()
    pdf.columns = [str(col).strip() for col in pdf.columns]
    return pl.from_pandas(pdf)


def _ensure_columns(frame: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    """Add missing columns as nulls so downstream selects do not fail."""

    missing = [col for col in columns if col not in frame.columns]
    if not missing:
        return frame
    additions = [pl.lit(None).alias(col) for col in missing]
    return frame.with_columns(additions)


def _normalize_dates(frame: pl.DataFrame, date_columns: Iterable[str]) -> pl.DataFrame:
    for col in date_columns:
        if col not in frame.columns:
            continue
        dtype = frame.schema.get(col)
        if dtype in (pl.Datetime, pl.Date):
            continue
        frame = frame.with_columns(
            pl.col(col)
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, strict=False, exact=True)
            .alias(col)
        )
    return frame


def _dedupe_snapshots(frame: pl.DataFrame, key_columns: Iterable[str], last_update_column: str) -> pl.DataFrame:
    if last_update_column not in frame.columns:
        logger.warning("`%s` column missing; duplicate collapse may be unstable", last_update_column)
        frame = frame.with_columns(pl.lit(None).alias(last_update_column))
    sorted_frame = frame.sort(last_update_column, descending=True, nulls_last=True)
    columns = [col for col in key_columns if col in frame.columns]
    if not columns:
        return sorted_frame
    return sorted_frame.unique(subset=columns, keep="first")


def _add_currency_columns(
    frame: pl.DataFrame,
    currency_columns: Iterable[str],
    currency_column: str,
    rate_column: str,
) -> pl.DataFrame:
    prepared = frame
    if currency_column not in prepared.columns:
        prepared = prepared.with_columns(pl.lit(None).alias(currency_column))
    if rate_column not in prepared.columns:
        prepared = prepared.with_columns(pl.lit(1.0).alias(rate_column))

    prepared = prepared.with_columns(
        pl.col(rate_column).cast(pl.Float64).fill_null(1.0).alias(rate_column),
        pl.col(currency_column).cast(pl.Utf8).str.to_uppercase().alias(currency_column),
    )

    fx_series = pl.when(pl.col(currency_column) == "CAD").then(
        pl.col(rate_column).fill_null(1.0)
    ).otherwise(1.0)

    usd_columns = []
    for column in currency_columns:
        if column not in prepared.columns:
            prepared = prepared.with_columns(pl.lit(None).alias(column))
        usd_columns.append((pl.col(column).cast(pl.Float64) * fx_series).alias(f"{column}_usd"))

    return prepared.with_columns(usd_columns)


def _load_product_tag_mapping(engine, cfg, sources_cfg) -> pl.DataFrame:
    """Return a mapping DataFrame linking product internal IDs to item rollups and Goal divisions."""

    product_table = getattr(sources_cfg, "product_info", None) if sources_cfg else None
    category_table = getattr(sources_cfg, "items_category_limited", None) if sources_cfg else None
    tags_table = getattr(sources_cfg, "product_tags", None) if sources_cfg else None

    if not tags_table:
        logger.info(
            "Product tags source missing; skipping division tags bootstrap."
        )
        return pl.DataFrame()

    product_df = _read_source_table(engine, cfg, product_table)
    category_df = _read_source_table(engine, cfg, category_table) if category_table else pl.DataFrame()
    tags_df = _read_source_table(engine, cfg, tags_table)

    if tags_df.is_empty():
        logger.warning(
            "Product tags source `%s` returned no rows; skipping division tags bootstrap.",
            tags_table,
        )
        return pl.DataFrame()

    tags_pd = tags_df.to_pandas()
    tags_pd.columns = [str(col).strip().casefold() for col in tags_pd.columns]
    if "item_rollup" not in tags_pd.columns:
        logger.warning(
            "Product tags source `%s` missing `item_rollup` column; skipping division tags bootstrap.",
            tags_table,
        )
        return pl.DataFrame()

    goal_col = next((col for col in ("goal", "division_goal") if col in tags_pd.columns), None)
    if goal_col is None:
        logger.warning(
            "Product tags source `%s` missing Goal field; skipping division tags bootstrap.",
            tags_table,
        )
        return pl.DataFrame()

    def _prepare_base(frame: pl.DataFrame, source_name: str | None) -> tuple[pd.DataFrame | None, str]:
        if frame.is_empty():
            return None, ""
        pdf = frame.to_pandas()
        pdf.columns = [str(col).strip().casefold() for col in pdf.columns]
        internal_id_col = next(
            (c for c in ("internalid", "internal_id", "item_internalid", "product_internal_id", "id") if c in pdf.columns),
            None,
        )
        rollup_col = next((c for c in ("item_rollup", "item rollup") if c in pdf.columns), None)
        if internal_id_col is None or rollup_col is None:
            return None, ""
        subset = pdf[[internal_id_col, rollup_col]].dropna(subset=[internal_id_col]).copy()
        if subset.empty:
            return None, ""
        subset[internal_id_col] = subset[internal_id_col].astype("string").str.strip()
        subset[rollup_col] = subset[rollup_col].astype("string").str.strip()
        subset.replace("", pd.NA, inplace=True)
        subset = subset.dropna(subset=[internal_id_col])
        subset = subset.rename(columns={internal_id_col: "product_internal_id", rollup_col: "item_rollup"})
        subset = subset.drop_duplicates(subset=["product_internal_id"], keep="first")
        return subset, str(source_name or "")

    base_pd: pd.DataFrame | None = None
    base_source = ""
    for candidate_df, source_name in ((category_df, category_table), (product_df, product_table)):
        prepared, src = _prepare_base(candidate_df, source_name)
        if prepared is not None:
            base_pd = prepared
            base_source = src or base_source
            break

    if base_pd is None:
        logger.warning(
            "No usable product metadata source found (items_category_limited=%r, product_info=%r); skipping division tags bootstrap.",
            category_table,
            product_table,
        )
        return pl.DataFrame()

    tags_subset = tags_pd[["item_rollup", goal_col]].dropna(subset=["item_rollup"]).copy()
    tags_subset["item_rollup"] = tags_subset["item_rollup"].astype("string").str.strip()
    tags_subset[goal_col] = tags_subset[goal_col].astype("string").str.strip()
    tags_subset.replace("", pd.NA, inplace=True)

    if tags_subset.empty:
        logger.warning(
            "Product tags source `%s` provided no item_rollup -> Goal pairs; division tags bootstrap will carry null Goals.",
            tags_table,
        )
        rollup_goal_map = pd.DataFrame(columns=["item_rollup", DIVISION_GOAL_COLUMN])
    else:
        counts = (
            tags_subset.groupby(["item_rollup", goal_col], dropna=False)
            .size()
            .reset_index(name="rows")
        )
        counts = counts.sort_values(["item_rollup", "rows", goal_col], ascending=[True, False, True])
        rollup_goal_map = counts.drop_duplicates(subset=["item_rollup"], keep="first")[["item_rollup", goal_col]]
        rollup_goal_map = rollup_goal_map.rename(columns={goal_col: DIVISION_GOAL_COLUMN})

    mapping = base_pd.merge(rollup_goal_map, on="item_rollup", how="left")
    mapping["product_internal_id"] = mapping["product_internal_id"].astype("string").str.strip()
    mapping["item_rollup"] = mapping["item_rollup"].astype("string").str.strip()
    mapping.replace("", pd.NA, inplace=True)

    mapping_df = pl.from_pandas(
        mapping[["product_internal_id", "item_rollup", DIVISION_GOAL_COLUMN]]
    )

    if mapping_df.is_empty():
        logger.warning(
            "Combined product/tag mapping produced no rows; skipping division tags bootstrap."
        )
    else:
        multi_goal_rollups = (
            mapping_df.group_by(ITEM_ROLLUP_COLUMN, maintain_order=False)
            .agg(pl.col(DIVISION_GOAL_COLUMN).n_unique().alias("goal_count"))
            .filter(pl.col("goal_count") > 1)
            .height
        )
        logger.info(
            "Product mapping source `%s` produced %s rows (%s distinct sub-divisions); multi-goal sub-divisions=%s.",
            base_source or "unknown",
            mapping_df.height,
            mapping_df.select(pl.col(ITEM_ROLLUP_COLUMN).n_unique()).item(),
            multi_goal_rollups,
        )

    return mapping_df


def _attach_division_metadata(
    frame: pl.DataFrame,
    engine,
    cfg,
    sources_cfg,
) -> pl.DataFrame:
    """Augment the sales detail frame with item rollup and Goal-derived divisions."""

    required = [ITEM_ROLLUP_COLUMN, DIVISION_GOAL_COLUMN, DIVISION_CANONICAL_COLUMN]
    product_mapping_df = _load_product_tag_mapping(engine, cfg, sources_cfg)

    if frame.is_empty():
        return _ensure_columns(frame, required + ["product_item_rollup", "product_goal"])

    enriched = frame.with_columns(
        pl.col("Item_internalid").cast(pl.Utf8).str.strip_chars().alias("Item_internalid")
    )

    if product_mapping_df.is_empty():
        enriched = _ensure_columns(enriched, ["product_item_rollup", "product_goal"])
    else:
        enriched = enriched.join(
            product_mapping_df.rename(
                {
                    ITEM_ROLLUP_COLUMN: "product_item_rollup",
                    DIVISION_GOAL_COLUMN: "product_goal",
                }
            ),
            left_on="Item_internalid",
            right_on="product_internal_id",
            how="left",
            suffix="_prod",
        )
        enriched = enriched.drop(["product_internal_id", "product_internal_id_prod"], strict=False)
        enriched = _ensure_columns(enriched, ["product_item_rollup", "product_goal"])

    enriched = enriched.with_columns(
        pl.col("product_item_rollup").alias(ITEM_ROLLUP_COLUMN),
        pl.col("product_goal").alias(DIVISION_GOAL_COLUMN),
    )

    enriched = _ensure_columns(enriched, required)

    total_rows = int(enriched.height)
    rollup_non_null = int(enriched.select(pl.col(ITEM_ROLLUP_COLUMN).is_not_null().sum()).item()) if total_rows else 0
    goal_non_null = int(enriched.select(pl.col(DIVISION_GOAL_COLUMN).is_not_null().sum()).item()) if total_rows else 0
    distinct_rollups = int(enriched.select(pl.col(ITEM_ROLLUP_COLUMN).n_unique()).item()) if rollup_non_null else 0

    logger.info(
        "Division goal coverage: total=%s product_tags=%s missing=%s",
        total_rows,
        goal_non_null,
        total_rows - goal_non_null,
    )

    multi_goal_rollups = 0
    if not product_mapping_df.is_empty():
        multi_goal_rollups = (
            product_mapping_df.group_by(ITEM_ROLLUP_COLUMN, maintain_order=False)
            .agg(pl.col(DIVISION_GOAL_COLUMN).n_unique().alias("goal_count"))
            .filter(pl.col("goal_count") > 1)
            .height
        )
        if multi_goal_rollups > 0:
            logger.warning(
                "Detected %s sub-divisions with multiple Goal assignments; modal resolution applied.",
                multi_goal_rollups,
            )

    coverage_rows = [
        {"metric": "total_rows", "value": total_rows},
        {"metric": "item_rollup_non_null", "value": rollup_non_null},
        {"metric": "division_goal_non_null", "value": goal_non_null},
        {"metric": "missing_item_rollup", "value": total_rows - rollup_non_null},
        {"metric": "missing_division_goal", "value": total_rows - goal_non_null},
        {"metric": "distinct_item_rollups", "value": distinct_rollups},
        {"metric": "multi_goal_item_rollups", "value": multi_goal_rollups},
    ]
    if total_rows:
        coverage_rows.append(
            {"metric": "item_rollup_coverage_pct", "value": round(rollup_non_null / total_rows, 6)}
        )
        coverage_rows.append(
            {"metric": "division_goal_coverage_pct", "value": round(goal_non_null / total_rows, 6)}
        )

    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(coverage_rows).write_csv(OUTPUTS_DIR / "mapping_coverage.csv")
    except Exception as exc:
        logger.warning("Unable to write mapping coverage CSV: %s", exc)

    enriched = enriched.with_columns(
        pl.struct(
            [
                pl.col(DIVISION_GOAL_COLUMN),
                pl.col(ITEM_ROLLUP_COLUMN),
                pl.col("Division"),
            ]
        )
        .map_elements(
            lambda row: _resolve_canonical_division(
                row.get(DIVISION_GOAL_COLUMN),
                row.get(ITEM_ROLLUP_COLUMN),
                row.get("Division"),
            ),
            return_dtype=pl.Utf8,
        )
        .alias(DIVISION_CANONICAL_COLUMN)
    )

    for rollup_name in ("draftsight", "geomagic"):
        mask = (
            pl.col(ITEM_ROLLUP_COLUMN)
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.to_lowercase()
            == rollup_name
        )
        subset = enriched.filter(mask)
        if subset.is_empty():
            continue
        divisions = subset.select(pl.col(DIVISION_CANONICAL_COLUMN).unique())[
            DIVISION_CANONICAL_COLUMN
        ].to_list()
        logger.info(
            "Mapped %s rows for rollup `%s` to divisions %s",
            subset.height,
            rollup_name,
            divisions,
        )

    enriched = enriched.drop(
        [
            "product_item_rollup",
            "product_goal",
        ],
        strict=False,
    )

    missing_goal = int(
        enriched.select(pl.col(DIVISION_GOAL_COLUMN).is_null().sum()).item()
    )
    if missing_goal > 0:
        logger.info(
            "Division Goal missing for %s line items; defaulting to legacy division or UNKNOWN.",
            missing_goal,
        )

    return enriched


def _apply_behavior_config(
    frame: pl.DataFrame,
    behavior_cfg,
    *,
    revenue_column: str,
) -> pl.DataFrame:
    """Apply configured line-item behavior policies (filters/flags)."""

    if frame.is_empty() or behavior_cfg is None:
        return frame

    result = frame
    exclude_line_types = [
        _normalize_rollup_label(value)
        for value in getattr(behavior_cfg, "exclude_line_types", []) or []
        if value is not None
    ]
    exclude_line_types = [value for value in exclude_line_types if value]

    if exclude_line_types:
        if "Rev_type" in result.columns:
            before = result.height
            result = result.with_columns(
                pl.col("Rev_type")
                .map_elements(_normalize_rollup_label, return_dtype=pl.Utf8)
                .alias("_line_type_norm")
            )
            result = result.filter(~pl.col("_line_type_norm").is_in(exclude_line_types))
            filtered = before - result.height
            if filtered:
                logger.info(
                    "Excluded %s line items using Rev_type filters %s",
                    filtered,
                    sorted(exclude_line_types),
                )
            result = result.drop("_line_type_norm")
        else:
            logger.warning(
                "Configured exclude_line_types=%s but `Rev_type` column missing; no rows filtered.",
                exclude_line_types,
            )

    return_policy = getattr(behavior_cfg, "return_treatment", "net_amount") or "net_amount"
    if return_policy == "exclude_returns":
        if revenue_column in result.columns:
            before = result.height
            result = result.filter(pl.col(revenue_column).fill_null(0) >= 0)
            removed = before - result.height
            if removed:
                logger.info(
                    "Excluded %s return line items (return_treatment=exclude_returns).",
                    removed,
                )
        else:
            logger.warning(
                "Configured return_treatment=exclude_returns but `%s` column missing; no rows filtered.",
                revenue_column,
            )
    elif return_policy == "separate_flag":
        if revenue_column in result.columns:
            result = result.with_columns(
                pl.col(revenue_column)
                .fill_null(0)
                .cast(pl.Float64, strict=False)
                .lt(0)
                .alias("is_return_line")
            )
        else:
            logger.warning(
                "Configured return_treatment=separate_flag but `%s` column missing; defaulting flag to False.",
                revenue_column,
            )
            result = result.with_columns(pl.lit(False).alias("is_return_line"))
    elif return_policy != "net_amount":
        logger.warning("Unknown return_treatment `%s`; defaulting to net_amount.", return_policy)

    kit_policy = getattr(behavior_cfg, "kit_handling", "prefer_children") or "prefer_children"
    if kit_policy not in {"prefer_children", "include_parent"}:
        logger.warning("Unknown kit_handling `%s`; defaulting to prefer_children.", kit_policy)
        kit_policy = "prefer_children"

    if kit_policy == "prefer_children":
        kit_columns = [col for col in ("kit_parent_line", "is_kit_parent", "kit_parent") if col in result.columns]
        if kit_columns:
            indicator = kit_columns[0]
            before = result.height
            result = result.filter(~pl.col(indicator).fill_null(False).cast(pl.Boolean, strict=False))
            removed = before - result.height
            if removed:
                logger.info(
                    "Removed %s kit parent rows using column `%s` (kit_handling=prefer_children).",
                    removed,
                    indicator,
                )
        else:
            logger.info(
                "kit_handling=prefer_children configured but no kit indicator column found; retaining all rows."
            )

    return result


def build_fact_sales_line(engine=None, config_path: str | None = None, cfg=None) -> pl.DataFrame:
    """Return a Polars DataFrame representing the deduped sales line fact."""

    cfg = cfg or load_config(config_path)
    engine = engine or get_db_connection()

    line_cfg = getattr(getattr(cfg, "etl", object()), "line_items", None)
    sources_cfg = getattr(line_cfg, "sources", None)
    dedupe_cfg = getattr(line_cfg, "dedupe", None)
    behavior_cfg = getattr(line_cfg, "behavior", None)

    sales_detail_table = None
    if sources_cfg is not None:
        sales_detail_table = getattr(sources_cfg, "sales_detail", None)

    order_column = getattr(dedupe_cfg, "order_column", DEFAULT_KEY_COLUMNS[0]) if dedupe_cfg else DEFAULT_KEY_COLUMNS[0]
    item_column = getattr(dedupe_cfg, "item_column", DEFAULT_KEY_COLUMNS[1]) if dedupe_cfg else DEFAULT_KEY_COLUMNS[1]
    revenue_column = getattr(dedupe_cfg, "revenue_column", DEFAULT_KEY_COLUMNS[2]) if dedupe_cfg else DEFAULT_KEY_COLUMNS[2]
    cogs_column = getattr(dedupe_cfg, "cogs_column", DEFAULT_KEY_COLUMNS[3]) if dedupe_cfg else DEFAULT_KEY_COLUMNS[3]
    gp_column = getattr(dedupe_cfg, "gross_profit_column", DEFAULT_KEY_COLUMNS[4]) if dedupe_cfg else DEFAULT_KEY_COLUMNS[4]
    term_gp_column = getattr(dedupe_cfg, "term_gross_profit_column", DEFAULT_KEY_COLUMNS[5]) if dedupe_cfg else DEFAULT_KEY_COLUMNS[5]
    last_update_column = getattr(dedupe_cfg, "last_update_column", "last_update") if dedupe_cfg else "last_update"

    key_columns = [order_column, item_column, revenue_column, cogs_column, gp_column, term_gp_column]
    currency_columns = [revenue_column, cogs_column, gp_column, term_gp_column]

    frame = _read_sales_detail(engine, cfg, sales_detail_table)
    if frame.is_empty():
        return frame

    required_cols = list(KEEP_COLUMNS)
    for col in [
        order_column,
        item_column,
        revenue_column,
        cogs_column,
        gp_column,
        term_gp_column,
        "SalesOrder_Currency",
        "USD_CAD_Conversion_rate",
    ]:
        if col not in required_cols:
            required_cols.append(col)

    frame = _ensure_columns(frame, required_cols + [last_update_column])
    frame = _normalize_dates(frame, DEFAULT_DATE_COLUMNS)
    frame = _dedupe_snapshots(frame, key_columns, last_update_column)
    frame = _attach_division_metadata(frame, engine, cfg, sources_cfg)
    frame = _apply_behavior_config(frame, behavior_cfg, revenue_column=revenue_column)

    # Normalize and attach per-line quantity for downstream projection/metrics
    frame = normalize_line_quantity(frame, behavior_cfg)

    # Surface COGS and USD normalised amounts
    frame = frame.with_columns(pl.col(cogs_column).alias("COGS"))
    frame = _add_currency_columns(frame, currency_columns, "SalesOrder_Currency", "USD_CAD_Conversion_rate")

    # Manual adjustments (true/false) default to False for now
    frame = frame.with_columns(pl.lit(False).alias("manual_adjustment_flag"))

    select_columns = required_cols + [
        "COGS",
        *[f"{col}_usd" for col in currency_columns],
        "manual_adjustment_flag",
        ITEM_ROLLUP_COLUMN,
        DIVISION_GOAL_COLUMN,
        DIVISION_CANONICAL_COLUMN,
        "Line_Qty",
    ]

    if "is_return_line" in frame.columns and "is_return_line" not in select_columns:
        select_columns.append("is_return_line")

    missing = [col for col in select_columns if col not in frame.columns]
    if missing:
        frame = _ensure_columns(frame, missing)

    return frame.select(select_columns)


def summarise_sales_header(
    fact_sales_line: pl.DataFrame,
    *,
    group_columns: Sequence[str] | None = None,
    currency_columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Aggregate the line-item fact to a header-level view keyed by Sales_Order."""

    if fact_sales_line.is_empty():
        return fact_sales_line

    group_cols = [col for col in (group_columns or HEADER_GROUP_COLUMNS) if col in fact_sales_line.columns]
    numeric_cols = currency_columns or DEFAULT_CURRENCY_COLUMNS

    # Include COGS alias and USD-normalised columns by default
    sum_candidates: set[str] = set()
    for col in fact_sales_line.columns:
        dtype = fact_sales_line.schema[col]
        if col in group_cols or col == "Item_internalid":
            continue
        is_numeric = False
        numeric_checker = getattr(dtype, "is_numeric", None)
        if callable(numeric_checker):
            try:
                is_numeric = bool(numeric_checker())
            except Exception:
                is_numeric = False
        if not is_numeric:
            # Fallback for older Polars versions
            is_numeric = str(dtype).startswith(("Int", "UInt", "Float", "Decimal"))

        is_boolean = dtype == pl.Boolean

        if is_numeric or is_boolean:
            sum_candidates.add(col)
    for base in numeric_cols:
        sum_candidates.add(base)
        sum_candidates.add(f"{base}_usd")
    sum_candidates.add("COGS")

    aggs = [pl.col(col).sum().alias(col) for col in sorted(sum_candidates) if col in fact_sales_line.columns]
    result = fact_sales_line.group_by(group_cols, maintain_order=True).agg(aggs)
    if "Sales_Order" in result.columns:
        result = result.sort("Sales_Order")
    return result


__all__ = [
    "build_fact_sales_line",
    "summarise_sales_header",
    "get_rollup_goal_mappings",
    "KEEP_COLUMNS",
    "DEFAULT_KEY_COLUMNS",
    "normalize_line_quantity",
]
