from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl

from sqlalchemy import text

from gosales.utils.db import get_db_connection, get_curated_connection
from gosales.utils.logger import get_logger
from gosales.utils.identifiers import normalize_identifier_series
from gosales.utils.paths import OUTPUTS_DIR

logger = get_logger(__name__)


DESIRED_CLEANED_COLUMNS: tuple[str, ...] = (
    "internalid",
    "entityid",
    "companyname",
    "parent",
    "Parent_Customer",
    "Parent_Child_Account",
    "custom_parent",
    "ShippingCity",
    "ShippingState",
    "ShippingZip",
    "ShippingCountry",
    "Territory_Name",
    "region",
    "CAD_Territory",
    "AM_Territory",
    "Standardized_Territory_Name",
    "datecreated",
    "lastmodifieddate",
    "entitystatus",
    "entityStatus_Value",
    "stage",
    "Stage_value",
    "creditlimit",
    "overduebalance",
    "unbilledorders",
    "taxable",
    "Terms_Value",
    "email",
    "phone",
    "url",
    "weblead",
    "leadsource",
    "leadsource_name",
    "isinactive",
    "salesrep_Name",
    "am_sales_rep",
    "cam_sales_rep",
    "PDM_Sales_Rep",
    "Sim_Sales_Rep",
    "additive_rep",
    "Hardware_CSM",
    "Subscription_Service_Representative",
    "Subscription_Service_Rep",
    "customer_success_rep",
    "Known_Competitor",
    "Other_VAR",
    "othervar",
    "CAD_Named_Account",
    "SIM_Named_Account",
    "AM_Named_Account",
    "Electrical_Named_Account",
    "Current_CAM_Software",
    "Current_Simulation_Software",
    "Current_PDM_Software",
    "Current_PLM_Software",
    "Current_ECAD_Software",
    "Current_ERP_MRP_Software",
    "Current_Tech_Pubs_Software",
    "Current_3D_Printer",
    "Current_3D_Print_Solution",
    "first_cre_date",
    "first_cpe_date",
    "first_hw_date",
    "first_3dx_date",
    "globalsubscriptionstatus",
    "Success_Plan_Level",
    "Success_Plan_Date",
    "Success_plan_used",
    "Success_plan_total",
    "Strategic_Account",
    "Strategic_Account_Level",
    "top_50",
    "milestone_3dxactivation",
    "milestone_aevisit",
    "milestone_customerportal",
    "milestone_kickoffcall",
    "milestone_successhubaccess",
    "milestone_successplan",
    "milestone_supportused",
)

DESIRED_CUSTOMER_ONLY_COLUMNS: tuple[str, ...] = (
    "internalid",
    "top_50",
    "Other_VAR",
    "othervar",
    "PLM_Sales_Rep",
    "ShippingCity",
    "ShippingState",
    "ShippingZip",
    "ShippingCountry",
)


ENRICHED_NS_COLUMNS: dict[str, str] = {
    "entityid": "ns_entityid",
    "companyname": "ns_companyname",
    "parent": "ns_parent",
    "Parent_Customer": "ns_parent_customer",
    "Parent_Child_Account": "ns_parent_child_account",
    "custom_parent": "ns_custom_parent",
    "ShippingCity": "ns_shipping_city",
    "ShippingState": "ns_shipping_state",
    "ShippingZip": "ns_shipping_zip",
    "ShippingCountry": "ns_shipping_country",
    "Territory_Name": "ns_territory_name",
    "region": "ns_region",
    "CAD_Territory": "ns_cad_territory",
    "AM_Territory": "ns_am_territory",
    "Standardized_Territory_Name": "ns_standardized_territory",
    "datecreated": "ns_date_created",
    "lastmodifieddate": "ns_last_modified",
    "entitystatus": "ns_entity_status",
    "entityStatus_Value": "ns_entity_status_value",
    "stage": "ns_stage",
    "Stage_value": "ns_stage_value",
    "creditlimit": "ns_credit_limit",
    "overduebalance": "ns_overdue_balance",
    "unbilledorders": "ns_unbilled_orders",
    "taxable": "ns_taxable",
    "Terms_Value": "ns_terms_value",
    "email": "ns_email",
    "phone": "ns_phone",
    "url": "ns_url",
    "weblead": "ns_weblead",
    "leadsource": "ns_lead_source",
    "leadsource_name": "ns_lead_source_name",
    "isinactive": "ns_is_inactive",
    "salesrep_Name": "ns_salesrep_name",
    "am_sales_rep": "ns_am_sales_rep",
    "cam_sales_rep": "ns_cam_sales_rep",
    "PDM_Sales_Rep": "ns_pdm_sales_rep",
    "Sim_Sales_Rep": "ns_sim_sales_rep",
    "additive_rep": "ns_additive_rep",
    "PLM_Sales_Rep": "ns_plm_sales_rep",
    "Hardware_CSM": "ns_hardware_csm",
    "Subscription_Service_Representative": "ns_subscription_service_rep",
    "Subscription_Service_Rep": "ns_subscription_service_rep_alt",
    "customer_success_rep": "ns_customer_success_rep",
    "Known_Competitor": "ns_known_competitor",
    "Other_VAR": "ns_other_var",
    "othervar": "ns_othervar_flag",
    "CAD_Named_Account": "ns_cad_named_account",
    "SIM_Named_Account": "ns_sim_named_account",
    "AM_Named_Account": "ns_am_named_account",
    "Electrical_Named_Account": "ns_electrical_named_account",
    "Current_CAM_Software": "ns_current_cam_software",
    "Current_Simulation_Software": "ns_current_simulation_software",
    "Current_PDM_Software": "ns_current_pdm_software",
    "Current_PLM_Software": "ns_current_plm_software",
    "Current_ECAD_Software": "ns_current_ecad_software",
    "Current_ERP_MRP_Software": "ns_current_erp_mrp_software",
    "Current_Tech_Pubs_Software": "ns_current_tech_pubs_software",
    "Current_3D_Printer": "ns_current_3d_printer",
    "Current_3D_Print_Solution": "ns_current_3d_print_solution",
    "globalsubscriptionstatus": "ns_global_subscription_status",
    "Success_Plan_Level": "ns_success_plan_level",
    "Success_Plan_Date": "ns_success_plan_date",
    "Success_plan_used": "ns_success_plan_used",
    "Success_plan_total": "ns_success_plan_total",
    "Strategic_Account": "ns_strategic_account",
    "Strategic_Account_Level": "ns_strategic_account_level",
    "top_50": "ns_top_50",
    "milestone_3dxactivation": "ns_milestone_3dx_activation",
    "milestone_aevisit": "ns_milestone_ae_visit",
    "milestone_customerportal": "ns_milestone_customer_portal",
    "milestone_kickoffcall": "ns_milestone_kickoff_call",
    "milestone_successhubaccess": "ns_milestone_success_hub_access",
    "milestone_successplan": "ns_milestone_success_plan",
    "milestone_supportused": "ns_milestone_support_used",
    "first_cre_date": "ns_first_cre_date",
    "first_cpe_date": "ns_first_cpe_date",
    "first_hw_date": "ns_first_hw_date",
    "first_3dx_date": "ns_first_3dx_date",
}


def _get_available_columns(view: str, engine) -> set[str]:
    tbl = view.split(".")[-1]
    sql = (
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME='" + tbl + "'"
    )
    try:
        df = pd.read_sql(sql, engine)
        return set(df["COLUMN_NAME"].str.strip())
    except Exception as exc:
        logger.warning("Failed to fetch column metadata for %s: %s", view, exc)
        return set()


def _read_view(view: str, columns: Iterable[str]) -> pd.DataFrame:
    src_engine = get_db_connection()
    available = _get_available_columns(view, src_engine)
    use_cols = [c for c in columns if c in available]
    if not use_cols:
        raise ValueError(f"No requested columns found in {view}")
    query = f"SELECT {', '.join(use_cols)} FROM {view}"
    logger.info("Reading %s (%d columns)", view, len(use_cols))
    try:
        data_iter = pd.read_sql_query(query, src_engine, chunksize=200_000)
        frames = [chunk for chunk in data_iter]
        if not frames:
            return pd.DataFrame(columns=use_cols)
        return pd.concat(frames, ignore_index=True)
    except Exception:
        return pd.read_sql(query, src_engine)


def build_dim_ns_customer(write: bool = True) -> pl.DataFrame:
    """Materialize dim_ns_customer by blending NetSuite customer views."""
    cleaned = _read_view("dbo.customer_cleaned_headers", DESIRED_CLEANED_COLUMNS)
    customer_only = _read_view("dbo.customer_customerOnly", DESIRED_CUSTOMER_ONLY_COLUMNS)

    logger.info(
        "Loaded %d rows from customer_cleaned_headers and %d from customer_customerOnly",
        len(cleaned),
        len(customer_only),
    )

    # Normalize identifiers
    cleaned["internalid"] = normalize_identifier_series(cleaned["internalid"])
    customer_only["internalid"] = normalize_identifier_series(customer_only["internalid"])

    # Drop entries without internalid
    cleaned = cleaned[cleaned["internalid"].notna()].copy()
    customer_only = customer_only[customer_only["internalid"].notna()].copy()

    # Merge supplemental columns
    merged = cleaned.merge(
        customer_only.drop_duplicates("internalid"),
        on="internalid",
        how="left",
        suffixes=("", "_supp"),
    )

    # Ensure canonical ordering
    rename_map = {
        col: alias
        for col, alias in ENRICHED_NS_COLUMNS.items()
        if col in merged.columns
    }
    if rename_map:
        merged.rename(columns=rename_map, inplace=True)

    # Derived indicators
    if "ns_first_cre_date" in merged.columns:
        merged["ns_has_cre_purchase"] = merged["ns_first_cre_date"].notna().astype("int8")
    if "ns_first_cpe_date" in merged.columns:
        merged["ns_has_cpe_purchase"] = merged["ns_first_cpe_date"].notna().astype("int8")
    if "ns_first_hw_date" in merged.columns:
        merged["ns_has_hw_purchase"] = merged["ns_first_hw_date"].notna().astype("int8")

    stage_series = merged.get("ns_stage_value", pd.Series(dtype=str)).fillna("")
    if stage_series.empty and "ns_stage" in merged.columns:
        stage_series = merged["ns_stage"].fillna("")
    status_series = merged.get("ns_entity_status_value", pd.Series(dtype=str)).fillna("")

    customer_mask = (
        stage_series.str.lower().str.contains("customer", na=False)
        | status_series.str.lower().str.contains("customer", na=False)
        | merged.get("ns_has_cre_purchase", pd.Series(0)).astype(bool)
    )
    merged["ns_account_type"] = np.where(customer_mask, "customer", "prospect")

    merged = merged.loc[:, sorted(merged.columns)]

    # Coerce numeric columns
    for col in [
        "creditlimit",
        "overduebalance",
        "unbilledorders",
        "employees",
        "companyrevenue",
        "revenue",
        "Success_plan_total",
        "Success_plan_used",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Coerce dates
    for col in [
        "datecreated",
        "lastmodifieddate",
        "Success_Plan_Date",
        "first_cre_date",
        "first_cpe_date",
        "first_hw_date",
        "first_3dx_date",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_datetime(merged[col], errors="coerce")

    dim = pl.from_pandas(merged, include_index=False)

    if write:
        curated = get_curated_connection()
        dim.write_database("dim_ns_customer", curated, if_table_exists="replace")
        logger.info("Wrote dim_ns_customer with %d rows", len(dim))
        try:
            create_dim_customer_enriched()
            logger.info("Refreshed dim_customer_enriched view")
        except Exception as exc:
            logger.warning(f"Failed to refresh dim_customer_enriched view: {exc}")
    return dim


def fact_assets_internalid_coverage() -> dict[str, int | float]:
    """Return coverage metrics for fact_assets internalid vs dim_ns_customer."""
    curated = get_curated_connection()
    assets = pl.read_database("SELECT customer_id, internalid FROM fact_assets", curated).to_pandas()
    dim = pl.read_database("SELECT internalid FROM dim_ns_customer", curated).to_pandas()

    assets['customer_id_norm'] = normalize_identifier_series(assets['customer_id'])
    dim['internalid'] = normalize_identifier_series(dim['internalid'])

    total = int(len(assets))
    missing_customer_id = int(assets['customer_id_norm'].isna().sum())
    assets_with_id = assets[assets['customer_id_norm'].notna()].copy()
    with_id_total = int(len(assets_with_id))

    dim_ids = set(dim['internalid'].dropna())
    matched = int(assets_with_id['customer_id_norm'].isin(dim_ids).sum())

    coverage = {
        'fact_assets_total': total,
        'fact_assets_missing_customer_id': missing_customer_id,
        'fact_assets_with_customer_id': with_id_total,
        'matched_to_dim_ns_customer': matched,
    }
    if total:
        coverage['match_rate'] = round(matched / total, 4)
    if with_id_total:
        coverage['match_rate_on_non_null_ids'] = round(matched / with_id_total, 4)

    try:
        (OUTPUTS_DIR / 'qa').mkdir(parents=True, exist_ok=True)
        pd.DataFrame([coverage]).to_csv(OUTPUTS_DIR / 'qa' / 'fact_assets_internalid_coverage.csv', index=False)
    except Exception as exc:
        logger.warning('Failed to persist coverage report: %s', exc)

    return coverage


def create_dim_customer_enriched() -> None:
    """Create/refresh a view that merges dim_customer with dim_ns_customer."""
    curated = get_curated_connection()

    selected_columns = []
    for source_col, alias in ENRICHED_NS_COLUMNS.items():
        selected_columns.append(f'ns."{source_col}" AS {alias}')

    select_clause = "dc.*"
    if selected_columns:
        select_clause = "dc.*,\n        " + ",\n        ".join(selected_columns)

    view_sql = f"""
CREATE VIEW dim_customer_enriched AS
SELECT
        {select_clause}
FROM dim_customer AS dc
LEFT JOIN dim_ns_customer AS ns
    ON ns.internalid = dc.customer_id;
"""

    with curated.connect() as conn:
        meta = conn.execute(text("PRAGMA table_info(dim_ns_customer)")).fetchall()
        available_cols = {row[1] for row in meta}

        base_columns = [
            f'ns."{col}"'
            for col in ENRICHED_NS_COLUMNS.values()
            if col in available_cols
        ]
        derived_columns = [
            f'ns."{col}"'
            for col in [
                "ns_has_cre_purchase",
                "ns_has_cpe_purchase",
                "ns_has_hw_purchase",
                "ns_account_type",
            ]
            if col in available_cols
        ]
        selected_columns = base_columns + derived_columns

        select_clause = "dc.*"
        if selected_columns:
            select_clause = "dc.*,\n        " + ",\n        ".join(selected_columns)

        view_sql = f"""
CREATE VIEW dim_customer_enriched AS
SELECT
        {select_clause}
FROM dim_customer AS dc
LEFT JOIN dim_ns_customer AS ns
    ON ns.internalid = dc.customer_id;
"""

        conn.execute(text("DROP VIEW IF EXISTS dim_customer_enriched"))
        conn.execute(text(view_sql))
        conn.commit()


if __name__ == "__main__":
    dim = build_dim_ns_customer(write=True)
    metrics = fact_assets_internalid_coverage()
    logger.info("dim_ns_customer materialized rows: %d", len(dim))
    logger.info("fact_assets coverage: %s", metrics)
    print("Coverage summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
