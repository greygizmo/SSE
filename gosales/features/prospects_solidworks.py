from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from gosales.labels.prospects_solidworks import prospect_filter_expr
from gosales.utils.db import get_curated_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProspectFeatureConfig:
    cutoff_dates: Sequence[str]


def _load_feature_snapshot(engine=None) -> pl.DataFrame:
    engine = engine or get_curated_connection()
    query = """
        SELECT
            internalid AS customer_id,
            ns_account_type,
            ns_is_inactive,
            ns_stage,
            ns_stage_value,
            ns_entity_status_value,
            ns_standardized_territory,
            ns_territory_name,
            ns_region,
            ns_date_created,
            ns_last_modified,
            ns_email,
            ns_phone,
            ns_url,
            ns_weblead,
            ns_lead_source,
            ns_lead_source_name,
            ns_terms_value,
            ns_taxable,
            ns_known_competitor,
            ns_cad_named_account,
            ns_sim_named_account,
            ns_am_named_account,
            ns_electrical_named_account,
            ns_salesrep_name,
            ns_am_sales_rep,
            ns_cam_sales_rep,
            ns_additive_rep,
            ns_customer_success_rep,
            ns_first_cre_date,
            ns_first_cpe_date,
            ns_first_hw_date,
            ns_first_3dx_date
        FROM dim_ns_customer
    """
    date_cols = [
        "ns_date_created",
        "ns_last_modified",
        "ns_first_cre_date",
        "ns_first_cpe_date",
        "ns_first_hw_date",
        "ns_first_3dx_date",
    ]
    pdf = pd.read_sql(query, engine, parse_dates=date_cols)
    return pl.from_pandas(pdf)


def _presence(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("").astype(np.int8)


def _sanitize_enum(series: pd.Series) -> pd.Series:
    series = series.fillna("").astype(str)
    extracted = series.str.extract(r'value":"([^"}]+)', expand=False)
    return extracted.fillna(series)


def build_features(config: ProspectFeatureConfig, engine=None) -> pl.DataFrame:
    engine = engine or get_curated_connection()
    base = _load_feature_snapshot(engine)

    if not config.cutoff_dates:
        raise ValueError("cutoff_dates must not be empty")

    frames: list[pd.DataFrame] = []
    for cutoff_str in config.cutoff_dates:
        cutoff_ts = pd.Timestamp(cutoff_str)
        cohort = base.filter(prospect_filter_expr(cutoff_ts.to_pydatetime()))
        pdf = cohort.to_pandas()

        date_cols = [
            "ns_date_created",
            "ns_last_modified",
            "ns_first_cre_date",
            "ns_first_cpe_date",
            "ns_first_hw_date",
            "ns_first_3dx_date",
        ]
        for col in date_cols:
            if col in pdf.columns:
                pdf[col] = pd.to_datetime(pdf[col], errors="coerce")
                if pd.api.types.is_datetime64tz_dtype(pdf[col]):
                    pdf[col] = pdf[col].dt.tz_convert(None)

        pdf["cutoff_date"] = cutoff_ts.date()

        pdf["feat_account_age_days"] = (cutoff_ts - pdf["ns_date_created"]).dt.days
        pdf["feat_days_since_last_activity"] = (cutoff_ts - pdf["ns_last_modified"]).dt.days

        pdf["feat_has_email"] = _presence(pdf["ns_email"])
        pdf["feat_has_phone"] = _presence(pdf["ns_phone"])
        pdf["feat_has_url"] = _presence(pdf["ns_url"])
        pdf["feat_has_weblead"] = _presence(pdf["ns_weblead"])
        pdf["feat_contact_score"] = (
            pdf["feat_has_email"] + pdf["feat_has_phone"] + pdf["feat_has_url"]
        )

        pdf["feat_has_cpe_history"] = (
            pdf["ns_first_cpe_date"].notna() & (pdf["ns_first_cpe_date"] <= cutoff_ts)
        ).astype(np.int8)
        pdf["feat_has_hw_history"] = (
            pdf["ns_first_hw_date"].notna() & (pdf["ns_first_hw_date"] <= cutoff_ts)
        ).astype(np.int8)
        pdf["feat_has_3dx_history"] = (
            pdf["ns_first_3dx_date"].notna() & (pdf["ns_first_3dx_date"] <= cutoff_ts)
        ).astype(np.int8)

        pdf["feat_days_since_first_cpe"] = np.where(
            pdf["feat_has_cpe_history"].astype(bool),
            (cutoff_ts - pdf["ns_first_cpe_date"]).dt.days,
            np.nan,
        )
        pdf["feat_days_since_first_hw"] = np.where(
            pdf["feat_has_hw_history"].astype(bool),
            (cutoff_ts - pdf["ns_first_hw_date"]).dt.days,
            np.nan,
        )
        pdf["feat_days_since_first_3dx"] = np.where(
            pdf["feat_has_3dx_history"].astype(bool),
            (cutoff_ts - pdf["ns_first_3dx_date"]).dt.days,
            np.nan,
        )

        pdf["feat_taxable"] = pdf["ns_taxable"].fillna(0).astype(float).astype(np.int8)
        pdf["feat_known_competitor"] = pdf["ns_known_competitor"].notna().astype(np.int8)
        pdf["feat_named_account_cad"] = pdf["ns_cad_named_account"].fillna(0).astype(float).astype(np.int8)
        pdf["feat_named_account_sim"] = pdf["ns_sim_named_account"].fillna(0).astype(float).astype(np.int8)
        pdf["feat_named_account_am"] = pdf["ns_am_named_account"].fillna(0).astype(float).astype(np.int8)
        pdf["feat_named_account_electrical"] = (
            pdf["ns_electrical_named_account"].fillna(0).astype(float).astype(np.int8)
        )

        pdf["cat_lead_source"] = pdf["ns_lead_source_name"].fillna(pdf["ns_lead_source"])
        pdf["cat_terms_value"] = pdf["ns_terms_value"]
        pdf["cat_stage"] = _sanitize_enum(pdf["ns_stage_value"]).fillna(_sanitize_enum(pdf["ns_stage"]))
        pdf["cat_status"] = _sanitize_enum(pdf["ns_entity_status_value"])
        pdf["cat_territory_standardized"] = pdf["ns_standardized_territory"]
        pdf["cat_territory_name"] = pdf["ns_territory_name"]
        pdf["cat_region"] = pdf["ns_region"]

        feature_cols = [
            "customer_id",
            "cutoff_date",
            "feat_account_age_days",
            "feat_days_since_last_activity",
            "feat_has_email",
            "feat_has_phone",
            "feat_has_url",
            "feat_has_weblead",
            "feat_contact_score",
            "feat_has_cpe_history",
            "feat_has_hw_history",
            "feat_has_3dx_history",
            "feat_days_since_first_cpe",
            "feat_days_since_first_hw",
            "feat_days_since_first_3dx",
            "feat_taxable",
            "feat_known_competitor",
            "feat_named_account_cad",
            "feat_named_account_sim",
            "feat_named_account_am",
            "feat_named_account_electrical",
            "cat_lead_source",
            "cat_terms_value",
            "cat_stage",
            "cat_status",
            "cat_territory_standardized",
            "cat_territory_name",
            "cat_region",
        ]
        frames.append(pdf[feature_cols])

    combined = pd.concat(frames, ignore_index=True)
    result = pl.from_pandas(combined)

    out_dir = OUTPUTS_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solidworks_prospect_features.parquet"
    result.write_parquet(out_path)
    logger.info("Wrote SolidWorks prospect features to %s", out_path)

    return result


__all__ = ["ProspectFeatureConfig", "build_features"]
