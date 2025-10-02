from __future__ import annotations

from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR
from gosales.etl.sku_map import get_model_targets

logger = get_logger(__name__)

SUMMARY_FILENAME = "labels_summary.csv"
SUMMARY_KEY_COLUMNS = ["division", "cutoff_date", "window_months"]


def _append_summary(summary: pd.DataFrame, destination: Path) -> None:
    """Persist a one-row summary without clobbering prior divisions."""

    if destination.exists():
        try:
            existing = pd.read_csv(destination)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=summary.columns)

        if not existing.empty and "division" in existing.columns:
            # Guard against prior runs that accidentally appended headers as rows.
            existing = existing[existing["division"].astype(str) != "division"]

        combined = pd.concat([existing, summary], ignore_index=True)
        key_cols = [col for col in SUMMARY_KEY_COLUMNS if col in combined.columns]
        if key_cols:
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
        else:
            combined = combined.drop_duplicates()
    else:
        combined = summary.copy()

    combined = combined.reindex(columns=summary.columns, fill_value=pd.NA)
    combined.to_csv(destination, index=False)


def compute_label_audit(
    engine,
    division_name: str,
    cutoff_date: str,
    prediction_window_months: int,
) -> pd.DataFrame:
    """Compute leakage-safe label prevalence and cohort sizes for a division.

    Creates a single-row summary with window dates, total customers, positives, and prevalence,
    and writes artifacts to the outputs directory.

    Args:
        engine: SQLAlchemy engine connected to the workspace database.
        division_name: Target division (e.g., 'Solidworks').
        cutoff_date: Feature cutoff date as YYYY-MM-DD (inclusive for features).
        prediction_window_months: Number of months in the future window for targets.

    Returns:
        pandas.DataFrame: One-row summary with audit metrics.
    """
    logger.info(
        f"Label audit for {division_name}: cutoff={cutoff_date}, window={prediction_window_months} months"
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load base tables
    tx = pd.read_sql("SELECT customer_id, order_date, product_division, product_sku FROM fact_transactions", engine)
    dim = pd.read_sql("SELECT customer_id FROM dim_customer", engine)
    if tx.empty or dim.empty:
        logger.warning("Missing transactions or customers for label audit; skipping.")
        return pd.DataFrame()

    # Parse dates and define windows
    tx["order_date"] = pd.to_datetime(tx["order_date"], errors="coerce")
    cutoff_dt = pd.to_datetime(cutoff_date)
    window_end = cutoff_dt + relativedelta(months=prediction_window_months)

    feature_tx = tx[tx["order_date"] <= cutoff_dt].copy()
    window_tx = tx[(tx["order_date"] > cutoff_dt) & (tx["order_date"] <= window_end)].copy()

    # Buyers in prediction window for target division or SKU-modeled target
    sku_targets = tuple(get_model_targets(division_name))
    if sku_targets:
        window_div = window_tx[window_tx["product_sku"].astype(str).isin(sku_targets)].copy()
    else:
        window_div = window_tx[window_tx["product_division"] == division_name].copy()
    buyers = window_div["customer_id"].dropna().unique()
    total_customers = int(dim["customer_id"].nunique())
    positives = int(len(buyers))
    prevalence = round(positives / total_customers, 6) if total_customers else 0.0

    # Customers with any feature-period activity (cohort)
    cohort_customers = int(feature_tx["customer_id"].nunique())

    # Cohort flags for positives
    # Treat IDs as strings (GUID-safe) for cohort flags
    feature_any = set(feature_tx["customer_id"].dropna().astype(str).tolist())
    if sku_targets:
        pre_div_buyers = set(
            feature_tx.loc[feature_tx["product_sku"].astype(str).isin(sku_targets), "customer_id"].dropna().astype(str).tolist()
        )
    else:
        pre_div_buyers = set(
            feature_tx.loc[feature_tx["product_division"] == division_name, "customer_id"].dropna().astype(str).tolist()
        )
    positive_rows = []
    for cid in buyers:
        cid_str = str(cid)
        is_new_logo = int(cid_str not in feature_any)
        had_pre_div = cid_str in pre_div_buyers
        is_renewal_like = int((not is_new_logo) and had_pre_div)
        is_expansion = int((not is_new_logo) and (not had_pre_div))
        positive_rows.append(
            {
                "customer_id": cid_str,
                "is_new_logo": is_new_logo,
                "is_expansion": is_expansion,
                "is_renewal_like": is_renewal_like,
            }
        )
    cohorts_df = pd.DataFrame(positive_rows)
    if not cohorts_df.empty:
        cohorts_df.to_csv(OUTPUTS_DIR / f"labels_cohorts_{division_name.lower()}.csv", index=False)
        cohort_counts = cohorts_df[["is_new_logo", "is_expansion", "is_renewal_like"]].sum().rename("count").to_frame()
        cohort_counts.to_csv(OUTPUTS_DIR / f"labels_cohort_counts_{division_name.lower()}.csv")

    summary = pd.DataFrame(
        [
            {
                "division": division_name,
                "cutoff_date": cutoff_dt.date().isoformat(),
                "window_months": int(prediction_window_months),
                "window_start": (cutoff_dt + pd.Timedelta(days=1)).date().isoformat(),
                "window_end": window_end.date().isoformat(),
                "total_customers": total_customers,
                "feature_cohort_customers": cohort_customers,
                "positive_customers": positives,
                "prevalence": prevalence,
            }
        ]
    )

    # Persist artifacts
    try:
        _append_summary(summary, OUTPUTS_DIR / SUMMARY_FILENAME)
        pd.DataFrame({"customer_id": buyers}).to_csv(
            OUTPUTS_DIR / f"labels_positive_{division_name.lower()}.csv", index=False
        )
        logger.info(
            "Appended label audit summary for %s (prevalence=%.4f, positives=%s/%s)",
            division_name,
            prevalence,
            positives,
            total_customers,
        )
    except Exception as e:
        logger.warning(f"Failed to persist label audit artifacts: {e}")

    return summary


