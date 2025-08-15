from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR


logger = get_logger(__name__)


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    prec, rec, _ = skm.precision_recall_curve(y_true, y_score)
    return float(skm.auc(rec, prec))


def _lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k_percent: int) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(n * (k_percent / 100.0)))
    idx = np.argsort(-y_score)[:k]
    top_rate = float(np.mean(y_true[idx]))
    base = float(np.mean(y_true)) if np.mean(y_true) > 0 else 1e-9
    return top_rate / base


def validate_holdout(icp_scores_csv: str | Path, *, year_tag: str | None = None, gates: Dict[str, float] | None = None) -> Path:
    df = pd.read_csv(icp_scores_csv)
    df = df.dropna(subset=['icp_score'])
    # Default gates
    gates = gates or {"auc": 0.70, "lift_at_10": 2.0, "cal_mae": 0.10}
    results: List[Dict[str, float]] = []
    status_ok = True
    for div, g in df.groupby('division_name'):
        y = pd.to_numeric(g.get('bought_in_division', 0), errors='coerce').fillna(0).astype(int).to_numpy()
        p = pd.to_numeric(g['icp_score'], errors='coerce').fillna(0.0).to_numpy()
        if len(y) == 0:
            continue
        try:
            auc = float(skm.roc_auc_score(y, p))
        except Exception:
            auc = float('nan')
        try:
            # Calibration MAE via 10 quantile bins
            bins = pd.qcut(pd.Series(p), q=10, labels=False, duplicates='drop')
            cal = pd.DataFrame({"y": y, "p": p, "bin": bins})
            grp = cal.groupby('bin', observed=False).agg(mean_p=("p","mean"), frac_pos=("y","mean"), count=("y","size")).dropna()
            cal_mae = float((grp['mean_p'].sub(grp['frac_pos']).abs() * grp['count']).sum() / max(1, grp['count'].sum()))
        except Exception:
            cal_mae = float('nan')
        brier = float(np.mean((p - y) ** 2))
        lift10 = _lift_at_k(y, p, 10)
        res = {
            "division_name": div,
            "auc": auc,
            "pr_auc": _pr_auc(y, p),
            "brier": brier,
            "cal_mae": cal_mae,
            "lift_at_10": lift10,
        }
        results.append(res)
        # Gate checks (ignore NaNs)
        if not np.isnan(auc) and auc < gates["auc"]:
            status_ok = False
        if not np.isnan(lift10) and lift10 < gates["lift_at_10"]:
            status_ok = False
        if not np.isnan(cal_mae) and cal_mae > gates["cal_mae"]:
            status_ok = False

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / (f"validation_metrics_{year_tag}.json" if year_tag else "validation_metrics.json")
    out.write_text(json.dumps({"divisions": results, "gates": gates, "status": "ok" if status_ok else "fail"}, indent=2), encoding='utf-8')
    logger.info(f"Wrote validation metrics to {out}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(OUTPUTS_DIR / "icp_scores.csv"))
    ap.add_argument("--year", default=None)
    args = ap.parse_args()
    validate_holdout(args.scores, year_tag=args.year)

#!/usr/bin/env python3
"""
Validation pipeline that tests the trained model against 2025 YTD holdout data.
"""
import pandas as pd
import polars as pl
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sqlalchemy import text
import numpy as np

from gosales.utils.db import get_db_connection
from gosales.etl.load_csv import load_csv_to_db
from gosales.etl.cleaners import clean_currency_value
from gosales.etl.build_star import build_star_schema
from gosales.etl.sku_map import get_sku_mapping
from gosales.features.engine import create_feature_matrix
from gosales.utils.logger import get_logger
from gosales.utils.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from dateutil.relativedelta import relativedelta

logger = get_logger(__name__)

def validate_against_holdout():
    """
    Validates the trained Solidworks model against 2025 YTD holdout data.
    
    This function:
    1. Loads the 2025 YTD holdout data into a separate database table
    2. Rebuilds the star schema with all data (2023-2024 + 2025 YTD)
    3. Creates features using 2024-12-31 cutoff and tests against actual 2025 purchases
    4. Generates validation metrics and saves results
    """
    logger.info("Starting holdout validation against 2025 YTD data...")
    
    # Get database connection
    db_engine = get_db_connection()
    
    # --- Phase 1: Load 2025 YTD Holdout Data ---
    logger.info("--- Phase 1: Loading 2025 YTD holdout data ---")
    holdout_file_path = DATA_DIR / "holdout" / "Sales Log 2025 YTD.csv"
    
    if not holdout_file_path.exists():
        logger.error(f"Holdout file not found at {holdout_file_path}")
        return
    
    # Load holdout data into a separate table
    load_csv_to_db(str(holdout_file_path), "sales_log_2025_ytd", db_engine)
    
    # Combine the training data with holdout data for complete star schema
    logger.info("Combining training and holdout data...")
    with db_engine.connect() as connection:
        # Create a unified sales_log_combined table
        connection.execute(text("DROP TABLE IF EXISTS sales_log_combined;"))
        connection.execute(text("""
            CREATE TABLE sales_log_combined AS 
            SELECT * FROM sales_log
            UNION ALL
            SELECT * FROM sales_log_2025_ytd;
        """))
        connection.commit()
    
    # Rebuild star schema with all data
    logger.info("Rebuilding star schema with combined data...")
    # Temporarily replace the sales_log table name in build_star_schema
    original_transactions = pd.read_sql("SELECT * FROM fact_transactions", db_engine)
    
    # Build new fact_transactions with 2025 data included
    # We need to manually run the star schema build on the combined data
    combined_df = pd.read_sql("SELECT * FROM sales_log_combined", db_engine)
    # Alias analytics-style columns to canonical ones (aligns with build_star normalization)
    alias_pairs = [
        ("PDM", "EPDM_CAD_Editor"),
        ("PDM_Qty", "EPDM_CAD_Editor_Qty"),
        ("Supplies", "Consumables"),
    ]
    for target, source in alias_pairs:
        if target not in combined_df.columns and source in combined_df.columns:
            combined_df[target] = combined_df[source]
    # Coerce all object/floating id/date fields to proper types to avoid Arrow errors
    for col in combined_df.columns:
        if col in ("CustomerId",):
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce").astype("Int64")
        elif col in ("Rec Date",):
            combined_df[col] = pd.to_datetime(combined_df[col], errors="coerce")
        elif combined_df[col].dtype == object:
            # Ensure strings are proper str (no mixed float) for Arrow conversion
            combined_df[col] = combined_df[col].astype(str)
    sales_log_combined = pl.from_pandas(combined_df)
    
    # Use the same unpivot logic from build_star.py but on combined data
    logger.info("Unpivoting combined sales data...")
    
    # SKU mapping: use the central mapping to preserve all divisions
    sku_mapping = get_sku_mapping()
    
    all_transactions = []
    id_vars = ["CustomerId", "Rec Date", "Division"]
    
    for gp_col, details in sku_mapping.items():
        qty_col = details['qty_col']
        division = details['division']

        if gp_col in sales_log_combined.columns and qty_col in sales_log_combined.columns:
            melted_df = (
                sales_log_combined.lazy()
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
    
    if all_transactions:
        fact_transactions_combined = pl.concat(all_transactions, how="vertical_relaxed")
        
        # Clean the data (same logic as build_star.py)
        fact_transactions_pd = fact_transactions_combined.to_pandas()
        fact_transactions_pd['customer_id'] = pd.to_numeric(fact_transactions_pd['CustomerId'], errors='coerce')
        fact_transactions_pd['order_date'] = pd.to_datetime(fact_transactions_pd['Rec Date'])
        
        # Robust currency cleaner reused from ETL
        fact_transactions_pd['gross_profit'] = fact_transactions_pd['gross_profit'].apply(clean_currency_value)
        fact_transactions_pd['quantity'] = pd.to_numeric(fact_transactions_pd['quantity'], errors='coerce').fillna(0)
        
        # Filter meaningful transactions
        fact_transactions_pd = fact_transactions_pd[
            (fact_transactions_pd['gross_profit'] != 0) | (fact_transactions_pd['quantity'] != 0)
        ]
        
        fact_transactions_pd = fact_transactions_pd[[
            'customer_id', 'order_date', 'product_sku', 'product_division', 'gross_profit', 'quantity'
        ]]
        
        # Replace the fact_transactions table
        fact_transactions_combined_clean = pl.from_pandas(fact_transactions_pd)
        fact_transactions_combined_clean.write_database("fact_transactions", db_engine, if_table_exists="replace")
        logger.info(f"Created combined fact_transactions table with {len(fact_transactions_pd)} transactions")
    
    # --- Phase 2: Generate Features and Labels for Validation ---
    logger.info("--- Phase 2: Generating validation features and labels ---")
    
    # Create feature matrix with cutoff date 2024-12-31, predict 6 months into 2025
    feature_matrix = create_feature_matrix(db_engine, "Solidworks", cutoff_date="2024-12-31", prediction_window_months=6)
    
    if feature_matrix.is_empty():
        logger.error("Feature matrix is empty. Cannot validate.")
        return
    
    # --- Phase 3: Load Model and Predict ---
    logger.info("--- Phase 3: Loading model and generating predictions ---")
    
    model_path = MODELS_DIR / "solidworks_model"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    try:
        model = mlflow.sklearn.load_model(str(model_path))
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Prepare features for prediction
    feature_matrix_pd = feature_matrix.to_pandas()

    # Recompute true labels directly from raw combined sales log to avoid dependence on SKU unpivot
    try:
        # Derive labels from holdout table directly to avoid schema misalignment from UNION ALL
        raw_combined = pd.read_sql("SELECT * FROM sales_log_2025_ytd", db_engine)
        # Parse dates and define the validation window (6 months after 2024-12-31)
        raw_combined["Rec Date"] = pd.to_datetime(raw_combined.get("Rec Date"), errors="coerce")
        cutoff_dt = pd.to_datetime("2024-12-31")
        window_end = cutoff_dt + relativedelta(months=6)

        mask_window = (raw_combined["Rec Date"] > cutoff_dt) & (raw_combined["Rec Date"] <= window_end)
        mask_division = raw_combined.get("Division").astype(str).str.strip().str.casefold() == "solidworks"
        buyers_series = pd.to_numeric(
            raw_combined.loc[mask_window & mask_division, "CustomerId"], errors="coerce"
        ).dropna().astype("Int64").unique()
        logger.info(f"Holdout label derivation: found {len(buyers_series)} Solidworks buyers in 2025 H1 window.")
        labels_df = pd.DataFrame({"customer_id": buyers_series, "bought_in_division": 1})

        # Replace existing target with holdout-derived labels
        if "bought_in_division" in feature_matrix_pd.columns:
            feature_matrix_pd.drop(columns=["bought_in_division"], inplace=True)
        # Ensure compatible key dtype for merge
        feature_matrix_pd["customer_id"] = pd.to_numeric(feature_matrix_pd["customer_id"], errors="coerce").astype("Int64")
        feature_matrix_pd = feature_matrix_pd.merge(labels_df, on="customer_id", how="left")
        feature_matrix_pd["bought_in_division"] = feature_matrix_pd["bought_in_division"].fillna(0).astype(int)
    except Exception as e:
        logger.warning(f"Failed to recompute holdout labels from raw data; falling back to feature matrix labels: {e}")

    X = feature_matrix_pd.drop(["customer_id", "bought_in_division"], axis=1)
    y_true = feature_matrix_pd["bought_in_division"].astype(int)
    cust_ids = feature_matrix_pd["customer_id"].astype("Int64")

    # If some positive-label customers are missing from the feature matrix (new 2025 logos),
    # append zero-imputed rows so metrics reflect true prevalence
    try:
        if 'labels_df' in locals() and not labels_df.empty:
            labels_df["customer_id"] = pd.to_numeric(labels_df["customer_id"], errors="coerce").astype("Int64")
            present = set(cust_ids.dropna().tolist())
            missing_buyers = [cid for cid in labels_df["customer_id"].dropna().unique().tolist() if cid not in present]
            if missing_buyers:
                zeros = pd.DataFrame(0, index=range(len(missing_buyers)), columns=X.columns)
                X = pd.concat([X, zeros], ignore_index=True)
                y_true = pd.concat([y_true, pd.Series([1]*len(missing_buyers), name='bought_in_division')], ignore_index=True)
                cust_ids = pd.concat([cust_ids, pd.Series(missing_buyers, name='customer_id')], ignore_index=True)
                logger.info(f"Added {len(missing_buyers)} missing positive customers to evaluation set with zero-imputed features.")
    except Exception as e:
        logger.warning(f"Failed to append missing buyers to evaluation set: {e}")

    # Align evaluation features to training feature set using model metadata
    try:
        import json
        meta_path = model_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            train_cols = meta.get('feature_names', [])
            if train_cols:
                for col in train_cols:
                    if col not in X.columns:
                        X[col] = 0
                # Drop any extra columns and enforce order
                X = X[train_cols]
                logger.info(f"Aligned evaluation features to {len(train_cols)} training columns.")
    except Exception as e:
        logger.warning(f"Failed to align evaluation features to training metadata: {e}")
    
    # Generate predictions
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of class 1
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # --- Phase 4: Calculate Validation Metrics ---
    logger.info("--- Phase 4: Calculating validation metrics ---")
    
    # Basic metrics
    auc_score = roc_auc_score(y_true, y_pred_proba)
    logger.info(f"Validation AUC: {auc_score:.4f}")
    
    # Classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    logger.info(f"Validation Precision: {class_report['1']['precision']:.4f}")
    logger.info(f"Validation Recall: {class_report['1']['recall']:.4f}")
    logger.info(f"Validation F1-Score: {class_report['1']['f1-score']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    logger.info(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    # Calculate conversion rates by score deciles
    results_df = pd.DataFrame({
        'customer_id': cust_ids,
        'bought_in_division': y_true,
        'prediction_score': y_pred_proba,
    })
    results_df['decile'] = pd.qcut(results_df['prediction_score'], 10, labels=False, duplicates='drop') + 1
    
    decile_analysis = results_df.groupby('decile').agg({
        'bought_in_division': ['count', 'sum', 'mean'],
        'prediction_score': ['min', 'max', 'mean']
    }).round(4)
    
    logger.info("\\nDecile Analysis:")
    logger.info(decile_analysis.to_string())
    
    # --- Phase 5: Save Validation Results ---
    logger.info("--- Phase 5: Saving validation results ---")
    
    # Save detailed predictions
    validation_results = results_df[['customer_id', 'bought_in_division', 'prediction_score', 'decile']].copy()
    validation_results_path = OUTPUTS_DIR / "validation_results_2025.csv"
    validation_results.to_csv(validation_results_path, index=False)
    logger.info(f"Saved validation results to {validation_results_path}")
    
    # Save gains/deciles table
    try:
        gains_path = OUTPUTS_DIR / "validation_gains_2025.csv"
        decile_analysis.to_csv(gains_path)
        logger.info(f"Saved validation gains table to {gains_path}")
    except Exception as e:
        logger.warning(f"Failed to save validation gains table: {e}")

    # Save metrics summary
    metrics_summary = {
        'validation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cutoff_date': '2024-12-31',
        'prediction_window_months': 6,
        'total_customers': len(feature_matrix_pd),
        'actual_buyers': int(y_true.sum()),
        'conversion_rate': float(y_true.mean()),
        'auc_score': float(auc_score),
        'precision': float(class_report['1']['precision']),
        'recall': float(class_report['1']['recall']),
        'f1_score': float(class_report['1']['f1-score'])
    }
    
    metrics_path = OUTPUTS_DIR / "validation_metrics_2025.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved validation metrics to {metrics_path}")
    
    # Restore original curated fact_transactions for subsequent scoring runs
    try:
        if not original_transactions.empty:
            pl.from_pandas(original_transactions).write_database("fact_transactions", db_engine, if_table_exists="replace")
            logger.info("Restored original fact_transactions table after validation.")
    except Exception as e:
        logger.warning(f"Failed to restore original fact_transactions: {e}")

    logger.info("Holdout validation completed successfully!")
    return metrics_summary

if __name__ == "__main__":
    validate_against_holdout()
