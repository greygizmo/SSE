import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False
import pandas as pd
from gosales.utils.paths import OUTPUTS_DIR
import mlflow
import shutil
import click

from gosales.utils.db import get_db_connection
from gosales.features.engine import create_feature_matrix
from gosales.utils.logger import get_logger
from gosales.utils.paths import MODELS_DIR
from gosales.models.metrics import calibration_bins as _calibration_bins_helper, calibration_mae as _calibration_mae_helper
from gosales.utils.config import load_config as _load_config
import json
from datetime import datetime

logger = get_logger(__name__)

def train_division_model(
    engine,
    division_name: str,
    cutoff_date: str = "2024-12-31",
    prediction_window_months: int = 6,
    shap_sample: int = 0,
    shap_max_rows: int | None = None,
):
    """
    Trains a model to predict which customers are likely to buy from a specific division.

    This function orchestrates the model training process:
    1.  Calls the feature engine to create a feature matrix for the given division.
    2.  Splits the data into training and testing sets.
    3.  Trains two models: Logistic Regression (baseline) and LightGBM.
    4.  Compares their performance using the AUC metric.
    5.  Saves the best-performing model to a dynamically named folder using MLflow.

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
        division_name (str): The name of the division to train a model for (e.g., 'Solidworks').
        cutoff_date (str): Date string (YYYY-MM-DD) to use as feature cutoff for time-based split.
        prediction_window_months (int): Number of months after cutoff_date to define prediction target.
    """
    logger.info(f"Starting model training for division: {division_name}...")
    logger.info(f"Training cutoff: {cutoff_date}, prediction window: {prediction_window_months} months")
    cfg = _load_config()
    if shap_max_rows is None:
        shap_max_rows = getattr(cfg.modeling, "shap_max_rows", 50000)

    # Create the feature matrix using the new engine with time-based split
    feature_matrix = create_feature_matrix(engine, division_name, cutoff_date, prediction_window_months)

    if feature_matrix.is_empty():
        logger.warning(f"Feature matrix for {division_name} is empty. Cannot train model.")
        return

    # Check for enough positive examples for training
    positive_examples = feature_matrix.filter(pl.col("bought_in_division") == 1).height
    if positive_examples < 10:  # Increased threshold for more stable training
        logger.warning(f"Not enough positive examples ({positive_examples}) for {division_name}. Cannot train model.")
        return

    # Define features (X) and target (y) + basic sanitation to avoid numeric issues
    X = feature_matrix.drop(["customer_id", "bought_in_division"]).to_pandas()
    # Sanitize features: coerce to numeric where possible, replace inf with NaN, then fill
    for col in X.columns:
        if not (pd.api.types.is_integer_dtype(X[col]) or pd.api.types.is_float_dtype(X[col])):
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    X = X.fillna(0.0)
    y = feature_matrix["bought_in_division"].to_pandas()
    
    logger.info(f"Training with {len(X)} customers and {positive_examples} positive examples for {division_name} division.")
    
    # Stratify is important for imbalanced datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute class weights
    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    scale_pos_weight = (negative_count / positive_count) if positive_count > 0 else 1.0

    # Train Logistic Regression model (with class weights)
    lr = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
    lr.fit(X_train, y_train)

    # Train LightGBM model with light class imbalance handling
    lgbm = LGBMClassifier(
        random_state=42,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
    )
    lgbm.fit(X_train, y_train)

    # Compare models using AUC score
    try:
        lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        lgbm_auc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])
    except ValueError as e:
        logger.error(f"Could not calculate AUC score, likely only one class present in y_test. Error: {e}")
        return

    best_model = lr if lr_auc > lgbm_auc else lgbm
    best_model_name = "Logistic Regression" if lr_auc > lgbm_auc else "LightGBM"
    best_auc = max(lr_auc, lgbm_auc)

    # --- Probability calibration ---
    try:
        from gosales.utils.config import load_config as _load_cfg
        _cfg = _load_cfg()
        _iso_thr = int(getattr(_cfg.modeling, 'sparse_isotonic_threshold_pos', 1000))
    except Exception:
        _iso_thr = 1000
    calibration_method = "isotonic" if positive_examples >= _iso_thr else "sigmoid"
    try:
        calibrator = CalibratedClassifierCV(estimator=best_model, method=calibration_method, cv=3)
        calibrator.fit(X_train, y_train)
        # Replace best_model with calibrated version
        best_model = calibrator
        # Recompute AUC on test set with calibrated probabilities
        try:
            calibrated_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
            best_auc = float(calibrated_auc)
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Calibration failed ({calibration_method}); proceeding without calibration: {e}")
        calibration_method = "none"

    # Save a simple model card CSV
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        model_card_rows = [
            {"metric": "auc_lr", "value": lr_auc},
            {"metric": "auc_lgbm", "value": lgbm_auc},
            {"metric": "best_model", "value": best_model_name},
            {"metric": "positives", "value": int(positive_examples)},
            {"metric": "total", "value": int(len(X))},
        ]
        try:
            # Compute calibration diagnostics on test split
            prob_test = best_model.predict_proba(X_test)[:, 1]
            bins_df = _calibration_bins_helper(y_test.to_numpy(), prob_test, n_bins=10)
            cal_mae = float(_calibration_mae_helper(bins_df, weighted=True))
            # Brier score: mean squared error of probabilities vs outcomes
            brier = float(((prob_test - y_test.to_numpy()) ** 2).mean())
            # Persist bins for analysis
            bins_df.to_csv(OUTPUTS_DIR / f"calibration_bins_{division_name.lower()}.csv", index=False)
            # Add to model card
            model_card_rows.append({"metric": "calibration_mae", "value": cal_mae})
            model_card_rows.append({"metric": "brier_score", "value": brier})
        except Exception as e:
            logger.warning(f"Failed to compute calibration diagnostics: {e}")

        pd.DataFrame(model_card_rows).to_csv(OUTPUTS_DIR / f"model_card_{division_name.lower()}.csv", index=False)
    except Exception:
        pass
    
    logger.info(f"Best model for {division_name} is {best_model_name} with AUC: {best_auc:.4f}")

    # --- Save the best model ---
    # The model path is now dynamic based on the division name
    model_path = MODELS_DIR / f"{division_name.lower()}_model"
    
    # Overwrite existing model if it exists
    if model_path.exists():
        logger.warning(f"Model path {model_path} already exists. Overwriting.")
        shutil.rmtree(model_path)
    
    mlflow.sklearn.save_model(best_model, str(model_path))
    logger.info(f"Successfully trained and saved {division_name} model to {model_path}")

    # --- SHAP Explainability export ---
    try:
        if _HAS_SHAP:
            if shap_sample <= 0:
                logger.warning("SHAP sample N is zero; skipping SHAP computation")
            elif len(X) > shap_max_rows:
                logger.warning(
                    "Skipping SHAP: dataset has %d rows exceeding threshold %d",
                    len(X),
                    shap_max_rows,
                )
            else:
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                feature_names = list(X.columns)
                rng = np.random.RandomState(cfg.modeling.seed)
                sample_n = min(shap_sample, len(X))
                sample_idx = rng.choice(len(X), size=sample_n, replace=False)
                X_sample = X.iloc[sample_idx]
                cust_ids = feature_matrix['customer_id'].to_pandas().iloc[sample_idx].values
                if best_model_name == "LightGBM":
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_sample)
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_matrix = shap_values[1]
                    else:
                        shap_matrix = shap_values
                else:
                    explainer = shap.LinearExplainer(best_model, X_sample)
                    shap_matrix = explainer.shap_values(X_sample)

                shap_df = pd.DataFrame(shap_matrix, columns=feature_names)
                shap_df.insert(0, 'customer_id', cust_ids)
                shap_df.to_csv(OUTPUTS_DIR / f"shap_values_{division_name.lower()}.csv", index=False)
                logger.info("Exported SHAP values for explainability.")
    except Exception as e:
        logger.warning(f"Failed to compute/export SHAP values: {e}")

    # --- Persist feature metadata for safe scoring alignment ---
    try:
        # Enrich metadata with calibration diagnostics if available
        cal_mae_val = None
        brier_val = None
        try:
            prob_test = best_model.predict_proba(X_test)[:, 1]
            bins_df = _calibration_bins_helper(y_test.to_numpy(), prob_test, n_bins=10)
            cal_mae_val = float(_calibration_mae_helper(bins_df, weighted=True))
            brier_val = float(((prob_test - y_test.to_numpy()) ** 2).mean())
        except Exception:
            pass

        metadata = {
            "division": division_name,
            "cutoff_date": cutoff_date,
            "prediction_window_months": int(prediction_window_months),
            "feature_names": list(X.columns),
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "best_model": best_model_name,
            "best_auc": float(best_auc),
            "calibration_method": calibration_method,
            "calibration_mae": cal_mae_val,
            "brier_score": brier_val,
            "class_balance": {
                "positives": positive_count,
                "negatives": negative_count,
                "scale_pos_weight": float(scale_pos_weight),
            },
        }
        with open(model_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved model feature metadata for scoring alignment.")
    except Exception as e:
        logger.warning(f"Failed to write model metadata: {e}")

    # --- Export calibration bins on test split (already covered above) ---
    try:
        prob_pos = best_model.predict_proba(X_test)[:, 1]
        bins_df = _calibration_bins_helper(y_test.to_numpy(), prob_pos, n_bins=10)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        bins_df.to_csv(OUTPUTS_DIR / f"calibration_bins_{division_name.lower()}.csv", index=False)
        logger.info("Exported calibration bins CSV.")
    except Exception as e:
        logger.warning(f"Failed to export calibration bins: {e}")

@click.command()
@click.option('--division', default='Solidworks', help='The division to train a model for.')
@click.option('--shap-sample', default=0, type=int, help='Rows to sample for SHAP; 0 disables')
def main(division, shap_sample):
    """Main function to run the training script from the command line."""
    db_engine = get_db_connection()
    train_division_model(db_engine, division, shap_sample=shap_sample)

if __name__ == "__main__":
    main()
