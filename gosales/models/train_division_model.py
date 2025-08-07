import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import shap
import pandas as pd
from gosales.utils.paths import OUTPUTS_DIR
import mlflow
import shutil
import click

from gosales.utils.db import get_db_connection
from gosales.features.engine import create_feature_matrix
from gosales.utils.logger import get_logger
from gosales.utils.paths import MODELS_DIR

logger = get_logger(__name__)

def train_division_model(engine, division_name: str, cutoff_date: str = "2024-12-31", prediction_window_months: int = 6):
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

    # Define features (X) and target (y)
    # The new target column is 'bought_in_division'
    X = feature_matrix.drop(["customer_id", "bought_in_division"]).to_pandas()
    y = feature_matrix["bought_in_division"].to_pandas()
    
    logger.info(f"Training with {len(X)} customers and {positive_examples} positive examples for {division_name} division.")
    
    # Stratify is important for imbalanced datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression model
    lr = LogisticRegression(max_iter=1000, solver='liblinear')
    lr.fit(X_train, y_train)

    # Train LightGBM model
    lgbm = LGBMClassifier(random_state=42)
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
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        feature_names = list(X.columns)
        if best_model_name == "LightGBM":
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X)
            # For binary classification, shap_values is a list [neg, pos]
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_matrix = shap_values[1]
            else:
                shap_matrix = shap_values
        else:
            # Linear model SHAP
            explainer = shap.LinearExplainer(best_model, X_train)
            shap_matrix = explainer.shap_values(X)

        shap_df = pd.DataFrame(shap_matrix, columns=feature_names)
        shap_df.insert(0, 'customer_id', feature_matrix['customer_id'].to_pandas().values)
        shap_df.to_csv(OUTPUTS_DIR / f"shap_values_{division_name.lower()}.csv", index=False)
        logger.info("Exported SHAP values for explainability.")
    except Exception as e:
        logger.warning(f"Failed to compute/export SHAP values: {e}")

@click.command()
@click.option('--division', default='Solidworks', help='The division to train a model for.')
def main(division):
    """Main function to run the training script from the command line."""
    db_engine = get_db_connection()
    train_division_model(db_engine, division)

if __name__ == "__main__":
    main()
