import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import mlflow
from gosales.utils.db import get_db_connection
from gosales.features.engine import create_feature_matrix
from gosales.utils.logger import get_logger
from gosales.utils.paths import MODELS_DIR

logger = get_logger(__name__)

def train_supplies_model(engine):
    """Trains a model to predict which customers are most likely to buy the Supplies product.

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
    """
    logger.info("Training Supplies model...")

    # Create the feature matrix
    feature_matrix = create_feature_matrix(engine, "Supplies")

    # If the feature matrix is empty, log a warning and return
    if feature_matrix.is_empty():
        logger.warning("Feature matrix is empty. Cannot train model.")
        return

    # Check if we have enough positive examples for training
    positive_examples = feature_matrix.filter(pl.col("bought_product") == 1).height
    if positive_examples < 2:
        logger.warning(f"Not enough positive examples ({positive_examples}). Cannot train model.")
        return

    # FIXED: Use proper binary target and feature columns
    # Features: customer behavior (exclude customer_id and target)
    X = feature_matrix.drop(["customer_id", "bought_product"]).to_pandas()
    y = feature_matrix["bought_product"].to_pandas()  # FIXED: Use binary target
    
    logger.info(f"Training with {len(X)} customers, {positive_examples} positive examples")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # stratify to maintain class balance
    )

    # Train the logistic regression model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Train the LightGBM model
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train)

    # Compare the models and save the best one
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    lgbm_auc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])

    if lr_auc > lgbm_auc:
        best_model = lr
        logger.info(f"Logistic regression is the best model with AUC: {lr_auc}")
    else:
        best_model = lgbm
        logger.info(f"LightGBM is the best model with AUC: {lgbm_auc}")

    # Save the best model
    mlflow.sklearn.save_model(best_model, MODELS_DIR / "supplies_model")
    logger.info("Successfully trained and saved Supplies model.")


if __name__ == "__main__":
    # Get database connection
    db_engine = get_db_connection()

    # Train the Supplies model
    train_supplies_model(db_engine)
