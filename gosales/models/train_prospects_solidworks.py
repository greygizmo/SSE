"""Train the Solidworks prospects model using bespoke label/feature builders.

The Solidworks prospecting flow has specialized labeling and feature logic.
This module stitches those components together, fits the LightGBM classifier,
and serializes calibrated artifacts plus evaluation metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from gosales.labels.prospects_solidworks import (
    ProspectLabelConfig,
    build_labels,
    default_cutoffs,
)
from gosales.features.prospects_solidworks import (
    ProspectFeatureConfig,
    build_features,
)
from gosales.utils.logger import get_logger
from gosales.utils.paths import MODELS_DIR, OUTPUTS_DIR

logger = get_logger(__name__)


def _prepare_dataset(cutoffs: Sequence[str]) -> pd.DataFrame:
    label_cfg = ProspectLabelConfig(cutoff_dates=cutoffs)
    labels_pl = build_labels(label_cfg)
    feature_cfg = ProspectFeatureConfig(cutoff_dates=cutoffs)
    features_pl = build_features(feature_cfg)

    dataset = labels_pl.join(features_pl, on=["customer_id", "cutoff_date"], how="inner")
    drop_cols = [col for col in dataset.columns if col.startswith("ns_")]
    if drop_cols:
        dataset = dataset.drop(drop_cols)

    df = dataset.to_pandas()
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])
    return df



def _prepare_features(df: pd.DataFrame):
    exclude = {"customer_id", "cutoff_date", "horizon_end", "label", "first_cre_date"}
    feature_cols = [c for c in df.columns if c not in exclude]
    cat_cols = [c for c in feature_cols if c.startswith("cat_")]
    num_cols = [c for c in feature_cols if c.startswith("feat_")]

    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)
        df[col] = pd.Categorical(df[col])

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return feature_cols, cat_cols, num_cols


def _time_based_split(df: pd.DataFrame, valid_periods: int = 3):
    df = df.sort_values("cutoff_date")
    unique_cutoffs = df["cutoff_date"].drop_duplicates().sort_values()
    if len(unique_cutoffs) <= valid_periods:
        valid_cutoffs = unique_cutoffs
    else:
        valid_cutoffs = unique_cutoffs[-valid_periods:]
    train_df = df[~df["cutoff_date"].isin(valid_cutoffs)]
    valid_df = df[df["cutoff_date"].isin(valid_cutoffs)]
    if train_df.empty:
        raise ValueError("Training dataset is empty. Increase history or adjust split.")
    if valid_df.empty:
        raise ValueError("Validation dataset is empty. Increase history or adjust split.")
    return train_df, valid_df, valid_cutoffs.tolist()


def train_prospect_model(months_back: int = 24, horizon_months: int = 6) -> dict:
    cutoffs = default_cutoffs(months_back)
    logger.info("Building dataset for cutoffs: %s", cutoffs)
    df = _prepare_dataset(cutoffs)
    df = df[df["label"].notna()]

    feature_cols, cat_cols, num_cols = _prepare_features(df)

    train_df, valid_df, valid_cutoffs = _time_based_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int)
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["label"].astype(int)

    model = LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=400,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, categorical_feature=list(cat_cols))

    valid_pred = model.predict_proba(X_valid)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_valid, valid_pred)) if y_valid.nunique() > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_valid, valid_pred)),
        "brier": float(brier_score_loss(y_valid, valid_pred)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "valid_cutoffs": [d.strftime("%Y-%m-%d") for d in valid_cutoffs],
    }

    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X_valid, y_valid)
    calibrated_pred = calibrator.predict_proba(X_valid)[:, 1]
    metrics.update(
        {
            "calibrated_pr_auc": float(average_precision_score(y_valid, calibrated_pred)),
            "calibrated_brier": float(brier_score_loss(y_valid, calibrated_pred)),
        }
    )

    category_levels = {
        col: [str(cat) for cat in X_train[col].cat.categories]
        for col in cat_cols
    }

    model_dir = MODELS_DIR / "prospects" / "solidworks"
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "feature_cols": feature_cols,
        "categorical_cols": list(cat_cols),
        "numeric_cols": list(num_cols),
        "category_levels": category_levels,
        "cutoff_history": [c for c in cutoffs],
        "horizon_months": horizon_months,
    }

    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(calibrator, model_dir / "calibrator.pkl")
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance
    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }
    )
    importances.to_csv(model_dir / "feature_importance.csv", index=False)

    logger.info("Training complete. Metrics: %s", metrics)

    return metrics


if __name__ == "__main__":
    train_prospect_model()
