from __future__ import annotations

import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sqlalchemy import text

from gosales.features.prospects_solidworks import ProspectFeatureConfig, build_features
from gosales.labels.prospects_solidworks import default_cutoffs
from gosales.utils.db import get_curated_connection
from gosales.utils.logger import get_logger
from gosales.models.shap_utils import compute_shap_reasons
from gosales.utils.paths import OUTPUTS_DIR, MODELS_DIR

logger = get_logger(__name__)


MODEL_DIR = MODELS_DIR / "prospects" / "solidworks"
SCORES_TABLE = "scores_prospects_solidworks"


def _load_artifacts():
    model = joblib.load(MODEL_DIR / "model.pkl")
    calibrator_path = MODEL_DIR / "calibrator.pkl"
    calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    return model, calibrator, metadata


def _prepare_feature_frame(features: pl.DataFrame, metadata: dict) -> pd.DataFrame:
    df = features.to_pandas()
    feature_cols = metadata["feature_cols"]
    categorical_cols = metadata["categorical_cols"]
    numeric_cols = metadata["numeric_cols"]
    category_levels = metadata["category_levels"]

    for col in categorical_cols:
        levels = category_levels.get(col)
        df[col] = df[col].fillna("missing").astype(str)
        if levels:
            extras = [val for val in df[col].unique().tolist() if val not in levels]
            categories = list(levels) + extras
            df[col] = pd.Categorical(df[col], categories=categories)
        else:
            df[col] = pd.Categorical(df[col])

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, feature_cols, categorical_cols


def score(cutoff: str | None = None, top_k: int | None = None) -> pd.DataFrame:
    cutoff = cutoff or default_cutoffs(1)[-1]
    logger.info("Scoring SolidWorks prospects for cutoff %s", cutoff)

    features = build_features(ProspectFeatureConfig([cutoff]))
    model, calibrator, metadata = _load_artifacts()

    df, feature_cols, categorical_cols = _prepare_feature_frame(features, metadata)
    X = df[feature_cols]

    if calibrator is not None:
        probs = calibrator.predict_proba(X)[:, 1]
    else:
        probs = model.predict_proba(X)[:, 1]

    df["score"] = probs
    df["rank_global"] = df["score"].rank(method="first", ascending=False).astype(int)
    df["rank_territory"] = (
        df.groupby("cat_territory_standardized")["score"].rank(method="first", ascending=False).astype(int)
    )
    df["cutoff_date"] = pd.to_datetime(cutoff)

    if top_k is not None:
        df = df[df["rank_territory"] <= top_k]

    # SHAP reason codes on the scored rows
    try:
        reasons = compute_shap_reasons(model, df, feature_cols, top_k=3)
        df = pd.concat([df, reasons], axis=1)
    except Exception as exc:
        logger.warning("Skipping SHAP reasons: %s", exc)

    output_cols = [
        "customer_id",
        "cutoff_date",
        "score",
        "rank_global",
        "rank_territory",
        "cat_territory_standardized",
        "cat_territory_name",
        "cat_region",
        "feat_account_age_days",
        "feat_contact_score",
        "feat_has_weblead",
        "feat_has_email",
        "feat_has_phone",
        "feat_has_cpe_history",
        "feat_has_hw_history",
        "feat_has_3dx_history",
        "reason_1",
        "reason_2",
        "reason_3",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()

    curated = get_curated_connection()
    result_pl = pl.from_pandas(result)

    with curated.begin() as conn:
        try:
            conn.execute(
                text(f"DELETE FROM {SCORES_TABLE} WHERE cutoff_date = :cutoff"),
                {"cutoff": cutoff},
            )
        except Exception as exc:
            logger.info("Skipping delete for %s: %s", SCORES_TABLE, exc)
    # Ensure table exists with superset schema before append
    try:
        result_pl.head(0).write_database(SCORES_TABLE, curated, if_table_exists="replace")
    except Exception as exc:
        logger.info('Ensured table schema: %s', exc)
    result_pl.write_database(SCORES_TABLE, curated, if_table_exists="append")

    out_dir = OUTPUTS_DIR / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"solidworks_prospect_scores_{cutoff}.parquet"
    result_pl.write_parquet(out_path)
    logger.info("Wrote scores to %s and table %s", out_path, SCORES_TABLE)

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Score SolidWorks prospects")
    parser.add_argument("--cutoff", type=str, default=None, help="ISO cutoff date (default last month end)")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k per territory filter")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score(args.cutoff, args.top_k)
