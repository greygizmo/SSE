from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError
from sqlalchemy.sql import sqltypes

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


def _map_sql_type_family(sql_type: sqltypes.TypeEngine) -> str | None:
    """Return a coarse type family for a SQLAlchemy column type."""

    if isinstance(sql_type, (sqltypes.Integer, sqltypes.BigInteger, sqltypes.SmallInteger)):
        return "integer"
    if isinstance(sql_type, (sqltypes.Numeric, sqltypes.DECIMAL)):
        return "decimal"
    if isinstance(sql_type, (sqltypes.Float, sqltypes.REAL)):
        return "float"
    if isinstance(sql_type, sqltypes.Boolean):
        return "boolean"
    if isinstance(sql_type, (sqltypes.DateTime, sqltypes.TIMESTAMP)):
        return "datetime"
    if isinstance(sql_type, (sqltypes.Date, sqltypes.TIME)):
        return "date"
    if isinstance(sql_type, sqltypes.LargeBinary):
        return "binary"
    if isinstance(sql_type, (sqltypes.String, sqltypes.Text, sqltypes.Enum)):
        return "string"
    return None



def _map_polars_dtype(dtype: pl.DataType, dialect_name: str) -> str:
    """Translate a Polars dtype into a SQL column definition string."""

    dialect = (dialect_name or "").lower()

    if dtype == pl.Boolean:
        return "BIT" if dialect == "mssql" else "INTEGER"
    if dtype.is_integer():
        return "BIGINT" if dialect == "mssql" else "INTEGER"
    if dtype.is_float():
        return "FLOAT" if dialect == "mssql" else "REAL"
    if dtype.is_numeric():
        precision = getattr(dtype, "precision", None) or 38
        scale = min(getattr(dtype, "scale", None) or 9, precision)
        precision = max(scale, min(precision, 38))
        return f"DECIMAL({precision},{scale})"
    if dtype == pl.Date:
        return "DATE"
    if dtype == pl.Datetime:
        return "DATETIME2" if dialect == "mssql" else "TIMESTAMP"
    if dtype == pl.Time:
        return "TIME"
    if dtype == pl.Utf8:
        return "NVARCHAR(MAX)" if dialect == "mssql" else "TEXT"
    if dtype == pl.Binary:
        return "VARBINARY(MAX)" if dialect == "mssql" else "BLOB"
    return "NVARCHAR(MAX)" if dialect == "mssql" else "TEXT"



def _desired_polars_dtype(sql_family: str) -> pl.DataType | None:
    if sql_family == "integer":
        return pl.Int64
    if sql_family in ("float", "decimal"):
        return pl.Float64
    if sql_family == "boolean":
        return pl.Boolean
    if sql_family == "datetime":
        return pl.Datetime
    if sql_family == "date":
        return pl.Date
    if sql_family == "string":
        return pl.Utf8
    if sql_family == "binary":
        return pl.Binary
    return None



def _align_scores_table_schema(df: pl.DataFrame, engine: Engine) -> pl.DataFrame:
    """Ensure the scores table has the columns required by the dataframe."""

    try:
        inspector = inspect(engine)
        existing_columns: list[Mapping[str, object]] = inspector.get_columns(SCORES_TABLE)
    except NoSuchTableError:
        return df
    except SQLAlchemyError as exc:
        logger.debug("Could not inspect %s: %s", SCORES_TABLE, exc)
        return df

    existing_by_name = {col["name"]: col for col in existing_columns}
    dialect_name = engine.dialect.name if engine.dialect else ""
    preparer = engine.dialect.identifier_preparer if engine.dialect else None

    df_schema = dict(df.schema)
    missing_columns: list[tuple[str, str]] = []
    for column_name, dtype in df_schema.items():
        if column_name not in existing_by_name:
            sql_definition = _map_polars_dtype(dtype, dialect_name)
            missing_columns.append((column_name, sql_definition))

    if missing_columns:
        quoted_table = preparer.quote(SCORES_TABLE) if preparer else SCORES_TABLE
        with engine.begin() as conn:
            for column_name, sql_definition in missing_columns:
                quoted_column = preparer.quote(column_name) if preparer else column_name
                conn.execute(
                    text(
                        f"ALTER TABLE {quoted_table} ADD COLUMN {quoted_column} {sql_definition}"
                    )
                )
                logger.info(
                    "Added column %s (%s) to %s", column_name, sql_definition, SCORES_TABLE
                )

        inspector = inspect(engine)
        existing_columns = inspector.get_columns(SCORES_TABLE)
        existing_by_name = {col["name"]: col for col in existing_columns}

    aligned_df = df
    for column_name, col_meta in existing_by_name.items():
        if column_name not in df_schema:
            continue
        sql_family = _map_sql_type_family(col_meta.get("type"))
        desired_dtype = _desired_polars_dtype(sql_family) if sql_family else None
        current_dtype = dict(aligned_df.schema)[column_name]
        if desired_dtype and current_dtype != desired_dtype:
            try:
                aligned_df = aligned_df.with_columns(pl.col(column_name).cast(desired_dtype))
            except Exception as exc:  # pragma: no cover - defensive casting
                msg = (
                    f"Unable to cast column '{column_name}' from {current_dtype} to {desired_dtype} "
                    f"for table {SCORES_TABLE}. Adjust model output or migrate the table."
                )
                raise RuntimeError(msg) from exc

    return aligned_df



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
        result_pl.head(0).write_database(SCORES_TABLE, curated, if_table_exists="fail")
    except Exception as exc:
        logger.debug("Table %s already exists or could not be created: %s", SCORES_TABLE, exc)
    result_pl_aligned = _align_scores_table_schema(result_pl, curated)
    result_pl_aligned.write_database(SCORES_TABLE, curated, if_table_exists="append")

    out_dir = OUTPUTS_DIR / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"solidworks_prospect_scores_{cutoff}.parquet"
    result_pl_aligned.write_parquet(out_path)
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
