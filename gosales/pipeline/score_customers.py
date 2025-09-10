#!/usr/bin/env python3
"""
Customer scoring pipeline that generates ICP scores and whitespace analysis for specific divisions.
"""
import polars as pl
import pandas as pd
import mlflow
import mlflow.sklearn
import json
from pathlib import Path
import joblib

from gosales.utils.db import get_db_connection, get_curated_connection, validate_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import MODELS_DIR, OUTPUTS_DIR
from gosales.utils.normalize import normalize_division
from gosales.utils.config import load_config
import numpy as np
from gosales.features.engine import create_feature_matrix
from gosales.pipeline.rank_whitespace import rank_whitespace, save_ranked_whitespace, RankInputs
from gosales.validation.deciles import emit_validation_artifacts
from gosales.validation.schema import validate_icp_scores_schema, validate_whitespace_schema, write_schema_report
from gosales.monitoring.drift import check_drift_and_emit_alerts
from gosales.utils.config import load_config

logger = get_logger(__name__)

class MissingModelMetadataError(Exception):
    pass

_DIM_CUSTOMER_CACHE: pd.DataFrame | None = None


def _get_dim_customer(engine) -> pd.DataFrame:
    """Read dim_customer once per run and cache in-process.

    Returns a DataFrame with at least [customer_id, customer_name] and
    customer_id coerced to string for safe joins.
    """
    global _DIM_CUSTOMER_CACHE
    if _DIM_CUSTOMER_CACHE is not None and isinstance(_DIM_CUSTOMER_CACHE, pd.DataFrame):
        return _DIM_CUSTOMER_CACHE
    try:
        df = pd.read_sql("select customer_id, customer_name from dim_customer", engine)
    except Exception:
        # Minimal fallback if dim_customer missing
        df = pd.DataFrame({"customer_id": pd.Series(dtype=str), "customer_name": pd.Series(dtype=str)})
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype(str)
    df = df.drop_duplicates(subset="customer_id")
    _DIM_CUSTOMER_CACHE = df
    return df

def discover_available_models(models_dir: Path | None = None) -> dict[str, Path]:
    """Discover available models under models_dir and key by exact metadata division.

    Falls back to folder name without transformation if metadata division missing.
    """
    root = models_dir or MODELS_DIR
    available: dict[str, Path] = {}
    for p in root.glob("*_model"):
        div = p.name.replace("_model", "")
        meta_path = p / "metadata.json"
        try:
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    meta_div = normalize_division(meta.get("division"))
                    if meta_div:
                        div = meta_div
        except Exception:
            pass
        available[div] = p
    return available
 
def _sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric float dtype; replace infs/NaNs with 0.0 for scoring."""
    Xc = X.copy()
    for col in Xc.columns:
        Xc[col] = pd.to_numeric(Xc[col], errors="coerce")
    Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
    return Xc.fillna(0.0).astype(float)


def _score_p_icp(model, X: pd.DataFrame) -> np.ndarray:
    """Predict calibrated probability after sanitizing features."""
    Xc = _sanitize_features(X)
    # Prefer predict_proba; fallback to decision_function if unavailable
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xc)[:, 1]
    if hasattr(model, "decision_function"):
        margins = model.decision_function(Xc)
        return 1.0 / (1.0 + np.exp(-margins))
    # Final fallback: predict() then cast to float
    preds = getattr(model, "predict", lambda Z: np.zeros(len(Z)))(Xc)
    return np.asarray(preds, dtype=float)

def score_customers_for_division(
    engine,
    division_name: str,
    model_path: Path,
    *,
    run_manifest: dict | None = None,
    cutoff_date: str | None = None,
    prediction_window_months: int | None = None,
):
    """Score all customers for a specific division using a trained ML model.

    Requires ``cutoff_date`` and ``prediction_window_months`` in the model's
    ``metadata.json``; raises ``MissingModelMetadataError`` if absent.
    """
    logger.info(f"Scoring customers for division: {division_name}")
    
    # Load model via joblib pickle
    pkl = model_path / "model.pkl"
    try:
        model = joblib.load(pkl)
        logger.info(f"Loaded joblib model from {pkl}")
    except Exception as e:
        logger.error(f"Failed to load model from {pkl}: {e}")
        return pl.DataFrame()
    
    # Get feature matrix for all customers for the specified division
    # Enforce presence of cutoff and window in metadata; if missing, fail fast
    meta_path = model_path / "metadata.json"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        logger.error(f"Missing or unreadable metadata.json at {meta_path}: {e}")
        if run_manifest is not None:
            run_manifest.setdefault("alerts", []).append({
                "division": division_name,
                "severity": "error",
                "code": "MISSING_METADATA",
                "message": f"metadata.json missing/unreadable for model at {model_path}",
            })
        raise MissingModelMetadataError(f"Missing metadata.json for {division_name}")

    cutoff = meta.get("cutoff_date")
    window_months = meta.get("prediction_window_months")
    if cutoff is None or window_months is None:
        msg = (
            f"Required metadata fields missing for {division_name}: cutoff_date or prediction_window_months"
        )
        logger.error(msg)
        if run_manifest is not None:
            run_manifest.setdefault("alerts", []).append({
                "division": division_name,
                "severity": "error",
                "code": "MISSING_METADATA_FIELDS",
                "message": msg,
            })
        raise MissingModelMetadataError(msg)
    try:
        window_months = int(window_months)
    except Exception:
        msg = f"Invalid prediction_window_months in metadata for {division_name}: {window_months}"
        logger.error(msg)
        if run_manifest is not None:
            run_manifest.setdefault("alerts", []).append({
                "division": division_name,
                "severity": "error",
                "code": "INVALID_METADATA_FIELD",
                "message": msg,
            })
        raise MissingModelMetadataError(msg)

    # Use exact division string from metadata if present
    div_from_meta = normalize_division(meta.get("division"))
    if div_from_meta:
        division_name = div_from_meta
    feature_matrix = create_feature_matrix(engine, division_name, cutoff, window_months)

    # Prevalence guardrail: if labels present and zero prevalence while training had positives, skip
    try:
        if "bought_in_division" in feature_matrix.columns:
            y_prev = feature_matrix["bought_in_division"].mean()
            trained_pos = 0
            try:
                trained_pos = int(((meta or {}).get("class_balance") or {}).get("positives", 0))
            except Exception:
                trained_pos = 0
            if float(y_prev) == 0.0 and trained_pos > 0:
                alert = {
                    "division": division_name,
                    "severity": "warning",
                    "code": "ZERO_PREVALENCE_UNEXPECTED",
                    "message": f"Prevalence is zero at cutoff {cutoff}, but training had {trained_pos} positives. Skipping division.",
                }
                # Instrumentation: show product_division uniques in window
                try:
                    import pandas as _pd
                    from dateutil.relativedelta import relativedelta as _rd
                    cutoff_dt = _pd.to_datetime(cutoff)
                    win_end = cutoff_dt + _rd(months=int(window_months))
                    window_df = _pd.read_sql(
                        "SELECT product_division, order_date FROM fact_transactions WHERE order_date > ? AND order_date <= ?",
                        engine,
                        params=[cutoff_dt.strftime('%Y-%m-%d'), win_end.strftime('%Y-%m-%d')],
                    )
                    top = (
                        window_df.assign(product_division=window_df['product_division'].astype(str))
                        .assign(pd_trim=window_df['product_division'].str.rstrip())
                        .value_counts(subset=['pd_trim'])
                        .sort_values(ascending=False)
                        .head(20)
                    )
                    logger.warning("Division string (repr/len): %r / %d", division_name, len(division_name))
                    try:
                        logger.warning("Top product_division in window (trimmed):\n%s", top.to_string())
                    except Exception:
                        logger.warning("Top product_division in window (trimmed) rows: %d", int(top.sum()))
                except Exception:
                    pass
                logger.warning(alert["message"])
                if run_manifest is not None:
                    run_manifest.setdefault("alerts", []).append(alert)
                return pl.DataFrame()
    except Exception:
        pass
    
    if feature_matrix.is_empty():
        logger.warning(f"No feature matrix for {division_name}")
        return pl.DataFrame()
    
    # Prepare features for scoring (must match training)
    X = feature_matrix.drop(["customer_id", "bought_in_division"]).to_pandas()
    # Align columns to training feature order using saved metadata or feature_list.json
    try:
        train_cols: list[str] = []
        meta_path = model_path / "metadata.json"
        feat_list_path = model_path / "feature_list.json"
        # Prefer explicit feature_list.json (canonical order); fallback to metadata
        if feat_list_path.exists():
            with open(feat_list_path, "r", encoding="utf-8") as f:
                try:
                    import json as _json
                    train_cols = list(_json.load(f) or [])
                except Exception:
                    train_cols = []
        if (not train_cols) and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                train_cols = list(meta.get("feature_names", []) or [])
        if train_cols:
            # Hard reindex guarantees exact shape/order and zero‑fills missing
            missing = [c for c in train_cols if c not in X.columns]
            extra = [c for c in X.columns if c not in train_cols]
            if missing or extra:
                logger.info(
                    "Feature alignment: %d missing, %d extra columns (will reindex)",
                    len(missing), len(extra)
                )
                try:
                    if missing:
                        logger.debug("Missing top20: %s", missing[:20])
                    if extra:
                        logger.debug("Extra top20: %s", extra[:20])
                except Exception:
                    pass
            X = X.reindex(columns=train_cols, fill_value=0.0)
    except Exception as e:
        # If metadata missing, proceed with current X but log
        logger.warning(f"Feature alignment skipped due to error: {e}")
    
    try:
        probabilities = _score_p_icp(model, X)

        # Build scores_df and carry select auxiliary features for ranker
        feature_matrix_pd = feature_matrix.to_pandas()
        scores_df = feature_matrix_pd[["customer_id", "bought_in_division"]].copy()
        scores_df['division_name'] = division_name
        scores_df['icp_score'] = probabilities
        # Optional EV and affinity signals
        aux_cols = [
            'rfm__all__gp_sum__12m',        # EV proxy
            'affinity__div__lift_topk__12m',# affinity aggregate (if present)
            'mb_lift_max',                  # primary basket-lift signal used by ranker
            'mb_lift_mean',                 # secondary (not required but useful)
            'total_gp_all_time',            # size proxy for segment weighting
            'total_transactions_all_time',  # size proxy for segment weighting
        ]
        for aux_col in aux_cols:
            if aux_col in feature_matrix_pd.columns and aux_col not in scores_df.columns:
                scores_df[aux_col] = pd.to_numeric(feature_matrix_pd[aux_col], errors='coerce').fillna(0.0)

        # Pass through ALS embedding columns so ranker can compute als_norm
        als_cols = [c for c in feature_matrix_pd.columns if str(c).startswith('als_f')]
        if als_cols:
            for c in als_cols:
                scores_df[c] = pd.to_numeric(feature_matrix_pd[c], errors='coerce').fillna(0.0)
        # Optional item2vec embeddings (fallback if ALS coverage low)
        i2v_cols = [c for c in feature_matrix_pd.columns if str(c).startswith('i2v_f')]
        if i2v_cols:
            for c in i2v_cols:
                scores_df[c] = pd.to_numeric(feature_matrix_pd[c], errors='coerce').fillna(0.0)

        # Segment columns (strings): copy as-is if present
        for seg_str in ['industry', 'industry_sub']:
            if seg_str in feature_matrix_pd.columns and seg_str not in scores_df.columns:
                try:
                    scores_df[seg_str] = feature_matrix_pd[seg_str].astype(str).fillna("")
                except Exception:
                    pass

        # Ownership flag for ALS centroid (last 12m div transactions)
        try:
            tx_div_col = 'rfm__div__tx_n__12m'
            if tx_div_col in feature_matrix_pd.columns:
                scores_df['owned_division_pre_cutoff'] = (pd.to_numeric(feature_matrix_pd[tx_div_col], errors='coerce').fillna(0.0) > 0).astype(int)
        except Exception:
            pass
        # Propagate scoring metadata for auditing
        scores_df['cutoff_date'] = cutoff
        scores_df['prediction_window_months'] = int(window_months)
        try:
            scores_df['calibration_method'] = meta.get('calibration_method')
            if run_manifest is not None:
                mv = run_manifest.get('git_sha') or run_manifest.get('run_id')
            else:
                mv = meta.get('trained_at')
            scores_df['model_version'] = mv
        except Exception:
            pass
        
        customer_names = _get_dim_customer(engine)
        scores_df["customer_id"] = scores_df["customer_id"].astype(str)
        customer_names["customer_id"] = customer_names["customer_id"].astype(str)
        scores_df = scores_df.merge(customer_names, on="customer_id", how="left")
        
        logger.info(f"Successfully scored {len(scores_df)} customers for {division_name}")
        return pl.from_pandas(scores_df)
        
    except Exception as e:
        logger.error(f"Failed to score customers for {division_name}: {e}")
        return pl.DataFrame()

def generate_whitespace_opportunities(engine):
    """Generate whitespace opportunities with a lightweight scoring heuristic.

    This heuristic blends normalized purchase frequency, recency and total
    gross profit to produce a ``whitespace_score`` in ``[0, 1]``.  Each
    feature is scaled to ``[0, 1]`` across all customers and combined using
    weights ``0.5, 0.3, 0.2`` respectively.
    """
    logger.info("Generating whitespace opportunities...")
    try:
        # Read transactions with graceful handling when order_date is missing
        tx_pd = pd.read_sql("SELECT * FROM fact_transactions", engine)
        if "order_date" in tx_pd.columns:
            tx_pd["order_date"] = pd.to_datetime(tx_pd["order_date"], errors="coerce")
        else:
            # Provide a neutral recency anchor if date not available
            tx_pd["order_date"] = pd.Timestamp("1970-01-01")
        transactions = pl.from_pandas(tx_pd)
        customers = pl.from_pandas(_get_dim_customer(engine))
        if "customer_id" in transactions.columns:
            transactions = transactions.with_columns(pl.col("customer_id").cast(pl.Utf8))
        if "customer_id" in customers.columns:
            customers = customers.with_columns(pl.col("customer_id").cast(pl.Utf8))

        customer_summary = (
            transactions
            .group_by("customer_id")
            .agg([
                pl.col("product_division").unique().alias("divisions_bought"),
                pl.len().alias("purchase_count"),
                pl.max("order_date").alias("last_purchase"),
                pl.sum("gross_profit").alias("total_gp"),
            ])
        )

        if customer_summary.is_empty():
            return pl.DataFrame()

        freq_max = max(customer_summary["purchase_count"].max(), 1)
        gp_max = max(customer_summary["total_gp"].max(), 1.0)
        ref_date = transactions["order_date"].max()
        min_date = transactions["order_date"].min()
        max_days = max((ref_date - min_date).days, 1)
        customer_summary = customer_summary.with_columns([
            (pl.col("purchase_count") / freq_max).alias("freq_norm"),
            (1 - ((pl.lit(ref_date) - pl.col("last_purchase")).dt.total_days() / max_days)).clip(0.0, 1.0).alias("recency_norm"),
            (pl.col("total_gp") / gp_max).alias("gp_norm"),
        ])

        # Only valid, non-empty divisions
        all_divisions = (
            transactions
            .filter(pl.col("product_division").is_not_null() & (pl.col("product_division").cast(pl.Utf8).str.strip_chars() != ""))
            .select("product_division").unique()["product_division"].to_list()
        )

        opportunities = []
        for row in customer_summary.iter_rows(named=True):
            not_bought = [div for div in all_divisions if div not in row["divisions_bought"]]
            base_score = 0.5 * row["freq_norm"] + 0.3 * row["recency_norm"] + 0.2 * row["gp_norm"]
            score = float(max(0.0, min(1.0, base_score)))
            for division in not_bought:
                opportunities.append({
                    "customer_id": row["customer_id"],
                    "whitespace_division": division,
                    "whitespace_score": score,
                    "reason": f"Customer has high engagement but has not bought from the {division} division.",
                })

        if not opportunities:
            return pl.DataFrame()

        whitespace_df = (
            pl.DataFrame(opportunities)
            .with_columns(pl.col("customer_id").cast(pl.Int64, strict=False))
            .join(customers, on="customer_id", how="left")
        )
        logger.info(f"Generated {len(whitespace_df)} whitespace opportunities")
        return whitespace_df

    except Exception as e:
        logger.error(f"Failed to generate whitespace opportunities: {e}")
        return pl.DataFrame()

def generate_scoring_outputs(
    engine,
    *,
    run_manifest: dict | None = None,
    cutoff_date: str | None = None,
    prediction_window_months: int | None = None,
):
    """Generate and save ICP scores and whitespace analysis.

    ``cutoff_date`` and ``prediction_window_months`` act as fallbacks when the
    model metadata is missing these fields.
    """
    logger.info("Starting customer scoring and whitespace analysis...")
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # Discover available models by folder convention *_model
    available_models = discover_available_models()
    # Filter to supported targets only (divisions minus excluded + logical models)
    try:
        from gosales.etl.sku_map import get_supported_models, division_set as _division_set
        exclude = {"Hardware", "Maintenance"}
        targets = sorted({d for d in _division_set() if d not in exclude} | set(get_supported_models()))
        available_models = {k: v for k, v in available_models.items() if k in targets}
        if not available_models:
            logger.warning("No supported models found for scoring after pruning legacy models.")
    except Exception as e:
        logger.warning(f"Could not prune legacy models: {e}")
    
    all_scores: list[pl.DataFrame] = []
    for division_name, model_path in available_models.items():
        if not model_path.exists():
            logger.warning(f"Model not found for {division_name}: {model_path}")
            continue
        try:
            scores = score_customers_for_division(
                engine,
                division_name,
                model_path,
                run_manifest=run_manifest,
                cutoff_date=cutoff_date,
                prediction_window_months=prediction_window_months,
            )
        except MissingModelMetadataError:
            # Already logged and alerted; skip this division
            continue
        if not scores.is_empty():
            if run_manifest is not None:
                run_manifest.setdefault("divisions_scored", []).append(division_name)
            all_scores.append(scores)
            
    def _align_score_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
        if not frames:
            return pl.DataFrame()
        # Union of columns across frames
        cols: list[str] = []
        seen = set()
        for df in frames:
            for c in df.columns:
                if c not in seen:
                    seen.add(c)
                    cols.append(c)
        # Normalize key columns' dtypes where present
        normed: list[pl.DataFrame] = []
        for df in frames:
            d = df
            if "customer_id" in d.columns:
                try:
                    d = d.with_columns(pl.col("customer_id").cast(pl.Utf8))
                except Exception:
                    pass
            if "division_name" in d.columns:
                try:
                    d = d.with_columns(pl.col("division_name").cast(pl.Utf8))
                except Exception:
                    pass
            if "icp_score" in d.columns:
                try:
                    d = d.with_columns(pl.col("icp_score").cast(pl.Float64))
                except Exception:
                    pass
            # Add any missing columns as Nulls
            missing = [c for c in cols if c not in d.columns]
            if missing:
                d = d.with_columns([pl.lit(None).alias(c) for c in missing])
            # Reorder to common column order
            d = d.select(cols)
            normed.append(d)
        return pl.concat(normed, how="vertical_relaxed")

    if all_scores:
        combined_scores = _align_score_frames(all_scores)
        # Whitespace-only wiring for divisions without models (e.g., Post_Processing)
        try:
            from gosales.etl.sku_map import division_set as _division_set
            need_pp = ("Post_Processing" in _division_set()) and ("Post_Processing" not in available_models)
            if need_pp:
                ws_df = generate_whitespace_opportunities(engine)
                if not ws_df.is_empty():
                    ws_pp = ws_df.filter(pl.col("whitespace_division") == "Post_Processing")
                    if not ws_pp.is_empty():
                        cols = [c for c in ["customer_id","whitespace_division","whitespace_score","customer_name"] if c in ws_pp.columns]
                        ws_min = ws_pp.select(cols)
                        # Rename to match ICP schema: division_name, icp_score
                        rename_map = {}
                        if "whitespace_division" in ws_min.columns:
                            ws_min = ws_min.rename({"whitespace_division": "division_name"})
                        if "whitespace_score" in ws_min.columns:
                            ws_min = ws_min.rename({"whitespace_score": "icp_score"})
                        # Cast types
                        if "customer_id" in ws_min.columns:
                            ws_min = ws_min.with_columns(pl.col("customer_id").cast(pl.Int64, strict=False))
                        combined_scores = pl.concat([combined_scores, ws_min], how="vertical")
                        logger.info("Added heuristic Post_Processing p_icp for whitespace (no model present)")
        except Exception as _e:
            logger.warning(f"Post_Processing whitespace backfill failed: {_e}")
        # Append run_id if available for provenance
        try:
            if run_manifest is not None and isinstance(run_manifest.get("run_id"), str):
                if "run_id" not in combined_scores.columns:
                    combined_scores = combined_scores.with_columns(pl.lit(run_manifest["run_id"]).alias("run_id"))
        except Exception:
            pass
        icp_scores_path = OUTPUTS_DIR / "icp_scores.csv"
        # Robust write: attempt primary path; on Windows lock, write a run_id-suffixed file
        try:
            combined_scores.write_csv(str(icp_scores_path))
        except OSError as e:
            try:
                # Fallback: write to a unique file (non-destructive) and log
                import datetime as _dt
                ts = _dt.datetime.utcnow().strftime('%Y%m%d%H%M%S')
                fallback = OUTPUTS_DIR / f"icp_scores_{ts}.csv"
                combined_scores.write_csv(str(fallback))
                logger.warning(f"icp_scores.csv locked or unavailable ({e}); wrote fallback file: {fallback}")
                icp_scores_path = fallback
            except Exception as ee:
                logger.error(f"Failed to write ICP scores due to file lock and fallback failed: {ee}")
                raise
        # Lightweight schema validation
        try:
            report = validate_icp_scores_schema(icp_scores_path)
            write_schema_report(report, OUTPUTS_DIR / "schema_icp_scores.json")
        except Exception:
            pass
        logger.info(f"Saved ICP scores for {len(combined_scores)} customer-division combinations to {icp_scores_path}")
    else:
        logger.warning("No models were available for scoring.")
    
    # Phase-4 ranker: replace legacy heuristic whitespace
    try:
        cutoff_tag = None
        try:
            if run_manifest is not None:
                cutoff_tag = str(run_manifest.get('cutoff', '')).replace('-', '') or None
        except Exception:
            cutoff_tag = None

        icp_path = OUTPUTS_DIR / "icp_scores.csv"
        if icp_path.exists():
            # Load only necessary columns for ranking to reduce memory
            try:
                use_cols = [
                    'division_name','customer_id','icp_score',
                    'rfm__all__gp_sum__12m','affinity__div__lift_topk__12m',
                    'bought_in_division'
                ]
                icp_df = pd.read_csv(icp_path, usecols=lambda c: c in use_cols)
            except Exception:
                icp_df = pd.read_csv(icp_path)
            ranked = rank_whitespace(RankInputs(scores=icp_df))
            # Attach run_id for schema contract if available
            try:
                if run_manifest is not None and isinstance(run_manifest.get('run_id'), str):
                    if 'run_id' not in ranked.columns:
                        ranked.insert(0, 'run_id', run_manifest['run_id'])
            except Exception:
                pass
            path = save_ranked_whitespace(ranked, cutoff_tag=cutoff_tag)
            logger.info(f"Saved Phase-4 ranked whitespace to {path}")

            # Shadow mode: emit legacy heuristic whitespace for comparison and report overlap metrics
            try:
                cfg = load_config()
                if bool(getattr(cfg.whitespace, 'shadow_mode', False)):
                    legacy = generate_whitespace_opportunities(engine)
                    if not legacy.is_empty():
                        legacy_pd = legacy.to_pandas()
                        legacy_pd.rename(columns={"whitespace_division": "division_name", "whitespace_score": "score"}, inplace=True)
                        # Standardize required columns for comparison
                        legacy_pd = legacy_pd[[c for c in ["customer_id", "division_name", "score", "customer_name"] if c in legacy_pd.columns]]
                        legacy_name = f"whitespace_legacy_{cutoff_tag}.csv" if cutoff_tag else "whitespace_legacy.csv"
                        legacy_path = OUTPUTS_DIR / legacy_name
                        legacy_pd.to_csv(legacy_path, index=False)
                        # Overlap metrics: Jaccard of top-N between champion and legacy
                        try:
                            topn = max(1, int(len(ranked) * 0.10))
                            champ_top = set(ranked.nlargest(topn, ["score","p_icp","customer_id"])['customer_id'].astype(int).tolist())
                            leg_top = set(legacy_pd.nlargest(topn, "score")['customer_id'].astype(int).tolist())
                            inter = len(champ_top & leg_top)
                            union = len(champ_top | leg_top)
                            jacc = float(inter) / max(1, union)
                            overlap = {"top_percent": 10, "intersection": int(inter), "union": int(union), "jaccard": jacc}
                            ov_name = f"whitespace_overlap_{cutoff_tag}.json" if cutoff_tag else "whitespace_overlap.json"
                            (OUTPUTS_DIR / ov_name).write_text(pd.Series(overlap).to_json(indent=2), encoding='utf-8')
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Shadow mode failed: {e}")

            # Challenger overlap (champion vs score_challenger) at top-10%
            try:
                if 'score_challenger' in ranked.columns and len(ranked) > 0:
                    topn = max(1, int(len(ranked) * 0.10))
                    champ_top = set(ranked.nlargest(topn, ["score","p_icp","customer_id"])['customer_id'].astype(int).tolist()) if 'score' in ranked.columns else set()
                    chall_top = set(ranked.nlargest(topn, ["score_challenger","p_icp","customer_id"])['customer_id'].astype(int).tolist())
                    inter = len(champ_top & chall_top)
                    union = len(champ_top | chall_top)
                    jacc = float(inter) / max(1, union)
                    overlap = {"top_percent": 10, "intersection": int(inter), "union": int(union), "jaccard": jacc}
                    ov_name = f"whitespace_challenger_overlap_{cutoff_tag}.json" if cutoff_tag else "whitespace_challenger_overlap.json"
                    (OUTPUTS_DIR / ov_name).write_text(pd.Series(overlap).to_json(indent=2), encoding='utf-8')
            except Exception as e:
                logger.warning(f"Failed challenger overlap export: {e}")

            # Additional Phase-4 artifacts: explanations, thresholds, metrics
            try:
                # Explanations export
                expl_cols = [c for c in ['customer_id','division_name','score','p_icp','p_icp_pct','lift_norm','als_norm','EV_norm','nba_reason'] if c in ranked.columns]
                if expl_cols:
                    expl_name = f"whitespace_explanations_{cutoff_tag}.csv" if cutoff_tag else "whitespace_explanations.csv"
                    ranked[expl_cols].to_csv(OUTPUTS_DIR / expl_name, index=False)

                # Thresholds for grid
                thresholds = []
                if len(ranked) > 0 and 'score' in ranked.columns:
                    scores_num = pd.to_numeric(ranked['score'], errors='coerce').dropna().values
                    for k in [5, 10, 20]:
                        thresholds.append({"mode": "top_percent", "k_percent": k, "threshold": None, "count": 0})
                    if scores_num.size > 0:
                        sort_cols = [c for c in ['score', 'p_icp', 'EV_norm', 'customer_id'] if c in ranked.columns]
                        for i, k in enumerate([5, 10, 20]):
                            kk = max(1, int(scores_num.size * (k / 100.0)))
                            pos = scores_num.size - kk
                            thr = float(np.partition(scores_num, pos)[pos])
                            count = int((pd.to_numeric(ranked['score'], errors='coerce') >= thr).sum())
                            thresholds[i]["threshold"] = thr
                            thresholds[i]["count"] = int(count)
                thr_name = f"thresholds_whitespace_{cutoff_tag}.csv" if cutoff_tag else "thresholds_whitespace.csv"
                pd.DataFrame(thresholds).to_csv(OUTPUTS_DIR / thr_name, index=False)

                # Metrics summary
                checksum = int(pd.util.hash_pandas_object(ranked[['customer_id','division_name','score']]).sum()) if len(ranked) else 0
                top10_n = max(1, int(len(ranked) * 0.10)) if len(ranked) > 0 else 0
                top10 = ranked.nlargest(top10_n, ['score','p_icp','customer_id']) if top10_n > 0 else ranked.head(0)
                shares = top10.groupby('division_name')['customer_id'].size().sort_values(ascending=False) if top10_n > 0 and 'division_name' in top10.columns else pd.Series(dtype=int)
                share_map = {str(k): float(v) / max(1, int(len(top10))) for k, v in shares.items()} if top10_n > 0 else {}
                metrics = {
                    "rows": int(len(ranked)),
                    "checksum": checksum,
                    "division_shares_top10pct": share_map,
                }
                met_name = f"whitespace_metrics_{cutoff_tag}.json" if cutoff_tag else "whitespace_metrics.json"
                (OUTPUTS_DIR / met_name).write_text(pd.Series(metrics).to_json(indent=2), encoding='utf-8')

                # Lightweight schema validation for whitespace
                try:
                    ws_report = validate_whitespace_schema(path)
                    ws_out = OUTPUTS_DIR / (f"schema_whitespace_{cutoff_tag}.json" if cutoff_tag else "schema_whitespace.json")
                    write_schema_report(ws_report, ws_out)
                except Exception:
                    pass

                # Capacity selection and bias/diversity sharing
                try:
                    cfg = load_config()
                    mode = str(cfg.whitespace.capacity_mode)
                    selected = ranked
                    sort_cols = [c for c in ['score', 'p_icp', 'EV_norm', 'customer_id'] if c in ranked.columns]
                    if mode == 'top_percent':
                        if len(ranked) > 0:
                            ksel = max(1, int(len(ranked) * (cfg.modeling.capacity_percent / 100.0)))
                            initial = ranked.nlargest(ksel, sort_cols).copy() if sort_cols else ranked.head(0)
                            # Capacity-aware rebalancing to enforce division max share when applicable
                            try:
                                max_share = float(getattr(getattr(cfg, 'whitespace', object()), 'bias_division_max_share_topN', 0.0))
                            except Exception:
                                max_share = 0.0
                            def _rebalance(df_all: pd.DataFrame, sel_n: int, max_div_share: float) -> pd.DataFrame:
                                if sel_n <= 0 or df_all.empty or max_div_share <= 0.0:
                                    return df_all.head(sel_n)
                                allowed = {d: int(sel_n * max_div_share) for d in df_all['division_name'].dropna().unique()}
                                taken = {d: 0 for d in allowed}
                                out_rows = []
                                for _, row in df_all.sort_values(sort_cols, ascending=False).iterrows():
                                    d = row.get('division_name')
                                    if len(out_rows) >= sel_n:
                                        break
                                    if d not in allowed:
                                        out_rows.append(row)
                                        continue
                                    if taken[d] < max(1, allowed[d]):
                                        out_rows.append(row)
                                        taken[d] += 1
                                # If we could not fill capacity due to strict caps, top up with next best regardless of cap
                                if len(out_rows) < sel_n:
                                    rem = sel_n - len(out_rows)
                                    pool = df_all.drop(pd.DataFrame(out_rows).index, errors='ignore') if out_rows else df_all
                                    top_up = pool.sort_values(sort_cols, ascending=False).head(rem)
                                    out_rows.extend(list(top_up.to_dict(orient='records')))
                                return pd.DataFrame(out_rows, columns=df_all.columns)
                            selected = _rebalance(initial if not initial.empty else ranked, ksel, max_share) if max_share > 0 else initial
                        else:
                            selected = ranked.head(0)
                    elif mode in ('per_rep', 'hybrid'):
                        # Fallback to top_percent until rep attribution/interleave available
                        ksel = max(1, int(len(ranked) * (cfg.modeling.capacity_percent / 100.0)))
                        selected = ranked.nlargest(ksel, sort_cols).copy() if len(ranked) and sort_cols else ranked.head(0)

                    sel_name = f"whitespace_selected_{cutoff_tag}.csv" if cutoff_tag else "whitespace_selected.csv"
                    selected.to_csv(OUTPUTS_DIR / sel_name, index=False)

                    # Capacity summary export (counts and shares per division)
                    try:
                        if len(selected) > 0 and 'division_name' in selected.columns:
                            cap = selected.groupby('division_name')['customer_id'].size().sort_values(ascending=False)
                            cap_df = cap.rename('selected_count').reset_index()
                            cap_df['selected_share'] = cap_df['selected_count'] / float(len(selected))
                            cap_name = f"capacity_summary_{cutoff_tag}.csv" if cutoff_tag else "capacity_summary.csv"
                            cap_df.to_csv(OUTPUTS_DIR / cap_name, index=False)
                    except Exception:
                        pass

                    share_series = selected.groupby('division_name')['customer_id'].size().sort_values(ascending=False) if len(selected) > 0 and 'division_name' in selected.columns else pd.Series(dtype=int)
                    total_sel = max(1, int(len(selected)))
                    share_map = {str(k): float(v) / total_sel for k, v in share_series.items()} if len(selected) > 0 else {}
                    threshold = float(cfg.whitespace.bias_division_max_share_topN)
                    max_share = max(share_map.values()) if share_map else 0.0
                    warn = {
                        "capacity_mode": mode,
                        "selected_rows": int(len(selected)),
                        "division_shares": share_map,
                        "threshold": threshold,
                        "warn": bool(max_share > threshold),
                    }
                    warn_name = f"bias_diversity_warnings_{cutoff_tag}.json" if cutoff_tag else "bias_diversity_warnings.json"
                    (OUTPUTS_DIR / warn_name).write_text(pd.Series(warn).to_json(indent=2), encoding='utf-8')
                except Exception as e:
                    logger.warning(f"Failed capacity/bias exports: {e}")
            except Exception as e:
                logger.warning(f"Failed to emit Phase-4 thresholds/metrics: {e}")

            # Emit validation artifacts from scores
            emit_validation_artifacts(icp_path, cutoff_tag=cutoff_tag)

            # Drift/alerts monitoring (basic): write alerts.json at top-level
            try:
                check_drift_and_emit_alerts(run_manifest)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Phase-4 ranker failed; skipping whitespace ranking: {e}")

    logger.info("Scoring pipeline completed successfully!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score customers across divisions")
    parser.add_argument(
        "--cutoff-date",
        dest="cutoff_date",
        help="Cutoff date to use when model metadata lacks it",
    )
    parser.add_argument(
        "--window-months",
        dest="window_months",
        type=int,
        help="Prediction window in months when metadata is missing",
    )
    args = parser.parse_args()

    # Prefer curated connection where fact tables exist; fallback to primary DB
    try:
        db_engine = get_curated_connection()
    except Exception:
        db_engine = get_db_connection()
    try:
        cfg = load_config()
        strict = bool(getattr(getattr(cfg, 'database', object()), 'strict_db', False))
    except Exception:
        strict = False
    if not validate_connection(db_engine):
        msg = "Primary database connection is unhealthy."
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    generate_scoring_outputs(
        db_engine,
        cutoff_date=args.cutoff_date,
        prediction_window_months=args.window_months,
    )
