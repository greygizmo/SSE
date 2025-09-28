#!/usr/bin/env python3
"""Generate customer propensity scores, whitespace lifts, and explanations.

This is the heart of the scoring release: it loads trained models, builds fresh
feature matrices, applies calibration, computes SHAP reasons, and assembles the
deliverables consumed by revenue operations for each division.  Running it keeps
our customer scorecards and whitespace reports current.
"""
import polars as pl
import pandas as pd
import mlflow
import mlflow.sklearn
import json
import os
import copy
from collections.abc import Iterable
from pathlib import Path
import joblib

from gosales.utils.db import get_db_connection, get_curated_connection, validate_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import MODELS_DIR, OUTPUTS_DIR
from gosales.models.shap_utils import compute_shap_reasons
from gosales.utils.normalize import normalize_division, normalize_model_key
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

    Prefers warm models (e.g., ``solidworks_model``) over cold-start folders (e.g.,
    ``solidworks_cold_model``). Falls back to folder name when metadata division is missing.
    """
    root = models_dir or MODELS_DIR
    primary: dict[str, Path] = {}
    cold: dict[str, Path] = {}
    for p in root.glob("*_model"):
        name = p.name
        is_cold = name.endswith("_cold_model")
        div = normalize_division(name.replace("_model", ""))
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
        if is_cold:
            cold[div] = p
        else:
            primary[div] = p
    # Prefer primary; fill gaps with cold
    available: dict[str, Path] = dict(primary)
    for d, path in cold.items():
        if d not in available:
            available[d] = path
    return available


def _filter_models_by_targets(
    available_models: dict[str, Path], targets: Iterable[str]
) -> dict[str, Path]:
    """Filter discovered models to only those that match supported targets.

    Matching is performed on a normalized form that is tolerant to case,
    underscores vs spaces, hyphens vs spaces, and extra whitespace.
    The returned mapping preserves the original discovered keys.
    """
    normalized_targets = {normalize_model_key(t) for t in targets if t}
    if not normalized_targets:
        return dict(available_models)

    # Group candidates by normalized key and de-duplicate with preference for non-cold models
    def _is_cold(path: Path) -> bool:
        return path.name.endswith("_cold_model")

    grouped: dict[str, list[tuple[str, Path]]] = {}
    unmatched: list[str] = []
    for key, path in available_models.items():
        nk = normalize_model_key(key)
        if nk in normalized_targets:
            grouped.setdefault(nk, []).append((key, path))
        else:
            unmatched.append(key)

    filtered: dict[str, Path] = {}
    dedup_pruned: list[str] = []
    for nk, items in grouped.items():
        # Choose best candidate: prefer non-cold; tie-break by key for determinism
        items_sorted = sorted(items, key=lambda it: (_is_cold(it[1]), it[0]))
        best_key, best_path = items_sorted[0]
        filtered[best_key] = best_path
        for other_key, _ in items_sorted[1:]:
            dedup_pruned.append(other_key)

    pruned = unmatched + dedup_pruned
    if pruned:
        # Debug aid to understand pruning behavior in ops
        sample = ", ".join(sorted(pruned)[:10])
        logger.debug(
            "Pruned %d model(s) during filtering (sample: %s)",
            len(pruned),
            sample,
        )
    return filtered


def _emit_capacity_and_logs(
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    cutoff_tag: str | None,
) -> pd.DataFrame:
    """Emit enriched capacity summary and division-level JSONL logs."""

    metadata = ranked.attrs if isinstance(ranked, pd.DataFrame) else {}
    elig_counts = metadata.get("eligibility_counts", {}) or {}
    per_div_counts = {}
    if isinstance(elig_counts, dict):
        per_div_counts = copy.deepcopy(elig_counts.get("per_division", {}) or {})
    coverage_meta = metadata.get("coverage", {}) or {}
    weight_meta = metadata.get("weight_adjustments", {}) or {}

    if not isinstance(per_div_counts, dict):
        per_div_counts = {}
    if not isinstance(coverage_meta, dict):
        coverage_meta = {}
    if not isinstance(weight_meta, dict):
        weight_meta = {}

    def _clean(obj):
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [_clean(v) for v in obj.tolist()]
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj

    if isinstance(selected, pd.DataFrame) and len(selected) and "division_name" in selected.columns:
        selected_counts = (
            selected.assign(division_name=selected["division_name"].astype(str))
            .groupby("division_name")
            ["customer_id"]
            .size()
        )
    else:
        selected_counts = pd.Series(dtype=int)

    total_selected = int(selected_counts.sum()) if len(selected_counts) else 0

    divisions = set(str(d) for d in selected_counts.index.tolist())
    divisions.update(str(d) for d in per_div_counts.keys())
    divisions = sorted(divisions)

    if divisions:
        cap_df = pd.DataFrame({"division_name": divisions})
        cap_df["selected_count"] = [int(selected_counts.get(div, 0)) for div in divisions]
        cap_df["selected_share"] = [
            (int(selected_counts.get(div, 0)) / float(total_selected)) if total_selected else 0.0
            for div in divisions
        ]

        elig_keys = [
            "start_rows",
            "kept_rows",
            "owned_excluded",
            "recent_contact_excluded",
            "open_deal_excluded",
            "region_mismatch_excluded",
        ]
        for key in elig_keys:
            cap_df[f"eligibility_{key}"] = [
                int((per_div_counts.get(div, {}) or {}).get(key, 0)) for div in divisions
            ]
        cap_df["eligibility_total_excluded"] = [
            int((per_div_counts.get(div, {}) or {}).get("start_rows", 0))
            - int((per_div_counts.get(div, {}) or {}).get("kept_rows", 0))
            for div in divisions
        ]

        if "segment" in getattr(selected, "columns", []):
            seg_counts = (
                selected.assign(
                    division_name=selected["division_name"].astype(str),
                    segment=selected["segment"].astype(str),
                )
                .groupby(["division_name", "segment"])
                ["customer_id"]
                .size()
                .unstack(fill_value=0)
            )
            for seg in seg_counts.columns:
                seg_key = str(seg).strip().lower().replace(" ", "_") or "unknown"
                col = f"segment_{seg_key}_selected"
                cap_df[col] = [int(seg_counts.loc[div, seg]) if div in seg_counts.index else 0 for div in divisions]
    else:
        cap_df = pd.DataFrame(
            columns=[
                "division_name",
                "selected_count",
                "selected_share",
                "eligibility_start_rows",
                "eligibility_kept_rows",
                "eligibility_owned_excluded",
                "eligibility_recent_contact_excluded",
                "eligibility_open_deal_excluded",
                "eligibility_region_mismatch_excluded",
                "eligibility_total_excluded",
            ]
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cap_name = f"capacity_summary_{cutoff_tag}.csv" if cutoff_tag else "capacity_summary.csv"
    cap_path = OUTPUTS_DIR / cap_name
    cap_df.to_csv(cap_path, index=False)

    log_name = f"whitespace_log_{cutoff_tag}.jsonl" if cutoff_tag else "whitespace_log.jsonl"
    log_path = OUTPUTS_DIR / log_name

    records = []
    empty_elig = {
        "start_rows": 0,
        "kept_rows": 0,
        "owned_excluded": 0,
        "recent_contact_excluded": 0,
        "open_deal_excluded": 0,
        "region_mismatch_excluded": 0,
    }

    coverage_payload = _clean(copy.deepcopy(coverage_meta))

    for _, row in cap_df.iterrows():
        div = str(row.get("division_name", ""))
        elig_payload = copy.deepcopy(per_div_counts.get(div, {}))
        if not isinstance(elig_payload, dict):
            elig_payload = {}
        merged_elig = {**empty_elig, **{k: int(v) for k, v in elig_payload.items()}}

        segments = {
            col[len("segment_") : -len("_selected")]: int(row[col])
            for col in cap_df.columns
            if col.startswith("segment_") and col.endswith("_selected")
        }

        seg_entries = []
        for entry in weight_meta.get("segments", []):
            entry_key = entry.get("segment_key", {}) or {}
            div_key = entry_key.get("division_name")
            if div_key is None or str(div_key) == div:
                seg_entries.append(copy.deepcopy(entry))

        weights_payload = {
            "base": _clean(copy.deepcopy(weight_meta.get("base", {}))),
            "global": _clean(copy.deepcopy(weight_meta.get("global", {}))),
            "segments": _clean(seg_entries),
            "segment_columns": _clean(copy.deepcopy(weight_meta.get("segment_columns", []))),
            "als_coverage_threshold": float(weight_meta.get("als_coverage_threshold", 0.0))
            if "als_coverage_threshold" in weight_meta
            else None,
        }

        record = {
            "division_name": div,
            "cutoff": cutoff_tag,
            "eligibility": _clean(merged_elig),
            "coverage": coverage_payload,
            "weights": weights_payload,
            "selection": {
                "selected_count": int(row.get("selected_count", 0)),
                "selected_share": float(row.get("selected_share", 0.0)),
                "segments": _clean(segments),
            },
        }
        records.append(record)

    with open(log_path, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return cap_df
 
def _load_feature_order(model_dir: Path, meta: dict | None = None) -> list[str]:
    """Return the canonical feature order for a trained model."""
    candidates: list[str] = []
    feat_list_path = model_dir / "feature_list.json"
    try:
        if feat_list_path.exists():
            with open(feat_list_path, "r", encoding="utf-8") as f:
                loaded = json.load(f) or []
                if isinstance(loaded, list):
                    candidates = [str(col) for col in loaded]
    except Exception:
        candidates = []
    if not candidates and meta:
        try:
            loaded = meta.get("feature_names") or []
            if isinstance(loaded, list):
                candidates = [str(col) for col in loaded]
        except Exception:
            candidates = []
    return candidates


def _estimate_dense_bytes(n_rows: int, n_features: int, dtype_size: int) -> int:
    if n_rows <= 0 or n_features <= 0:
        return 0
    return n_rows * n_features * dtype_size


def _should_use_batched_scoring(n_rows: int, n_features: int) -> bool:
    dtype_size = np.dtype(np.float32).itemsize
    approx_bytes = _estimate_dense_bytes(n_rows, n_features, dtype_size)
    return approx_bytes >= 800 * 1024 * 1024 or n_features >= 150_000 or n_rows >= 200_000


def _determine_batch_rows(
    n_rows: int,
    n_features: int,
    *,
    target_bytes: int | None = None,
) -> int:
    if n_rows <= 0:
        return 0
    if target_bytes is None:
        target_bytes = 160 * 1024 * 1024
        env_target = os.getenv("GOSALES_BATCH_TARGET_MB")
        if env_target:
            try:
                target_mb = max(32.0, float(env_target))
                target_bytes = int(target_mb * 1024 * 1024)
            except ValueError:
                logger.warning("Invalid GOSALES_BATCH_TARGET_MB=%s; using default 160MB", env_target)
    dtype_size = np.dtype(np.float32).itemsize
    if n_features <= 0:
        return max(1, min(n_rows, 5000))
    rows = max(1, target_bytes // max(1, n_features * dtype_size))
    return max(1, min(n_rows, rows))


def _prepare_customer_names(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    prepared = df.copy()
    if "customer_id" in prepared.columns:
        prepared["customer_id"] = prepared["customer_id"].astype(str)
    return prepared


def _compute_cold_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if df.empty:
        return mask
    if "rfm__all__tx_n__12m" in df.columns:
        tx12 = pd.to_numeric(df["rfm__all__tx_n__12m"], errors="coerce").fillna(0.0)
        mask |= tx12 <= 0
    als_cols = [c for c in df.columns if str(c).startswith("als_f")]
    if als_cols:
        als_strength = df[als_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs().sum(axis=1)
        mask |= als_strength <= 0
    return mask


def _assemble_scores_dataframe(
    feature_matrix_pd: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    division_name: str,
    cutoff: str,
    window_months: int,
    meta: dict,
    run_manifest: dict | None,
    customer_names: pd.DataFrame | None,
) -> pd.DataFrame:
    scores_df = feature_matrix_pd[["customer_id", "bought_in_division"]].copy()
    scores_df["division_name"] = division_name
    scores_df["icp_score"] = probabilities

    aux_cols = [
        "rfm__all__gp_sum__12m",
        "affinity__div__lift_topk__12m",
        "mb_lift_max",
        "mb_lift_mean",
        "total_gp_all_time",
        "total_transactions_all_time",
    ]
    for aux_col in aux_cols:
        if aux_col in feature_matrix_pd.columns and aux_col not in scores_df.columns:
            scores_df[aux_col] = pd.to_numeric(feature_matrix_pd[aux_col], errors="coerce").fillna(0.0)

    try:
        for base in ["mb_lift_max", "mb_lift_mean"]:
            if base not in scores_df.columns:
                candidates = [c for c in feature_matrix_pd.columns if str(c).startswith(base)]
                if candidates:
                    scores_df[base] = pd.to_numeric(feature_matrix_pd[candidates[0]], errors="coerce").fillna(0.0)
        aff_base = "affinity__div__lift_topk__12m"
        if aff_base not in scores_df.columns:
            aff_cands = [c for c in feature_matrix_pd.columns if str(c).startswith(aff_base)]
            if aff_cands:
                scores_df[aff_base] = pd.to_numeric(feature_matrix_pd[aff_cands[0]], errors="coerce").fillna(0.0)
    except Exception:
        pass

    als_cols = [c for c in feature_matrix_pd.columns if str(c).startswith("als_f")]
    if als_cols:
        try:
            scores_df[als_cols] = feature_matrix_pd[als_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        except Exception:
            for c in als_cols:
                scores_df[c] = pd.to_numeric(feature_matrix_pd[c], errors="coerce").fillna(0.0)

    i2v_cols = [c for c in feature_matrix_pd.columns if str(c).startswith("i2v_f")]
    if i2v_cols:
        try:
            scores_df[i2v_cols] = feature_matrix_pd[i2v_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        except Exception:
            for c in i2v_cols:
                scores_df[c] = pd.to_numeric(feature_matrix_pd[c], errors="coerce").fillna(0.0)

    for seg_str in ["industry", "industry_sub"]:
        if seg_str in feature_matrix_pd.columns and seg_str not in scores_df.columns:
            try:
                scores_df[seg_str] = feature_matrix_pd[seg_str].astype(str).fillna("")
            except Exception:
                pass

    try:
        tx_div_col = "rfm__div__tx_n__12m"
        if tx_div_col in feature_matrix_pd.columns:
            scores_df["owned_division_pre_cutoff"] = (
                pd.to_numeric(feature_matrix_pd[tx_div_col], errors="coerce").fillna(0.0) > 0
            ).astype(int)
    except Exception:
        pass

    scores_df["cutoff_date"] = cutoff
    scores_df["prediction_window_months"] = int(window_months)
    try:
        scores_df["calibration_method"] = meta.get("calibration_method")
        if run_manifest is not None:
            mv = run_manifest.get("git_sha") or run_manifest.get("run_id")
        else:
            mv = meta.get("trained_at")
        scores_df["model_version"] = mv
    except Exception:
        pass

    if customer_names is not None and not customer_names.empty:
        scores_df["customer_id"] = scores_df["customer_id"].astype(str)
        scores_df = scores_df.merge(customer_names, on="customer_id", how="left")

    return scores_df


def _sanitize_features(X: pd.DataFrame, *, dtype: np.dtype = np.float64) -> pd.DataFrame:
    """Ensure numeric float dtype; replace infs/NaNs with 0.0 for scoring."""
    Xc = X.copy()
    for col in Xc.columns:
        Xc[col] = pd.to_numeric(Xc[col], errors="coerce")
    Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
    return Xc.fillna(0.0).astype(dtype)


def _score_p_icp(model, X: pd.DataFrame, *, dtype: np.dtype = np.float64) -> np.ndarray:
    """Predict calibrated probability after sanitizing features."""
    Xc = _sanitize_features(X, dtype=dtype)
    # Prefer predict_proba; fallback to decision_function if unavailable
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xc)[:, 1]
    if hasattr(model, "decision_function"):
        margins = model.decision_function(Xc)
        return 1.0 / (1.0 + np.exp(-margins))
    # Final fallback: predict() then cast to float
    preds = getattr(model, "predict", lambda Z: np.zeros(len(Z)))(Xc)
    return np.asarray(preds, dtype=float)


def _load_cold_model(norm_key: str) -> tuple[object | None, dict | None, Path | None]:
    if not norm_key:
        return None, None, None
    cold_dir = MODELS_DIR / f"{norm_key}_cold_model"
    if not cold_dir.exists():
        return None, None, cold_dir
    cold_model = None
    cold_meta = None
    cold_pkl = cold_dir / "model.pkl"
    if cold_pkl.exists():
        try:
            cold_model = joblib.load(cold_pkl)
            logger.info(f"Loaded cold-start model from {cold_pkl}")
        except Exception as exc:
            logger.warning(f"Failed to load cold-start model from {cold_pkl}: {exc}")
    cold_meta_path = cold_dir / "metadata.json"
    if cold_meta_path.exists():
        try:
            with open(cold_meta_path, "r", encoding="utf-8") as f:
                cold_meta = json.load(f)
        except Exception as exc:
            logger.warning(f"Unable to read cold model metadata at {cold_meta_path}: {exc}")
    return cold_model, cold_meta, cold_dir


def _score_customers_batch(
    feature_matrix: pl.DataFrame,
    *,
    model,
    cold_model,
    division_name: str,
    cutoff: str,
    window_months: int,
    meta: dict,
    run_manifest: dict | None,
    train_cols: list[str],
    cold_train_cols: list[str],
    customer_names: pd.DataFrame | None,
) -> pl.DataFrame:
    n_rows = feature_matrix.height
    feature_cols_count = len(train_cols) if train_cols else max(len(feature_matrix.columns) - 2, 0)
    batch_rows = _determine_batch_rows(n_rows, feature_cols_count)
    logger.info("Using batch scoring for %s (rows=%d, features=%d, batch_rows=%d)", division_name, n_rows, feature_cols_count, batch_rows)
    frames: list[pd.DataFrame] = []
    idx = 0
    while idx < n_rows:
        length = min(batch_rows, n_rows - idx)
        part = feature_matrix.slice(idx, length)
        part_pd = part.to_pandas()
        feature_values = part_pd.drop(columns=["customer_id", "bought_in_division"], errors="ignore")
        aligned = feature_values
        if train_cols:
            aligned = feature_values.reindex(columns=train_cols, fill_value=0.0)
        aligned = aligned.astype(np.float32, copy=False)
        probs = _score_p_icp(model, aligned, dtype=np.float32)

        if cold_model is not None:
            cold_mask = _compute_cold_mask(part_pd)
            if cold_mask.any():
                cold_features = feature_values
                if cold_train_cols:
                    cold_features = cold_features.reindex(columns=cold_train_cols, fill_value=0.0)
                cold_features = cold_features.astype(np.float32, copy=False)
                cold_scores = _score_p_icp(cold_model, cold_features, dtype=np.float32)
                probs = np.asarray(probs, dtype=float)
                probs[cold_mask.values] = np.asarray(cold_scores)[cold_mask.values]

        chunk_scores = _assemble_scores_dataframe(
            part_pd,
            np.asarray(probs, dtype=float),
            division_name=division_name,
            cutoff=cutoff,
            window_months=window_months,
            meta=meta,
            run_manifest=run_manifest,
            customer_names=customer_names,
        )

        try:
            if len(aligned.columns) <= 5000:
                reasons = compute_shap_reasons(model, aligned, aligned.columns, top_k=3)
                for col in ["reason_1", "reason_2", "reason_3"]:
                    if col in reasons.columns:
                        chunk_scores[col] = reasons[col]
        except Exception:
            pass

        frames.append(chunk_scores)
        idx += length

    if not frames:
        return pl.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Successfully scored %d customers for %s (batch mode)", len(combined), division_name)
    return pl.from_pandas(combined)


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
    except Exception as exc:
        logger.error(f"Failed to load model from {pkl}: {exc}")
        return pl.DataFrame()

    # Load model metadata
    meta_path = model_path / "metadata.json"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        logger.error(f"Missing or unreadable metadata.json at {meta_path}: {exc}")
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

    div_from_meta = normalize_division(meta.get("division"))
    if div_from_meta:
        division_name = div_from_meta

    norm_key = normalize_division(division_name) or division_name
    norm_key_str = str(norm_key).lower() if norm_key else str(division_name).lower()

    cold_model, cold_meta, cold_dir = _load_cold_model(norm_key_str)
    cold_train_cols = _load_feature_order(cold_dir, cold_meta) if cold_dir is not None else []
    train_cols = _load_feature_order(model_path, meta)

    feature_matrix = create_feature_matrix(engine, division_name, cutoff, window_months)
    customer_names = _prepare_customer_names(_get_dim_customer(engine))

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

    feature_cols_count = len(train_cols) if train_cols else max(len(feature_matrix.columns) - 2, 0)
    if _should_use_batched_scoring(feature_matrix.height, feature_cols_count):
        return _score_customers_batch(
            feature_matrix,
            model=model,
            cold_model=cold_model,
            division_name=division_name,
            cutoff=cutoff,
            window_months=window_months,
            meta=meta,
            run_manifest=run_manifest,
            train_cols=train_cols,
            cold_train_cols=cold_train_cols,
            customer_names=customer_names,
        )

    try:
        X = feature_matrix.drop(["customer_id", "bought_in_division"]).to_pandas()
        if train_cols:
            missing = [c for c in train_cols if c not in X.columns]
            extra = [c for c in X.columns if c not in train_cols]
            if missing or extra:
                logger.info("Feature alignment: %d missing, %d extra columns (will reindex)", len(missing), len(extra))
                try:
                    if missing:
                        logger.debug("Missing top20: %s", missing[:20])
                    if extra:
                        logger.debug("Extra top20: %s", extra[:20])
                except Exception:
                    pass
            X = X.reindex(columns=train_cols, fill_value=0.0)
        X = _sanitize_features(X, dtype=np.float32)
        probabilities = _score_p_icp(model, X, dtype=np.float32)

        feature_matrix_pd = feature_matrix.to_pandas()
        cold_mask = _compute_cold_mask(feature_matrix_pd) if cold_model is not None else pd.Series(False, index=feature_matrix_pd.index)

        if cold_model is not None and cold_mask.any():
            cold_features = feature_matrix.drop(["customer_id", "bought_in_division"]).to_pandas()
            if cold_train_cols:
                cold_features = cold_features.reindex(columns=cold_train_cols, fill_value=0.0)
            cold_features = _sanitize_features(cold_features, dtype=np.float32)
            cold_scores = _score_p_icp(cold_model, cold_features, dtype=np.float32)
            probabilities = np.asarray(probabilities, dtype=float)
            probabilities[cold_mask.values] = np.asarray(cold_scores)[cold_mask.values]

        scores_df = _assemble_scores_dataframe(
            feature_matrix_pd,
            np.asarray(probabilities, dtype=float),
            division_name=division_name,
            cutoff=cutoff,
            window_months=window_months,
            meta=meta,
            run_manifest=run_manifest,
            customer_names=customer_names,
        )

        try:
            reasons = compute_shap_reasons(model, X, X.columns, top_k=3)
            reasons.index = scores_df.index
            for col in ["reason_1", "reason_2", "reason_3"]:
                if col not in scores_df.columns:
                    scores_df[col] = reasons[col]
        except Exception:
            pass

        logger.info(f"Successfully scored {len(scores_df)} customers for {division_name}")
        return pl.from_pandas(scores_df)
    except Exception as exc:
        logger.error(f"Failed to score customers for {division_name}: {exc}")
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
            .with_columns(pl.col("customer_id").cast(pl.Utf8, strict=False))
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
        available_models = _filter_models_by_targets(available_models, targets)
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
            known_divisions = {normalize_division(d) for d in _division_set()}
            available_keys = {normalize_division(name) for name in available_models}
            need_pp = (
                normalize_division("Post_Processing") in known_divisions
                and normalize_division("Post_Processing") not in available_keys
            )
            if need_pp:
                ws_df = generate_whitespace_opportunities(engine)
                if not ws_df.is_empty():
                    ws_pp = ws_df.filter(
                        pl.col("whitespace_division")
                        .cast(pl.Utf8)
                        .str.strip_chars()
                        .str.to_lowercase()
                        == normalize_division("Post_Processing")
                    )
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
        # Add per-division percentile and letter grade to core artifact
        try:
            import polars as _pl
            if {'division_name','icp_score'}.issubset(set(combined_scores.columns)):
                combined_scores = combined_scores.with_columns([
                    ((_pl.col('icp_score').rank('average').over('division_name') / _pl.len().over('division_name'))).alias('icp_percentile')
                ])
                combined_scores = combined_scores.with_columns([
                    _pl.when(_pl.col('icp_percentile') >= 0.90).then(_pl.lit('A'))
                       .when(_pl.col('icp_percentile') >= 0.70).then(_pl.lit('B'))
                       .when(_pl.col('icp_percentile') >= 0.40).then(_pl.lit('C'))
                       .when(_pl.col('icp_percentile') >= 0.20).then(_pl.lit('D'))
                       .otherwise(_pl.lit('F'))
                       .alias('icp_grade')
                ])
        except Exception as _e:
            logger.warning(f"Failed to compute ICP percentile/grade in-core: {_e}")

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
                    'division_name','customer_id','customer_name','icp_score',
                    'rfm__all__gp_sum__12m','affinity__div__lift_topk__12m',
                    'bought_in_division',
                    # Segment context
                    'rfm__all__tx_n__12m','assets_active_total','assets_on_subs_total'
                ]
                def _usecols(c: str) -> bool:
                    return (
                        (c in use_cols)
                        or c.startswith('als_f')
                        or c.startswith('mb_lift_')
                        or c.startswith('affinity__div__lift_topk__12m')
                        or c == 'owned_division_pre_cutoff'
                    )
                icp_df = pd.read_csv(icp_path, usecols=_usecols)
            except Exception:
                icp_df = pd.read_csv(icp_path)
            ranked = rank_whitespace(RankInputs(scores=icp_df))
            # Derive segment labels (warm/cold/prospect) and attach to ranked
            try:
                seg_df = icp_df.copy()
                warm = pd.to_numeric(seg_df.get('rfm__all__tx_n__12m', 0), errors='coerce').fillna(0.0) > 0
                has_assets = (
                    pd.to_numeric(seg_df.get('assets_active_total', 0), errors='coerce').fillna(0.0) > 0
                ) | (
                    pd.to_numeric(seg_df.get('assets_on_subs_total', 0), errors='coerce').fillna(0.0) > 0
                )
                cold = (~warm) & has_assets
                segment = pd.Series(np.where(warm, 'warm', np.where(cold, 'cold', 'prospect')), index=seg_df.index)
                seg_data = seg_df[['customer_id','division_name']].copy()
                seg_data['segment'] = segment.astype(str)
                ranked = ranked.merge(seg_data, on=['customer_id','division_name'], how='left')
            except Exception:
                pass
            # Attach run_id for schema contract if available
            try:
                if run_manifest is not None and isinstance(run_manifest.get('run_id'), str):
                    if 'run_id' not in ranked.columns:
                        ranked.insert(0, 'run_id', run_manifest['run_id'])
            except Exception:
                pass
            path = save_ranked_whitespace(ranked, cutoff_tag=cutoff_tag)
            logger.info(f"Saved Phase-4 ranked whitespace to {path}")
            # Also save segmented ranked outputs for warm/cold/prospect
            try:
                if 'segment' in ranked.columns:
                    for seg_name in ['warm','cold','prospect']:
                        seg_out = ranked[ranked['segment'] == seg_name]
                        seg_namefile = f"whitespace_{seg_name}_{cutoff_tag}.csv" if cutoff_tag else f"whitespace_{seg_name}.csv"
                        seg_out.to_csv(OUTPUTS_DIR / seg_namefile, index=False)
            except Exception:
                pass

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
                            sort_champ = [c for c in ['score','p_icp'] if c in ranked.columns]
                            champ_sel = ranked.sort_values(by=sort_champ, ascending=False, na_position='last').head(topn)
                            leg_sel = legacy_pd.sort_values(by=['score'], ascending=False, na_position='last').head(topn)
                            champ_top = set(champ_sel['customer_id'].astype(str).tolist())
                            leg_top = set(leg_sel['customer_id'].astype(str).tolist())
                            inter = len(champ_top & leg_top)
                            union = len(champ_top | leg_top)
                            jacc = float(inter) / max(1, union)
                            overlap = {"top_percent": 10, "intersection": int(inter), "union": int(union), "jaccard": jacc}
                            ov_name = f"whitespace_overlap_{cutoff_tag}.json" if cutoff_tag else "whitespace_overlap.json"
                            import json as _json
                            (OUTPUTS_DIR / ov_name).write_text(_json.dumps(overlap, indent=2), encoding='utf-8')
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Shadow mode failed: {e}")

            # Challenger overlap (champion vs score_challenger) at top-10%
            try:
                if 'score_challenger' in ranked.columns and len(ranked) > 0:
                    topn = max(1, int(len(ranked) * 0.10))
                    # Build numeric sort keys only to avoid dtype issues
                    sort_champ = [c for c in ['score','p_icp'] if c in ranked.columns]
                    sort_chall = [c for c in ['score_challenger','p_icp'] if c in ranked.columns]
                    champ_sel = ranked.sort_values(by=sort_champ, ascending=False, na_position='last').head(topn)
                    chall_sel = ranked.sort_values(by=sort_chall, ascending=False, na_position='last').head(topn)
                    champ_top = set(champ_sel['customer_id'].astype(str).tolist())
                    chall_top = set(chall_sel['customer_id'].astype(str).tolist())
                    inter = len(champ_top & chall_top)
                    union = len(champ_top | chall_top)
                    jacc = float(inter) / max(1, union)
                    overlap = {"top_percent": 10, "intersection": int(inter), "union": int(union), "jaccard": jacc}
                    ov_name = f"whitespace_challenger_overlap_{cutoff_tag}.json" if cutoff_tag else "whitespace_challenger_overlap.json"
                    import json as _json
                    (OUTPUTS_DIR / ov_name).write_text(_json.dumps(overlap, indent=2), encoding='utf-8')
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
                        sort_cols = [c for c in ['score', 'p_icp', 'EV_norm'] if c in ranked.columns]
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
                top10 = ranked.sort_values(by=[c for c in ['score','p_icp'] if c in ranked.columns], ascending=False, na_position='last').head(top10_n) if top10_n > 0 else ranked.head(0)
                shares = top10.groupby('division_name')['customer_id'].size().sort_values(ascending=False) if top10_n > 0 and 'division_name' in top10.columns else pd.Series(dtype=int)
                share_map = {str(k): float(v) / max(1, int(len(top10))) for k, v in shares.items()} if top10_n > 0 else {}
                metrics = {
                    "rows": int(len(ranked)),
                    "checksum": checksum,
                    "division_shares_top10pct": share_map,
                }
                met_name = f"whitespace_metrics_{cutoff_tag}.json" if cutoff_tag else "whitespace_metrics.json"
                import json as _json
                (OUTPUTS_DIR / met_name).write_text(_json.dumps(metrics, indent=2), encoding='utf-8')

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
                    sort_cols = [c for c in ['score', 'p_icp', 'EV_norm'] if c in ranked.columns]
                    if mode == 'top_percent':
                        if len(ranked) > 0:
                            ksel = max(1, int(len(ranked) * (cfg.modeling.capacity_percent / 100.0)))
                            # Segment-aware allocation (optional)
                            seg_alloc = getattr(getattr(cfg, 'whitespace', object()), 'segment_allocation', None)
                            if seg_alloc and 'segment' in ranked.columns and sort_cols:
                                try:
                                    alloc = {k.lower(): float(v) for k, v in dict(seg_alloc).items()}
                                except Exception:
                                    alloc = {}
                                # Normalize allocation and compute counts
                                warm_r = max(0.0, alloc.get('warm', 0.0))
                                cold_r = max(0.0, alloc.get('cold', 0.0))
                                pros_r = max(0.0, alloc.get('prospect', 0.0))
                                ssum = warm_r + cold_r + pros_r
                                if ssum > 0:
                                    warm_n = int(round(ksel * warm_r / ssum))
                                    cold_n = int(round(ksel * cold_r / ssum))
                                    pros_n = max(0, ksel - warm_n - cold_n)
                                    # Take top-N per segment
                                    def top_seg(name: str, n: int) -> pd.DataFrame:
                                        if n <= 0:
                                            return ranked.head(0)
                                        sub = ranked[ranked['segment'].astype(str).str.lower() == name]
                                        return sub.sort_values(by=sort_cols, ascending=False, na_position='last').head(n)
                                    parts = [
                                        top_seg('warm', warm_n),
                                        top_seg('cold', cold_n),
                                        top_seg('prospect', pros_n),
                                    ]
                                    initial = pd.concat(parts, ignore_index=True)
                                    # Top-up to ksel with next best remaining if short
                                    if len(initial) < ksel:
                                        remaining = ranked.merge(initial[['customer_id','division_name']], on=['customer_id','division_name'], how='left', indicator=True)
                                        remaining = remaining[remaining['_merge'] == 'left_only'].drop(columns=['_merge'])
                                        top_up = remaining.sort_values(by=sort_cols, ascending=False, na_position='last').head(ksel - len(initial))
                                        initial = pd.concat([initial, top_up], ignore_index=True)
                                else:
                                    initial = ranked.sort_values(by=sort_cols, ascending=False, na_position='last').head(ksel).copy()
                            else:
                                initial = ranked.sort_values(by=sort_cols, ascending=False, na_position='last').head(ksel).copy() if sort_cols else ranked.head(0)
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
                        selected = ranked.sort_values(by=sort_cols, ascending=False, na_position='last').head(ksel).copy() if len(ranked) and sort_cols else ranked.head(0)

                    sel_name = f"whitespace_selected_{cutoff_tag}.csv" if cutoff_tag else "whitespace_selected.csv"
                    selected.to_csv(OUTPUTS_DIR / sel_name, index=False)
                    # Emit per-segment selected files when segment column is present
                    try:
                        if 'segment' in selected.columns:
                            for seg_name in ['warm','cold','prospect']:
                                seg_sel = selected[selected['segment'] == seg_name]
                                seg_file = f"whitespace_selected_{seg_name}_{cutoff_tag}.csv" if cutoff_tag else f"whitespace_selected_{seg_name}.csv"
                                seg_sel.to_csv(OUTPUTS_DIR / seg_file, index=False)
                    except Exception:
                        pass

                    cap_df = None
                    try:
                        cap_df = _emit_capacity_and_logs(ranked, selected, cutoff_tag=cutoff_tag)
                    except Exception as exc:
                        logger.warning("Failed to emit capacity/log summaries: %s", exc)

                    if cap_df is not None and not cap_df.empty and 'division_name' in cap_df.columns:
                        share_series = cap_df.set_index('division_name')['selected_count'].sort_values(ascending=False)
                        total_sel = int(cap_df['selected_count'].sum()) or 0
                    else:
                        share_series = (
                            selected.groupby('division_name')['customer_id'].size().sort_values(ascending=False)
                            if len(selected) > 0 and 'division_name' in selected.columns
                            else pd.Series(dtype=int)
                        )
                        total_sel = int(len(selected))
                    total_sel = max(1, total_sel)
                    share_map = {str(k): float(v) / total_sel for k, v in share_series.items()} if len(share_series) else {}
                    threshold = float(cfg.whitespace.bias_division_max_share_topN)
                    max_share = max(share_map.values()) if share_map else 0.0
                    warn = {
                        "capacity_mode": mode,
                        "selected_rows": int(len(selected)),
                        "division_shares": share_map,
                        "threshold": float(threshold),
                        "warn": bool(max_share > threshold),
                    }
                    warn_name = f"bias_diversity_warnings_{cutoff_tag}.json" if cutoff_tag else "bias_diversity_warnings.json"
                    # Write via json to avoid pandas dtype issues when nested dicts present
                    import json as _json
                    (OUTPUTS_DIR / warn_name).write_text(_json.dumps(warn, indent=2), encoding='utf-8')
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
