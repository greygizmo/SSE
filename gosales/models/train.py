"""Train GoSales propensity models and persist calibrated artifacts.

This module powers the standard training entry point, wiring together feature
generation, model selection, evaluation metrics, and artifact serialization so
that CLI wrappers and pipelines can produce reproducible models.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

from gosales.utils.config import load_config
from gosales.utils.db import get_db_connection, get_curated_connection, validate_connection
from gosales.features.engine import create_feature_matrix
from gosales.utils.paths import OUTPUTS_DIR, MODELS_DIR
from gosales.utils.logger import get_logger
from gosales.models.metrics import (
    compute_lift_at_k,
    compute_weighted_lift_at_k,
    compute_topk_threshold,
    calibration_bins,
    calibration_mae,
)
from gosales.ops.run import run_context


logger = get_logger(__name__)


def _lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k_percent: int) -> float:
    return compute_lift_at_k(y_true, y_score, k_percent)


def _weighted_lift_at_k(y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray, k_percent: int) -> float:
    return compute_weighted_lift_at_k(y_true, y_score, weights, k_percent)


def _infer_revenue_weights(df: pd.DataFrame) -> np.ndarray:
    """Infer revenue-style weights for lift calculations.

    Prefers recent gross profit aggregates with fallback to lifetime totals.
    Returns an array aligned with ``df`` rows. Missing/invalid values become 0.0.
    """

    weights = np.ones(len(df), dtype=float)
    candidates = ["rfm__all__gp_sum__12m", "total_gp_all_time"]
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float).values
            if np.any(vals):
                weights = vals
                break
    return weights


def _is_backbone_feature(column: str, backbone_cols: set[str], backbone_prefixes: tuple[str, ...]) -> bool:
    col = str(column)
    if col in backbone_cols:
        return True
    return any(col.startswith(pref) for pref in backbone_prefixes if pref)


def _collect_topn_ids(
    probs: np.ndarray,
    valid_index: pd.Index,
    df: pd.DataFrame,
    top_k_percents: list[int],
) -> dict[int, list[str]]:
    """Return top-N customer ids for each configured percentile."""

    topn: dict[int, list[str]] = {}
    if len(probs) == 0:
        return topn
    for k in top_k_percents:
        try:
            frac = max(1, int(math.ceil(len(probs) * float(k) / 100.0)))
        except Exception:
            frac = max(1, int(len(probs) * 0.10))
        order = np.argsort(-probs)
        sel = order[:frac]
        ids = []
        for pos in sel:
            try:
                row_idx = valid_index[pos]
                ids.append(str(df.loc[row_idx, "customer_id"]))
            except Exception:
                continue
        topn[int(k)] = sorted(dict.fromkeys(ids))
    return topn


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float, int)):
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return value


def _aggregate_stability(
    results: list[dict],
    top_k_percents: list[int],
    penalty_lambda: float,
    cv_guard: float,
) -> tuple[pd.DataFrame, str, dict[str, dict]]:
    """Aggregate per-cutoff metrics and select champion via stability objective."""

    if not results:
        return pd.DataFrame(), "", {}

    df = pd.DataFrame(results)
    records = []
    summary: dict[str, dict] = {}

    for model, group in df.groupby("model"):
        entry: dict[str, dict | float | bool] = {
            "rev_lift": {},
            "lift": {},
        }
        objective_mean = np.nan
        objective_std = np.nan
        for k in top_k_percents:
            rev_col = f"rev_lift@{k}"
            lift_col = f"lift@{k}"
            rev_vals = pd.to_numeric(group.get(rev_col, pd.Series([], dtype=float)), errors="coerce")
            lift_vals = pd.to_numeric(group.get(lift_col, pd.Series([], dtype=float)), errors="coerce")
            rev_vals = rev_vals.replace({np.inf: np.nan, -np.inf: np.nan})
            lift_vals = lift_vals.replace({np.inf: np.nan, -np.inf: np.nan})
            rev_mean = float(np.nanmean(rev_vals.values)) if len(rev_vals) else np.nan
            rev_std = float(np.nanstd(rev_vals.values, ddof=0)) if len(rev_vals) else np.nan
            if rev_mean == 0 or np.isnan(rev_mean):
                rev_cv = float("inf") if rev_std not in (0.0, 0) else np.nan
            else:
                rev_cv = float(abs(rev_std) / abs(rev_mean))
            lift_mean = float(np.nanmean(lift_vals.values)) if len(lift_vals) else np.nan
            lift_std = float(np.nanstd(lift_vals.values, ddof=0)) if len(lift_vals) else np.nan
            if lift_mean == 0 or np.isnan(lift_mean):
                lift_cv = float("inf") if lift_std not in (0.0, 0) else np.nan
            else:
                lift_cv = float(abs(lift_std) / abs(lift_mean))
            entry["rev_lift"][int(k)] = {"mean": rev_mean, "std": rev_std, "cv": rev_cv}
            entry["lift"][int(k)] = {"mean": lift_mean, "std": lift_std, "cv": lift_cv}
            if int(k) == 10:
                objective_mean = rev_mean
                objective_std = rev_std

        guard_fail = False
        rev10 = entry["rev_lift"].get(10, {}) if isinstance(entry["rev_lift"], dict) else {}
        rev10_cv = float(rev10.get("cv", float("inf"))) if rev10 else float("inf")
        if cv_guard and math.isfinite(cv_guard):
            guard_fail = bool(rev10_cv > cv_guard)
        objective = float("-inf")
        if not math.isnan(objective_mean):
            std_term = float(objective_std) if not math.isnan(objective_std) else 0.0
            objective = float(objective_mean - penalty_lambda * std_term)
        summary[model] = {
            **entry,
            "objective": objective,
            "objective_components": {
                "mean_rev_lift10": objective_mean,
                "std_rev_lift10": objective_std,
                "lambda": penalty_lambda,
            },
            "cv_guard_threshold": cv_guard,
            "cv_guard_fail": guard_fail,
        }
        record = {
            "model": model,
            "objective": objective,
            "cv_guard_fail": guard_fail,
        }
        for k, stats in entry["rev_lift"].items():
            record[f"rev_lift_mean@{k}"] = float(stats.get("mean")) if stats else None
            record[f"rev_lift_std@{k}"] = float(stats.get("std")) if stats else None
            record[f"rev_lift_cv@{k}"] = float(stats.get("cv")) if stats else None
        for k, stats in entry["lift"].items():
            record[f"lift_mean@{k}"] = float(stats.get("mean")) if stats else None
            record[f"lift_std@{k}"] = float(stats.get("std")) if stats else None
            record[f"lift_cv@{k}"] = float(stats.get("cv")) if stats else None
        records.append(record)

    agg_df = pd.DataFrame(records) if records else pd.DataFrame()
    if not len(agg_df):
        return agg_df, "", summary

    ranked = agg_df.sort_values("objective", ascending=False)
    guard_candidates = ranked[~ranked["cv_guard_fail"].astype(bool)]
    if len(guard_candidates):
        winner_row = guard_candidates.iloc[0]
    else:
        winner_row = ranked.iloc[0]
    winner = str(winner_row["model"])
    return agg_df, winner, summary

def _train_test_split_time_aware(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Prefer time-aware split using recency if feature present; else fall back to stratified split
    recency_col = 'rfm__all__recency_days__life'
    if recency_col in X.columns:
        df = X.copy()
        df['_y'] = y
        # Smaller recency_days = more recent â†’ assign those to validation
        df = df.sort_values(recency_col, ascending=True)
        n = len(df)
        n_valid = max(1, int(0.2 * n))
        valid = df.iloc[:n_valid]
        train = df.iloc[n_valid:]
        X_train = train.drop(columns=['_y'])
        y_train = train['_y']
        X_valid = valid.drop(columns=['_y'])
        y_valid = valid['_y']
        return X_train, X_valid, y_train, y_valid
    # Fallback stratified
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def _sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric dtype; replace infs with NaN then fill NaN with 0.0.

    Protects scaler/estimators from infinities and mixed dtypes.
    """
    Xc = X.copy()
    for col in Xc.columns:
        if not (pd.api.types.is_integer_dtype(Xc[col]) or pd.api.types.is_float_dtype(Xc[col])):
            Xc[col] = pd.to_numeric(Xc[col], errors='coerce')
    Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
    return Xc.fillna(0.0)


def _drop_low_variance(X: pd.DataFrame, threshold: float = 1e-12) -> tuple[pd.DataFrame, list[str]]:
    """Drop constant and near-constant columns to help optimization stability."""
    variances = X.var(axis=0, numeric_only=True)
    low_var_cols = [c for c, v in variances.items() if (pd.isna(v) or float(v) <= threshold)]
    Xr = X.drop(columns=low_var_cols, errors="ignore")
    return Xr, low_var_cols


def _drop_high_correlation(X: pd.DataFrame, threshold: float = 0.995) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Drop one of each pair of highly correlated features (absolute Pearson > threshold)."""
    dropped_pairs: list[tuple[str, str]] = []
    try:
        num = X.select_dtypes(include=[np.number])
        if num.shape[1] <= 1:
            return X, dropped_pairs
        corr = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        for col in to_drop:
            offending = upper.index[upper[col] > threshold].tolist()
            if offending:
                dropped_pairs.append((offending[0], col))
        Xr = X.drop(columns=list(set(to_drop)), errors="ignore")
        return Xr, dropped_pairs
    except Exception:
        return X, dropped_pairs


def _maybe_export_shap(
    model,
    X_final: pd.DataFrame,
    df_final: pd.DataFrame,
    division: str,
    feature_names: list[str],
    shap_sample: int,
    shap_max_rows: int,
    seed: int,
) -> dict[str, str]:
    """Compute and export SHAP summaries if enabled.

    Returns a mapping of artifact filenames to their paths. When ``shap_sample`` is
    0 or the dataset size exceeds ``shap_max_rows`` the computation is skipped and a
    warning logged.
    """
    artifacts: dict[str, str] = {}
    if shap_sample <= 0:
        logger.warning("SHAP sample N is zero; skipping SHAP computation")
        return artifacts
    if len(X_final) > shap_max_rows:
        logger.warning(
            "Skipping SHAP: dataset has %d rows exceeding threshold %d",
            len(X_final),
            shap_max_rows,
        )
        return artifacts
    if not _HAS_SHAP:
        logger.warning("SHAP library not available; skipping SHAP computation")
        return artifacts

    base = getattr(model, "base_estimator", None)
    if base is None and hasattr(model, "estimator"):
        base = model.estimator
    if base is None:
        base = model
    if not hasattr(base, "predict_proba"):
        logger.warning("Model does not support SHAP; skipping")
        return artifacts

    sample_n = min(shap_sample, len(X_final))
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(len(X_final), size=sample_n, replace=False)
    X_sample = X_final.iloc[sample_idx]
    cust_ids = df_final.iloc[sample_idx]["customer_id"].values

    try:
        if isinstance(base, LGBMClassifier):
            explainer = shap.TreeExplainer(base)
            shap_vals = explainer.shap_values(X_sample)
            vals = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) == 2 else shap_vals
        elif isinstance(base, LogisticRegression):
            explainer = shap.LinearExplainer(base, X_sample)
            vals = explainer.shap_values(X_sample)
        else:
            logger.warning("Unsupported model type for SHAP; skipping")
            return artifacts

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        mean_abs = np.mean(np.abs(vals), axis=0)
        shap_global = OUTPUTS_DIR / f"shap_global_{division.lower()}.csv"
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})\
            .sort_values("mean_abs_shap", ascending=False)\
            .to_csv(shap_global, index=False)
        artifacts[shap_global.name] = str(shap_global)

        sample_df = pd.DataFrame(vals, columns=feature_names)
        sample_df.insert(0, "customer_id", cust_ids)
        shap_sample_path = OUTPUTS_DIR / f"shap_sample_{division.lower()}.csv"
        sample_df.to_csv(shap_sample_path, index=False)
        artifacts[shap_sample_path.name] = str(shap_sample_path)
    except Exception as e:
        logger.warning(f"Failed SHAP export: {e}")
    return artifacts


def _emit_diagnostics(out_dir: Path, division: str, context: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        diag_path = OUTPUTS_DIR / f"diagnostics_{division.lower()}.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)
    except Exception:
        pass

def _calibrate(
    clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    method: str,
    folds: int | None = 3,
    sparse_isotonic_threshold_pos: int | None = None,
):
    """Safely calibrate a classifier, adapting cv to class counts.

    Returns a tuple: (calibrated_model_or_none, chosen_method, reason_or_none)
    - chosen_method: 'platt' | 'isotonic' | 'none'
    - reason_or_none: string reason when calibration is skipped/fails
    """
    try:
        # Class balance checks on training labels
        pos = int(np.sum(y_train))
        neg = int(len(y_train) - pos)
        if pos == 0 or neg == 0:
            return None, "none", "single_class_train"

        # Prefer requested method; optionally downgrade isotonic for sparse positives
        chosen = str(method).strip().lower()
        if chosen not in ("platt", "isotonic"):
            chosen = "platt"
        if chosen == "isotonic" and sparse_isotonic_threshold_pos is not None:
            try:
                if pos < int(sparse_isotonic_threshold_pos):
                    chosen = "platt"
            except Exception:
                pass

        # Determine feasible cv folds given per-class counts
        req_folds = int(folds or 3)
        n_splits = min(req_folds, pos, neg)
        if n_splits < 2:
            return None, "none", "insufficient_per_class"

        calibrated = CalibratedClassifierCV(
            estimator=clf,
            method=("sigmoid" if chosen == "platt" else "isotonic"),
            cv=n_splits,
        )
        calibrated.fit(X_train, y_train)
        return calibrated, chosen, None
    except Exception as e:
        try:
            name = type(e).__name__
        except Exception:
            name = "CalibrationError"
        return None, "none", f"calibration_error:{name}"


@click.command()
@click.option("--division", required=True)
@click.option("--cutoffs", required=True, help="comma-separated cutoffs")
@click.option("--window-months", default=6, type=int)
@click.option("--models", default="logreg,lgbm")
@click.option("--calibration", default="platt,isotonic")
@click.option("--shap-sample", default=0, type=int, help="Rows to sample for SHAP; 0 disables")
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
@click.option("--group-cv/--no-group-cv", default=False, help="Use GroupKFold by customer_id for train/valid split (leakage guard)")
@click.option("--purge-days", default=0, type=int, help="Embargo/purge days between train and validation (time-aware splits)")
@click.option("--label-buffer-days", default=0, type=int, help="Start labels at cutoff+buffer_days (horizon buffer)")
@click.option("--safe-mode/--no-safe-mode", default=False, help="Apply SAFE feature policy (drop/lag high-risk adjacency feature families)")
@click.option("--dry-run/--no-dry-run", default=False, help="Skip training; only verify inputs and emit planned artifacts to manifest")
def main(division: str, cutoffs: str, window_months: int, models: str, calibration: str, shap_sample: int, config: str, group_cv: bool, purge_days: int, label_buffer_days: int, safe_mode: bool, dry_run: bool) -> None:
    cfg = load_config(config)
    # Determine SAFE policy: CLI flag or per-division config override
    auto_safe = bool(safe_mode)
    try:
        divs = [str(d).lower() for d in getattr(cfg, 'modeling', object()).safe_divisions]  # type: ignore[attr-defined]
        if str(division).lower() in divs:
            auto_safe = True
    except Exception:
        pass

    stability_cfg = getattr(getattr(cfg, 'modeling', object()), 'stability', object())
    coverage_floor = float(getattr(stability_cfg, 'coverage_floor', 0.0) or 0.0)
    min_positive_rate = float(getattr(stability_cfg, 'min_positive_rate', 0.0) or 0.0)
    min_positive_count = int(getattr(stability_cfg, 'min_positive_count', 0) or 0)
    penalty_lambda = float(getattr(stability_cfg, 'penalty_lambda', 0.0) or 0.0)
    cv_guard_threshold = getattr(stability_cfg, 'cv_guard_max', None)
    try:
        cv_guard = float(cv_guard_threshold) if cv_guard_threshold is not None else float('inf')
    except Exception:
        cv_guard = float('inf')
    sparse_family_prefixes = [str(s) for s in getattr(stability_cfg, 'sparse_family_prefixes', [])]
    backbone_features = {str(s) for s in getattr(stability_cfg, 'backbone_features', [])}
    backbone_prefixes = tuple(str(s) for s in getattr(stability_cfg, 'backbone_prefixes', []))

    cut_list = [c.strip() for c in cutoffs.split(",") if c.strip()]
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Prefer curated connection where fact tables exist; fallback to primary DB
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()
    # Connection health check
    try:
        strict = bool(getattr(getattr(cfg, 'database', object()), 'strict_db', False))
    except Exception:
        strict = False
    if not validate_connection(engine):
        msg = "Database connection is unhealthy."
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    # Accumulate metrics across cutoffs per model
    model_names = [m.strip() for m in models.split(",") if m.strip()]
    cal_methods = [c.strip() for c in calibration.split(",") if c.strip()]

    artifacts: dict[str, str] = {}
    results = []
    with run_context("phase3_train") as ctx:
        if dry_run:
            # Plan artifacts without training
            out_dir = MODELS_DIR / f"{division.lower()}_model"
            artifacts.update({
                "planned_model.pkl": str(out_dir / "model.pkl"),
                "planned_feature_list.json": str(out_dir / "feature_list.json"),
                f"planned_metrics_{division.lower()}.json": str(OUTPUTS_DIR / f"metrics_{division.lower()}.json"),
                f"planned_gains_{division.lower()}.csv": str(OUTPUTS_DIR / f"gains_{division.lower()}.csv"),
                f"planned_calibration_{division.lower()}.csv": str(OUTPUTS_DIR / f"calibration_{division.lower()}.csv"),
                f"planned_thresholds_{division.lower()}.csv": str(OUTPUTS_DIR / f"thresholds_{division.lower()}.csv"),
                f"planned_model_card_{division.lower()}.json": str(OUTPUTS_DIR / f"model_card_{division.lower()}.json"),
            })
            try:
                ctx["write_manifest"](artifacts)
                ctx["append_registry"]({"phase": "phase3_train", "division": division, "cutoffs": cut_list, "artifact_count": len(artifacts), "status": "dry-run"})
            except Exception:
                pass
            return
        all_dropped_low_var: list[str] = []
        all_dropped_corr: list[tuple[str, str]] = []
        prevalence_stats: list[dict[str, float]] = []
        coverage_drops_by_cutoff: dict[str, list[str]] = {}
        family_drops_by_cutoff: dict[str, list[str]] = {}
        topn_registry: dict[tuple[str, str], dict[int, list[str]]] = {}
        union_coverage_drop: set[str] = set()
        union_sparse_drop: set[str] = set()
        # In SAFE (audit) mode, apply gauntlet tail-mask to windowed features
        gauntlet_mask_tail = 0
        try:
            if auto_safe:
                gauntlet_mask_tail = int(getattr(getattr(cfg, 'validation', object()), 'gauntlet_mask_tail_days', 0) or 0)
        except Exception:
            gauntlet_mask_tail = 0

        for cutoff in cut_list:
            fm = create_feature_matrix(
                engine,
                division,
                cutoff,
                window_months,
                mask_tail_days=gauntlet_mask_tail if auto_safe else None,
                label_buffer_days=label_buffer_days,
            )
            # Persist features parquet for validation phase (Phase 5) compatibility
            try:
                from gosales.utils.paths import OUTPUTS_DIR as _OUT
                out_path = _OUT / f"features_{division.lower()}_{cutoff}.parquet"
                fm.write_parquet(out_path)
            except Exception:
                pass
            if fm.is_empty():
                logger.warning(f"Empty feature matrix for cutoff {cutoff}")
                continue
            df = fm.to_pandas()
            if 'bought_in_division' not in df.columns:
                logger.warning("Missing target column for cutoff %s", cutoff)
                continue
            y_series = df['bought_in_division'].astype(int)
            y = y_series.values
            total_rows = int(len(y))
            pos_count = int(y_series.sum())
            prevalence = float(pos_count / total_rows) if total_rows else 0.0
            weights_series = pd.Series(_infer_revenue_weights(df), index=df.index)
            X_raw = df.drop(columns=['customer_id', 'bought_in_division'])
            coverage_series = X_raw.notna().mean()
            coverage_drop_cols: list[str] = []
            if coverage_floor > 0:
                coverage_drop_cols = [
                    c for c, cov in coverage_series.items()
                    if (float(cov) < coverage_floor)
                    and not _is_backbone_feature(c, backbone_features, backbone_prefixes)
                ]
                if coverage_drop_cols:
                    X_raw = X_raw.drop(columns=coverage_drop_cols, errors='ignore')
                    coverage_drops_by_cutoff[cutoff] = sorted(dict.fromkeys([str(c) for c in coverage_drop_cols]))
                    union_coverage_drop.update(coverage_drop_cols)
            drop_sparse = False
            if total_rows:
                drop_sparse = (prevalence < min_positive_rate) or (pos_count < min_positive_count)
            sparse_drop_cols: list[str] = []
            if drop_sparse and sparse_family_prefixes:
                for col in list(X_raw.columns):
                    if _is_backbone_feature(col, backbone_features, backbone_prefixes):
                        continue
                    col_str = str(col)
                    if any(col_str.startswith(pref) for pref in sparse_family_prefixes):
                        sparse_drop_cols.append(col)
                if sparse_drop_cols:
                    X_raw = X_raw.drop(columns=sparse_drop_cols, errors='ignore')
                    family_drops_by_cutoff[cutoff] = sorted(dict.fromkeys([str(c) for c in sparse_drop_cols]))
                    union_sparse_drop.update(sparse_drop_cols)
            prevalence_stats.append({
                "cutoff": cutoff,
                "total": total_rows,
                "positives": pos_count,
                "prevalence": prevalence,
                "coverage_floor_dropped": len(coverage_drop_cols),
                "sparse_family_dropped": len(sparse_drop_cols),
            })
            X = _sanitize_features(X_raw)
            # SAFE policy: drop high-risk adjacency families
            if auto_safe:
                try:
                    cols = []
                    for c in X.columns:
                        s = str(c).lower()
                        if s.startswith('assets_expiring_'):
                            continue
                        if s.startswith('assets_subs_share_') or s.startswith('assets_on_subs_share_') or s.startswith('assets_off_subs_share_'):
                            continue
                        if 'days_since_last' in s or 'recency' in s:
                            continue
                        if '__3m' in s or s.endswith('_last_3m') or '__6m' in s or s.endswith('_last_6m') or '__12m' in s or s.endswith('_last_12m'):
                            continue
                        if s.startswith('als_f'):
                            continue
                        if s.startswith('gp_12m_') or s.startswith('tx_12m_'):
                            continue
                        if s in ('gp_2024', 'gp_2023'):
                            continue
                        # Division share momentum and SKU short-term families
                        if s.startswith('xdiv__div__gp_share__'):
                            continue
                        if s.startswith('sku_gp_12m_') or s.startswith('sku_qty_12m_') or s.startswith('sku_gp_per_unit_12m_'):
                            continue
                        cols.append(c)
                    X = X[cols]
                except Exception:
                    pass
            # Feature pruning for stability
            dropped_low_var: list[str] = []
            dropped_corr: list[tuple[str, str]] = []
            try:
                X, dropped_low_var = _drop_low_variance(X)
                X, dropped_corr = _drop_high_correlation(X)
                all_dropped_low_var.extend([c for c in dropped_low_var if c not in all_dropped_low_var])
                all_dropped_corr.extend(dropped_corr)
            except Exception:
                pass
            overlap_csv = None
            if group_cv:
                try:
                    rec_col = 'rfm__all__recency_days__life'
                    groups = df['customer_id'].astype(str).values
                    if int(purge_days) > 0 and rec_col in df.columns:
                        from gosales.models.cv import BlockedPurgedGroupCV
                        rec = pd.to_numeric(df[rec_col], errors='coerce').fillna(1e9).astype(float).values
                        cv = BlockedPurgedGroupCV(n_splits=cfg.modeling.folds, purge_days=int(purge_days), seed=cfg.modeling.seed)
                        splits = list(cv.split(X, y, groups, anchor_days_from_cutoff=rec))
                        tr, va = splits[-1]
                    else:
                        from sklearn.model_selection import GroupKFold
                        gkf = GroupKFold(n_splits=cfg.modeling.folds)
                        splits = list(gkf.split(X, y, groups=groups))
                        tr, va = splits[-1]
                    X_train, X_valid, y_train, y_valid = X.iloc[tr], X.iloc[va], y[tr], y[va]
                except Exception:
                    X_train, X_valid, y_train, y_valid = _train_test_split_time_aware(X, y, cfg.modeling.seed)
            else:
                X_train, X_valid, y_train, y_valid = _train_test_split_time_aware(X, y, cfg.modeling.seed)

            # Emit fold overlap audit and optionally fail if any overlap
            try:
                train_ids = set(df.iloc[X_train.index]['customer_id'].astype(str))
                valid_ids = set(df.iloc[X_valid.index]['customer_id'].astype(str))
                overlap = sorted(train_ids.intersection(valid_ids))
                from gosales.utils.paths import OUTPUTS_DIR as _OUT
                overlap_path = _OUT / f"fold_customer_overlap_{division.lower()}_{cutoff}.csv"
                import pandas as _pd
                _pd.DataFrame({"customer_id": overlap}).to_csv(overlap_path, index=False)
                overlap_csv = str(overlap_path)
                if group_cv and len(overlap) > 0:
                    raise RuntimeError(f"GroupKFold overlap detected ({len(overlap)} customers). See {overlap_path}")
            except Exception as _e:
                if group_cv:
                    raise
                # Non-fatal for non-group splits; proceed

        # Baseline LR (elastic-net via saga)
        if 'logreg' in model_names:
            lr_params = cfg.modeling.lr_grid
            best_lr_pipe = None
            best_lr_auc = -1
            best_conv = True
            # Expanded grid and higher iteration budget
            grid_l1 = lr_params.get('l1_ratio', [0.0, 0.2, 0.5, 0.8])
            grid_C = lr_params.get('C', [0.1, 0.5, 1.0])
            # Class-weight control
            cw_cfg = str(cfg.modeling.class_weight).lower() if getattr(cfg, 'modeling', None) else 'balanced'
            class_weight = None if cw_cfg in ('none', 'null', '') else 'balanced'
            for l1_ratio in grid_l1:
                for C in grid_C:
                    if float(l1_ratio) == 0.0:
                        lr = LogisticRegression(
                            penalty='l2', solver='lbfgs', C=C, max_iter=10000, tol=1e-3,
                            class_weight=class_weight, random_state=cfg.modeling.seed
                        )
                    else:
                        lr = LogisticRegression(
                            penalty='elasticnet', solver='saga', l1_ratio=l1_ratio, C=C,
                            max_iter=10000, tol=1e-3, class_weight=class_weight, random_state=cfg.modeling.seed
                        )
                    pipe = Pipeline([
                        ('scaler', StandardScaler(with_mean=False)),
                        ('model', lr),
                    ])
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter("always", ConvergenceWarning)
                        pipe.fit(X_train, y_train)
                        conv_warn = any(isinstance(w.message, ConvergenceWarning) for w in ws)
                    p = pipe.predict_proba(X_valid)[:, 1]
                    auc_lr = roc_auc_score(y_valid, p)
                    if auc_lr > best_lr_auc:
                        best_lr_auc = auc_lr
                        best_lr_pipe = pipe
                        best_conv = not conv_warn
            if best_lr_pipe is not None:
                # Calibration on the entire pipeline (safe, per-cutoff)
                best_cal = None
                best_brier = None
                best_cal_method = "none"
                best_cal_reason = None
                for m in cal_methods:
                    cal, cm, reason = _calibrate(
                        best_lr_pipe,
                        X_train,
                        y_train,
                        X_valid,
                        m,
                        folds=getattr(getattr(cfg, 'modeling', object()), 'folds', 3),
                        sparse_isotonic_threshold_pos=getattr(getattr(cfg, 'modeling', object()), 'sparse_isotonic_threshold_pos', None),
                    )
                    if cal is None:
                        if best_cal is None and best_cal_reason is None:
                            best_cal_reason = reason
                        continue
                    p_cal = cal.predict_proba(X_valid)[:, 1]
                    brier = brier_score_loss(y_valid, p_cal)
                    if best_brier is None or brier < best_brier:
                        best_brier = brier
                        best_cal = cal
                        best_cal_method = cm
                        best_cal_reason = None

                # Choose calibrated if available, else fallback to uncalibrated pipeline
                if best_cal is not None:
                    p = best_cal.predict_proba(X_valid)[:, 1]
                else:
                    p = best_lr_pipe.predict_proba(X_valid)[:, 1]
                    try:
                        best_brier = brier_score_loss(y_valid, p)
                    except Exception:
                        best_brier = None
                weights_valid = weights_series.loc[X_valid.index].astype(float).values
                lifts_dict: dict[str, float | None] = {}
                rev_lifts_dict: dict[str, float | None] = {}
                for k in getattr(cfg.modeling, 'top_k_percents', [10]):
                    key = f"lift@{int(k)}"
                    try:
                        lifts_dict[key] = float(_lift_at_k(y_valid, p, int(k)))
                    except Exception:
                        lifts_dict[key] = None
                    rev_key = f"rev_lift@{int(k)}"
                    try:
                        val = _weighted_lift_at_k(y_valid, p, weights_valid, int(k))
                        rev_lifts_dict[rev_key] = float(val) if val == val else None
                    except Exception:
                        rev_lifts_dict[rev_key] = None
                lift10 = lifts_dict.get('lift@10')
                topn_registry[("logreg", cutoff)] = _collect_topn_ids(
                    p,
                    X_valid.index,
                    df,
                    [int(k) for k in getattr(cfg.modeling, 'top_k_percents', [10])],
                )
                # Pull n_iter_ from underlying LR if available
                try:
                    lr_inner = best_lr_pipe.named_steps.get('model')
                    n_iter_val = getattr(lr_inner, 'n_iter_', None)
                    if isinstance(n_iter_val, (list, np.ndarray)):
                        n_iter_val = [int(x) for x in np.ravel(n_iter_val).tolist()]
                except Exception:
                    n_iter_val = None
                results.append({
                    "cutoff": cutoff,
                    "model": "logreg",
                    "auc": float(best_lr_auc),
                    "lift10": float(lift10) if lift10 is not None else None,
                    "brier": float(best_brier) if best_brier is not None else None,
                    "calibration": best_cal_method,
                    "calibration_reason": best_cal_reason,
                    "converged": bool(best_conv),
                    "n_iter": n_iter_val,
                    "positives": pos_count,
                    "prevalence": prevalence,
                    **{k: v for k, v in lifts_dict.items()},
                    **{k: v for k, v in rev_lifts_dict.items()},
                })

        # LGBM challenger
        if 'lgbm' in model_names:
            # scale_pos_weight
            pos = int(np.sum(y_train))
            neg = int(len(y_train) - pos)
            spw = (neg / max(1, pos))
            use_spw = bool(getattr(cfg.modeling, 'use_scale_pos_weight', True))
            spw_cap = float(getattr(cfg.modeling, 'scale_pos_weight_cap', 10.0))
            grid = cfg.modeling.lgbm_grid
            best_lgbm = None
            best_auc = -1
            for num_leaves in grid.get('num_leaves', [31]):
                for min_data_in_leaf in grid.get('min_data_in_leaf', [50]):
                    for learning_rate in grid.get('learning_rate', [0.05]):
                        for feature_fraction in grid.get('feature_fraction', [0.9]):
                            for bagging_fraction in grid.get('bagging_fraction', [0.9]):
                                params = dict(
                                    random_state=cfg.modeling.seed,
                                    deterministic=True,
                                    n_jobs=1,
                                    n_estimators=400,
                                    learning_rate=learning_rate,
                                    num_leaves=num_leaves,
                                    min_data_in_leaf=min_data_in_leaf,
                                    feature_fraction=feature_fraction,
                                    bagging_fraction=bagging_fraction,
                                )
                                if use_spw:
                                    params['scale_pos_weight'] = min(spw, spw_cap)
                                lgbm = LGBMClassifier(**params)
                                lgbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc')
                                p = lgbm.predict_proba(X_valid)[:,1]
                                auc_lgbm = roc_auc_score(y_valid, p)
                                # Overfit guard: compare train vs valid AUC; if large gap, try stronger regularization once
                                try:
                                    p_tr = lgbm.predict_proba(X_train)[:,1]
                                    auc_tr = roc_auc_score(y_train, p_tr)
                                    gap = float(auc_tr - auc_lgbm)
                                except Exception:
                                    gap = 0.0
                                if gap > 0.05:
                                    reg_params = dict(
                                        random_state=cfg.modeling.seed,
                                        deterministic=True,
                                        n_jobs=1,
                                        n_estimators=400,
                                        learning_rate=learning_rate,
                                        num_leaves=max(15, int(num_leaves * 0.8)),
                                        min_data_in_leaf=int(min_data_in_leaf * 2),
                                        feature_fraction=max(0.5, feature_fraction * 0.9),
                                        bagging_fraction=max(0.5, bagging_fraction * 0.9),
                                    )
                                    if use_spw:
                                        reg_params['scale_pos_weight'] = min(spw, spw_cap)
                                    reg_clf = LGBMClassifier(**reg_params)
                                    reg_clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc')
                                    p_reg = reg_clf.predict_proba(X_valid)[:,1]
                                    auc_reg = roc_auc_score(y_valid, p_reg)
                                    if auc_reg >= auc_lgbm - 0.002:  # accept similar or better valid AUC with stronger regularization
                                        lgbm = reg_clf
                                        p = p_reg
                                        auc_lgbm = auc_reg
                                        logger.info(f"Overfit guard applied: gap={gap:.3f} -> using regularized params for LGBM")
                                if auc_lgbm > best_auc:
                                    best_auc = auc_lgbm
                                    best_lgbm = lgbm
            if best_lgbm is not None:
                # Calibration (safe)
                best_cal = None
                best_brier = None
                best_cal_method = "none"
                best_cal_reason = None
                for m in cal_methods:
                    cal, cm, reason = _calibrate(
                        best_lgbm,
                        X_train,
                        y_train,
                        X_valid,
                        m,
                        folds=getattr(getattr(cfg, 'modeling', object()), 'folds', 3),
                        sparse_isotonic_threshold_pos=getattr(getattr(cfg, 'modeling', object()), 'sparse_isotonic_threshold_pos', None),
                    )
                    if cal is None:
                        if best_cal is None and best_cal_reason is None:
                            best_cal_reason = reason
                        continue
                    p_cal = cal.predict_proba(X_valid)[:, 1]
                    brier = brier_score_loss(y_valid, p_cal)
                    if best_brier is None or brier < best_brier:
                        best_brier = brier
                        best_cal = cal
                        best_cal_method = cm
                        best_cal_reason = None

                if best_cal is not None:
                    p = best_cal.predict_proba(X_valid)[:, 1]
                else:
                    p = best_lgbm.predict_proba(X_valid)[:, 1]
                    try:
                        best_brier = brier_score_loss(y_valid, p)
                    except Exception:
                        best_brier = None
                weights_valid = weights_series.loc[X_valid.index].astype(float).values
                lifts_dict: dict[str, float | None] = {}
                rev_lifts_dict: dict[str, float | None] = {}
                for k in getattr(cfg.modeling, 'top_k_percents', [10]):
                    key = f"lift@{int(k)}"
                    try:
                        lifts_dict[key] = float(_lift_at_k(y_valid, p, int(k)))
                    except Exception:
                        lifts_dict[key] = None
                    rev_key = f"rev_lift@{int(k)}"
                    try:
                        val = _weighted_lift_at_k(y_valid, p, weights_valid, int(k))
                        rev_lifts_dict[rev_key] = float(val) if val == val else None
                    except Exception:
                        rev_lifts_dict[rev_key] = None
                lift10 = lifts_dict.get('lift@10')
                topn_registry[("lgbm", cutoff)] = _collect_topn_ids(
                    p,
                    X_valid.index,
                    df,
                    [int(k) for k in getattr(cfg.modeling, 'top_k_percents', [10])],
                )
                results.append({
                    "cutoff": cutoff,
                    "model": "lgbm",
                    "auc": float(best_auc),
                    "lift10": float(lift10) if lift10 is not None else None,
                    "brier": float(best_brier) if best_brier is not None else None,
                    "calibration": best_cal_method,
                    "calibration_reason": best_cal_reason,
                    "positives": pos_count,
                    "prevalence": prevalence,
                    **{k: v for k, v in lifts_dict.items()},
                    **{k: v for k, v in rev_lifts_dict.items()},
                })

    # Aggregate and select winner by stability-adjusted revenue lift
    if not results:
        logger.warning("No training results produced")
        return
    res_df = pd.DataFrame(results)
    agg, winner, stability_summary = _aggregate_stability(
        results,
        [int(k) for k in getattr(cfg.modeling, 'top_k_percents', [10])],
        penalty_lambda,
        cv_guard,
    )
    if not winner:
        logger.warning("Failed to select champion: no winner from stability objective")
        return
    winner_summary = stability_summary.get(winner, {})
    objective = winner_summary.get("objective")
    components = winner_summary.get("objective_components", {})
    rev10_stats = winner_summary.get("rev_lift", {}).get(10, {})
    logger.info(
        "Selected model %s via stability objective=%.4f (mean_rev_lift10=%.4f, std_rev_lift10=%.4f, cv=%.4f, guard_fail=%s)",
        winner,
        float(objective) if objective not in (None, "") else float("nan"),
        float(components.get("mean_rev_lift10", float("nan"))) if components else float("nan"),
        float(components.get("std_rev_lift10", float("nan"))) if components else float("nan"),
        float(rev10_stats.get("cv", float("nan"))) if rev10_stats else float("nan"),
        bool(winner_summary.get("cv_guard_fail")),
    )

    top_k_list = [int(k) for k in getattr(cfg.modeling, 'top_k_percents', [10])]
    winner_rows = [r for r in results if str(r.get("model")) == str(winner)]
    last_cut = cut_list[-1]
    holdout_row = None
    for r in winner_rows:
        if str(r.get("cutoff")) == str(last_cut):
            holdout_row = r
            break
    if holdout_row is None and winner_rows:
        holdout_row = winner_rows[-1]
    delta_holdout_rev = {}
    if holdout_row:
        for k in top_k_list:
            rev_key = f"rev_lift@{k}"
            holdout_val = holdout_row.get(rev_key)
            mean_val = winner_summary.get("rev_lift", {}).get(k, {}).get("mean")
            if holdout_val is None or mean_val is None or (isinstance(holdout_val, float) and math.isnan(holdout_val)):
                delta_holdout_rev[f"@{k}"] = None
            else:
                try:
                    delta_holdout_rev[f"@{k}"] = float(holdout_val) - float(mean_val)
                except Exception:
                    delta_holdout_rev[f"@{k}"] = None
    jaccard_stats: dict[str, float | None] = {}
    winner_topn = [topn_registry.get((winner, cutoff)) for cutoff in cut_list if (winner, cutoff) in topn_registry]
    if len(winner_topn) >= 2:
        for k in top_k_list:
            sets = [set(t.get(int(k), [])) for t in winner_topn if t and int(k) in t]
            vals = []
            for i, j in combinations(range(len(sets)), 2):
                a = sets[i]
                b = sets[j]
                union = len(a | b)
                if union == 0:
                    continue
                vals.append(len(a & b) / union)
            jaccard_stats[f"@{k}"] = float(np.mean(vals)) if vals else None
    else:
        for k in top_k_list:
            jaccard_stats[f"@{k}"] = None

    stability_export = {
        "objective": _json_safe(objective),
        "lambda": _json_safe(penalty_lambda),
        "cv_guard": _json_safe(cv_guard),
        "cv_guard_fail": bool(winner_summary.get("cv_guard_fail")),
        "summary": _json_safe(agg.replace({np.nan: None}).to_dict(orient='records') if len(agg) else []),
        "winner": winner,
        "rev_lift": _json_safe(winner_summary.get("rev_lift", {})),
        "lift": _json_safe(winner_summary.get("lift", {})),
        "delta_holdout_rev_lift": _json_safe(delta_holdout_rev),
        "jaccard_topn": _json_safe(jaccard_stats),
        "cutoff_prevalence": _json_safe(prevalence_stats),
    }

    # Train final on last cutoff for simplicity here
    fm_final = create_feature_matrix(
        engine,
        division,
        last_cut,
        window_months,
        mask_tail_days=gauntlet_mask_tail if auto_safe else None,
        label_buffer_days=label_buffer_days,
    )
    df_final = fm_final.to_pandas()
    y_final = df_final['bought_in_division'].astype(int).values
    X_final_raw = df_final.drop(columns=['customer_id', 'bought_in_division'])
    final_coverage_series = X_final_raw.notna().mean()
    final_coverage_drop = []
    if coverage_floor > 0:
        final_coverage_drop = [
            c for c, cov in final_coverage_series.items()
            if (float(cov) < coverage_floor)
            and not _is_backbone_feature(c, backbone_features, backbone_prefixes)
        ]
        if final_coverage_drop:
            X_final_raw = X_final_raw.drop(columns=final_coverage_drop, errors='ignore')
    if union_coverage_drop:
        X_final_raw = X_final_raw.drop(columns=[c for c in union_coverage_drop if c in X_final_raw.columns], errors='ignore')
    if union_sparse_drop:
        X_final_raw = X_final_raw.drop(columns=[c for c in union_sparse_drop if c in X_final_raw.columns], errors='ignore')
    X_final = _sanitize_features(X_final_raw)
    if auto_safe:
        try:
            cols = []
            for c in X_final.columns:
                s = str(c).lower()
                if s.startswith('assets_expiring_'):
                    continue
                if s.startswith('assets_subs_share_') or s.startswith('assets_on_subs_share_') or s.startswith('assets_off_subs_share_'):
                    continue
                if 'days_since_last' in s or 'recency' in s:
                    continue
                if '__3m' in s or s.endswith('_last_3m') or '__6m' in s or s.endswith('_last_6m') or '__12m' in s or s.endswith('_last_12m'):
                    continue
                if s.startswith('als_f'):
                    continue
                if s.startswith('gp_12m_') or s.startswith('tx_12m_'):
                    continue
                if s in ('gp_2024', 'gp_2023'):
                    continue
                if s.startswith('xdiv__div__gp_share__'):
                    continue
                if s.startswith('sku_gp_12m_') or s.startswith('sku_qty_12m_') or s.startswith('sku_gp_per_unit_12m_'):
                    continue
                cols.append(c)
            X_final = X_final[cols]
        except Exception:
            pass
    if final_coverage_drop:
        coverage_drops_by_cutoff[f"final_{last_cut}"] = sorted(dict.fromkeys([str(c) for c in final_coverage_drop]))
    pos_final = int(np.sum(y_final))
    total_final = int(len(y_final))
    prevalence_final = float(pos_final / total_final) if total_final else 0.0
    final_sparse_drop: list[str] = []
    if total_final and ((prevalence_final < min_positive_rate) or (pos_final < min_positive_count)):
        for col in list(X_final.columns):
            if _is_backbone_feature(col, backbone_features, backbone_prefixes):
                continue
            col_str = str(col)
            if any(col_str.startswith(pref) for pref in sparse_family_prefixes):
                final_sparse_drop.append(col)
        if final_sparse_drop:
            X_final = X_final.drop(columns=final_sparse_drop, errors='ignore')
            family_drops_by_cutoff[f"final_{last_cut}"] = sorted(dict.fromkeys([str(c) for c in final_sparse_drop]))
    # Minimal: refit winner without hyper search for brevity (could repeat best params)
    # Choose calibration method per-division based on volume (stability heuristic)
    # If positives >= sparse_isotonic_threshold_pos -> prefer isotonic; else Platt (sigmoid)
    final_cal_method = None
    try:
        pos_final = int(np.sum(y_final))
        thr = int(getattr(getattr(cfg, 'modeling', object()), 'sparse_isotonic_threshold_pos', 1000) or 1000)
        avail = set([m for m in cal_methods])
        if pos_final >= thr and 'isotonic' in avail:
            final_cal_method = 'isotonic'
        elif 'platt' in avail or 'sigmoid' in avail:
            final_cal_method = 'sigmoid'
        elif 'isotonic' in avail:
            final_cal_method = 'isotonic'
        else:
            final_cal_method = 'sigmoid'
    except Exception:
        final_cal_method = 'sigmoid'

    if winner == 'logreg':
        cw_cfg = str(cfg.modeling.class_weight).lower() if getattr(cfg, 'modeling', None) else 'balanced'
        class_weight = None if cw_cfg in ('none', 'null', '') else 'balanced'
        lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.2, C=1.0, max_iter=10000, tol=1e-3, class_weight=class_weight, random_state=cfg.modeling.seed)
        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('model', lr),
        ])
        pipe.fit(X_final, y_final)
        # Calibrate entire pipeline using selected method with dynamic cv
        try:
            pos_f = int(np.sum(y_final))
            neg_f = int(len(y_final) - pos_f)
            folds_f = int(getattr(getattr(cfg, 'modeling', object()), 'folds', 3) or 3)
            n_splits_f = min(folds_f, pos_f, neg_f)
        except Exception:
            n_splits_f = 3
        if n_splits_f >= 2:
            cal = CalibratedClassifierCV(
                pipe,
                method='isotonic' if final_cal_method == 'isotonic' else 'sigmoid',
                cv=n_splits_f,
            ).fit(X_final, y_final)
            model = cal
        else:
            logger.warning("Skipping final calibration for LR: insufficient per-class counts; using uncalibrated model")
            model = pipe
        feature_names = list(X_final.columns)
    else:
        clf = LGBMClassifier(random_state=cfg.modeling.seed, n_estimators=400, learning_rate=0.05, deterministic=True, n_jobs=1)
        clf.fit(X_final, y_final)
        # Dynamic cv for final calibration; fallback to uncalibrated if infeasible
        try:
            pos_f = int(np.sum(y_final))
            neg_f = int(len(y_final) - pos_f)
            folds_f = int(getattr(getattr(cfg, 'modeling', object()), 'folds', 3) or 3)
            n_splits_f = min(folds_f, pos_f, neg_f)
        except Exception:
            n_splits_f = 3
        if n_splits_f >= 2:
            cal = CalibratedClassifierCV(
                clf,
                method='isotonic' if final_cal_method == 'isotonic' else 'sigmoid',
                cv=n_splits_f,
            ).fit(X_final, y_final)
            model = cal
        else:
            logger.warning("Skipping final calibration for LGBM: insufficient per-class counts; using uncalibrated model")
            model = clf
        feature_names = list(X_final.columns)

    # Save artifacts
    out_dir = MODELS_DIR / f"{division.lower()}_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save model pickle via joblib
    try:
        import joblib
        joblib.dump(model, out_dir / "model.pkl")
        artifacts["model.pkl"] = str(out_dir / "model.pkl")
    except Exception:
        pass
    with open(out_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f)
    artifacts["feature_list.json"] = str(out_dir / "feature_list.json")
    # Prepare minimal metadata for scoring; finalize after metrics computed
    try:
        pos = int(np.sum(y_final))
        neg = int(max(0, len(y_final) - pos))
        spw = float((neg / pos)) if pos > 0 else None
        cal_label = 'isotonic' if str(final_cal_method).lower() == 'isotonic' else 'sigmoid'
        best_model_label = 'Logistic Regression' if str(winner).lower() == 'logreg' else 'LightGBM'
        _metadata = {
            "division": division,
            "cutoff_date": last_cut,
            "prediction_window_months": int(window_months),
            "feature_names": feature_names,
            "trained_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "best_model": best_model_label,
            "best_auc": None,
            "calibration_method": cal_label,
            "calibration_mae": None,
            "brier_score": None,
            "class_balance": {
                "positives": pos,
                "negatives": neg,
                "scale_pos_weight": spw,
            },
        }
    except Exception:
        _metadata = None
    # Final predictions and guardrails
    try:
        p_final = model.predict_proba(X_final)[:,1]
        if float(np.std(p_final)) < 0.01:
            logger.warning("Degenerate classifier (std(p) < 0.01). Aborting artifact write.")
            # Ensure minimal metadata exists even on degenerate runs
            try:
                if _metadata is not None:
                    with open(out_dir / "metadata.json", "w", encoding="utf-8") as mf:
                        json.dump(_metadata, mf, indent=2)
                    artifacts["metadata.json"] = str(out_dir / "metadata.json")
            except Exception:
                pass
            return
    except Exception:
        p_final = None

    # Metrics & artifacts
    try:
        # Persist train-time p_hat snapshot for Phase 5 drift comparison
        try:
            if p_final is not None:
                ts_path = OUTPUTS_DIR / f"train_scores_{division.lower()}_{last_cut}.csv"
                pd.DataFrame({"customer_id": df_final['customer_id'].values, "p_hat": p_final}).to_csv(
                    ts_path, index=False
                )
                artifacts[ts_path.name] = str(ts_path)
            # Persist train-time feature sample for Phase 5 PSI (sample to control size)
            try:
                num_cols = [c for c in feature_names if pd.api.types.is_numeric_dtype(df_final[c])]
                sample_df = df_final[['customer_id'] + num_cols].copy()
                if len(sample_df) > 5000:
                    sample_df = sample_df.sample(n=5000, random_state=cfg.modeling.seed)
                fs_path = OUTPUTS_DIR / f"train_feature_sample_{division.lower()}_{last_cut}.parquet"
                sample_df.to_parquet(fs_path, index=False)
                artifacts[fs_path.name] = str(fs_path)
            except Exception:
                pass
        except Exception:
            pass

        auc_val = roc_auc_score(y_final, p_final) if p_final is not None else None
        pr_prec, pr_rec, _ = precision_recall_curve(y_final, p_final) if p_final is not None else (None, None, None)
        pr_auc = auc(pr_rec, pr_prec) if pr_prec is not None else None
        brier = brier_score_loss(y_final, p_final) if p_final is not None else None
        lifts = {f"lift@{k}": _lift_at_k(y_final, p_final, k) for k in cfg.modeling.top_k_percents} if p_final is not None else {}
        weights_final = _infer_revenue_weights(df_final)
        weighted_lifts = {f"rev_lift@{k}": _weighted_lift_at_k(y_final, p_final, weights_final, k) for k in cfg.modeling.top_k_percents} if p_final is not None else {}

        # Gains (deciles)
        gains_df = pd.DataFrame({"y": y_final, "p": p_final})
        gains_df = gains_df.sort_values("p", ascending=False).reset_index(drop=True)
        idx = np.arange(len(gains_df))
        gains_df["decile"] = (np.floor((idx / max(1, len(gains_df)-1)) * 10) + 1)
        gains_df["decile"] = np.clip(gains_df["decile"].astype(int), 1, 10)
        gains = gains_df.groupby("decile").agg(bought_in_division_mean=("y","mean"), count=("y","size"), p_mean=("p","mean")).reset_index()
        gains_path = OUTPUTS_DIR / f"gains_{division.lower()}.csv"
        gains.to_csv(gains_path, index=False)
        artifacts[gains_path.name] = str(gains_path)

        # Calibration bins & MAE (and plot)
        try:
            calib = calibration_bins(y_final, p_final, n_bins=10)
            calib_path = OUTPUTS_DIR / f"calibration_{division.lower()}.csv"
            calib.to_csv(calib_path, index=False)
            artifacts[calib_path.name] = str(calib_path)
            cal_mae = calibration_mae(calib, weighted=True)
            # Also emit a PNG plot for quick viewing
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 4))
                x = calib['mean_predicted'].values
                y = calib['fraction_positives'].values
                ax.plot([0,1], [0,1], linestyle='--', color='#2ca02c', label='Perfect')
                ax.plot(x, y, marker='o', color='#1f77b4', label='Observed')
                ax.set_xlabel('Predicted probability')
                ax.set_ylabel('Observed rate')
                ax.set_title(f'Calibration Curve - {division} (@ {last_cut})')
                ax.legend(loc='best')
                fig.tight_layout()
                png_path = OUTPUTS_DIR / f"calibration_plot_{division.lower()}.png"
                fig.savefig(png_path)
                plt.close(fig)
                artifacts[png_path.name] = str(png_path)
            except Exception:
                pass
        except Exception:
            cal_mae = None

        # Thresholds for top-K percents
        thr_rows = []
        topk_rows = []
        for k in cfg.modeling.top_k_percents:
            if p_final is None or len(p_final) == 0:
                continue
            thr = compute_topk_threshold(p_final, k)
            cutoff_idx = max(1, int(len(p_final) * (k/100.0)))
            thr_rows.append({"k_percent": k, "threshold": float(thr), "count": cutoff_idx})
            # Top-K yield and capture
            order = np.argsort(-p_final)
            top_idx = order[:cutoff_idx]
            pos_rate = float(np.mean(y_final[top_idx])) if cutoff_idx > 0 else None
            total_pos = float(np.sum(y_final)) if len(y_final) else 0.0
            capture = float(np.sum(y_final[top_idx]) / total_pos) if total_pos > 0 else None
            topk_rows.append({
                "k_percent": int(k),
                "count": int(cutoff_idx),
                "pos_rate": pos_rate,
                "capture": capture,
                "threshold": float(thr),
            })
        thr_path = OUTPUTS_DIR / f"thresholds_{division.lower()}.csv"
        pd.DataFrame(thr_rows).to_csv(thr_path, index=False)
        artifacts[thr_path.name] = str(thr_path)

        # LR coefficients (if available)
        try:
            # The calibrated model may wrap a Pipeline. Extract underlying LogisticRegression if present.
            base = getattr(model, "base_estimator", None)
            if base is None and hasattr(model, "estimator"):
                base = model.estimator
            # Unwrap Pipeline to its 'model' step if necessary
            try:
                from sklearn.pipeline import Pipeline as _SkPipeline  # local import to avoid top-level noise
                if isinstance(base, _SkPipeline) and 'model' in getattr(base, 'named_steps', {}):
                    base = base.named_steps['model']
            except Exception:
                pass
            if isinstance(base, LogisticRegression) and hasattr(base, "coef_"):
                coef = pd.DataFrame({"feature": feature_names, "coef": base.coef_.ravel().tolist()})
                coef.to_csv(OUTPUTS_DIR / f"coef_{division.lower()}.csv", index=False)
        except Exception:
            pass

        artifacts.update(
            _maybe_export_shap(
                model,
                X_final,
                df_final,
                division,
                feature_names,
                shap_sample,
                cfg.modeling.shap_max_rows,
                cfg.modeling.seed,
            )
        )

        # Write/update metadata.json with final metrics
        try:
            if _metadata is not None:
                _metadata["best_auc"] = float(auc_val) if auc_val is not None else None
                _metadata["calibration_mae"] = float(cal_mae) if 'cal_mae' in locals() and cal_mae is not None else None
                _metadata["brier_score"] = float(brier) if brier is not None else None
                with open(out_dir / "metadata.json", "w", encoding="utf-8") as mf:
                    json.dump(_metadata, mf, indent=2)
                artifacts["metadata.json"] = str(out_dir / "metadata.json")
        except Exception as _e:
            logger.warning(f"Failed to write metadata.json: {_e}")

        # Model card / metrics
        # Model card / metrics
        agg_records = agg.replace({np.nan: None}).to_dict(orient='records') if len(agg) else []
        stability_export["final_rev_lift"] = _json_safe({f"@{k}": weighted_lifts.get(f"rev_lift@{k}") for k in cfg.modeling.top_k_percents})
        stability_export["final_lift"] = _json_safe({f"@{k}": lifts.get(f"lift@{k}") for k in cfg.modeling.top_k_percents})
        stability_export["final_prevalence"] = _json_safe(prevalence_final)
        stability_export["holdout_cutoff"] = str(last_cut)

        metrics = {
            "division": division,
            "cutoffs": cut_list,
            "selection": winner,
            "aggregate": agg_records,
            "final": {
                "auc": float(auc_val) if auc_val is not None else None,
                "pr_auc": float(pr_auc) if pr_auc is not None else None,
                "brier": float(brier) if brier is not None else None,
                "cal_mae": float(cal_mae) if cal_mae is not None else None,
                **{k: _json_safe(v) for k, v in lifts.items()},
                **{k: _json_safe(v) for k, v in weighted_lifts.items()},
            },
            "seed": cfg.modeling.seed,
            "window_months": int(window_months),
            "stability": stability_export,
        }
        metrics_path = OUTPUTS_DIR / f"metrics_{division.lower()}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        artifacts[metrics_path.name] = str(metrics_path)

        # Model card JSON
        # Derive a human-friendly calibration method label
        try:
            cal_method_label = 'isotonic' if final_cal_method == 'isotonic' else 'platt'
        except Exception:
            cal_method_label = None

        card = {
            "division": division,
            "cutoffs": cut_list,
            "window_months": int(window_months),
            "selected_model": winner,
            "seed": int(cfg.modeling.seed),
            "params": {
                "lr_grid": cfg.modeling.lr_grid,
                "lgbm_grid": cfg.modeling.lgbm_grid,
            },
            "data": {
                "n_customers": int(len(y_final)),
                "prevalence": float(np.mean(y_final)) if len(y_final) > 0 else None,
            },
            "calibration": {"method": cal_method_label, "mae_weighted": float(cal_mae) if cal_mae is not None else None},
            "topk": topk_rows,
            "stability": stability_export,
            "artifacts": {
                "model_pickle": str(out_dir / "model.pkl"),
                "feature_list": str(out_dir / "feature_list.json"),
                "metrics_json": str(OUTPUTS_DIR / f"metrics_{division.lower()}.json"),
                "gains_csv": str(OUTPUTS_DIR / f"gains_{division.lower()}.csv"),
                "calibration_csv": str(OUTPUTS_DIR / f"calibration_{division.lower()}.csv"),
                "thresholds_csv": str(OUTPUTS_DIR / f"thresholds_{division.lower()}.csv"),
            },
        }
        model_card = OUTPUTS_DIR / f"model_card_{division.lower()}.json"
        with open(model_card, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
        artifacts[model_card.name] = str(model_card)
    except Exception as e:
        logger.warning(f"Failed writing Phase 3 artifacts: {e}")

    # Emit diagnostics summary (feature pruning collected on last iteration)
    try:
        diag_ctx = {
            "division": division,
            "cutoffs": cut_list,
            "selected_model": winner,
            "dropped_low_variance_cols": list(dict.fromkeys(all_dropped_low_var)),
            "dropped_high_corr_pairs": all_dropped_corr,
            "results_grid": results,
            "coverage_drops": {k: v for k, v in coverage_drops_by_cutoff.items()},
            "sparse_family_drops": {k: v for k, v in family_drops_by_cutoff.items()},
            "prevalence": prevalence_stats,
            "stability": stability_export,
        }
        _emit_diagnostics(OUTPUTS_DIR, division, diag_ctx)
        # If any LR result shows non-convergence, log a warning
        if any((r.get('model') == 'logreg' and r.get('converged') is False) for r in results):
            logger.warning("Logistic Regression did not fully converge for some grid settings. See diagnostics JSON for details.")
    except Exception:
        pass

    logger.info(f"Training complete for {division}")
    try:
        ctx["write_manifest"](artifacts)
        ctx["append_registry"]({"phase": "phase3_train", "division": division, "cutoffs": cut_list, "artifact_count": len(artifacts)})
    except Exception:
        pass


if __name__ == "__main__":
    main()




