from __future__ import annotations

import json
from pathlib import Path
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

def _calibrate(clf, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, method: str):
    calibrated = CalibratedClassifierCV(estimator=clf, method="sigmoid" if method == "platt" else "isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


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
            y = df['bought_in_division'].astype(int).values
            X = df.drop(columns=['customer_id','bought_in_division'])
            X = _sanitize_features(X)
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
                # Calibration on the entire pipeline
                best_cal = None
                best_brier = 1e9
                for m in cal_methods:
                    cal = _calibrate(best_lr_pipe, X_train, y_train, X_valid, m)
                    p = cal.predict_proba(X_valid)[:, 1]
                    brier = brier_score_loss(y_valid, p)
                    if brier < best_brier:
                        best_brier = brier
                        best_cal = cal
                p = best_cal.predict_proba(X_valid)[:, 1]
                lift10 = _lift_at_k(y_valid, p, 10)
                # Pull n_iter_ from underlying LR if available
                try:
                    lr_inner = best_lr_pipe.named_steps.get('model')
                    n_iter_val = getattr(lr_inner, 'n_iter_', None)
                    if isinstance(n_iter_val, (list, np.ndarray)):
                        n_iter_val = [int(x) for x in np.ravel(n_iter_val).tolist()]
                except Exception:
                    n_iter_val = None
                results.append({
                    "cutoff": cutoff, "model": "logreg", "auc": float(best_lr_auc),
                    "lift10": float(lift10), "brier": float(best_brier), "calibration": best_cal.method,
                    "converged": bool(best_conv), "n_iter": n_iter_val
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
                # Calibration
                best_cal = None
                best_brier = 1e9
                for m in cal_methods:
                    cal = _calibrate(best_lgbm, X_train, y_train, X_valid, m)
                    p = cal.predict_proba(X_valid)[:,1]
                    brier = brier_score_loss(y_valid, p)
                    if brier < best_brier:
                        best_brier = brier
                        best_cal = cal
                p = best_cal.predict_proba(X_valid)[:,1]
                lift10 = _lift_at_k(y_valid, p, 10)
                results.append({"cutoff": cutoff, "model": "lgbm", "auc": float(best_auc), "lift10": float(lift10), "brier": float(best_brier), "calibration": best_cal.method})

    # Aggregate and select winner by lift@10 then cal-Brier
    if not results:
        logger.warning("No training results produced")
        return
    res_df = pd.DataFrame(results)
    agg = res_df.groupby('model').agg({"lift10":"mean", "brier":"mean", "auc":"mean"}).reset_index()
    agg = agg.sort_values(['lift10', 'brier'], ascending=[False, True])
    winner = agg.iloc[0]['model']
    logger.info(f"Selected model: {winner} by mean lift@10 and brier across cutoffs")

    # Train final on last cutoff for simplicity here
    last_cut = cut_list[-1]
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
    X_final = df_final.drop(columns=['customer_id','bought_in_division'])
    X_final = _sanitize_features(X_final)
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
        # Calibrate entire pipeline using selected method (prefer isotonic)
        cal = CalibratedClassifierCV(pipe, method='isotonic' if final_cal_method == 'isotonic' else 'sigmoid', cv=3).fit(X_final, y_final)
        model = cal
        feature_names = list(X_final.columns)
    else:
        clf = LGBMClassifier(random_state=cfg.modeling.seed, n_estimators=400, learning_rate=0.05, deterministic=True, n_jobs=1)
        clf.fit(X_final, y_final)
        cal = CalibratedClassifierCV(clf, method='isotonic' if final_cal_method == 'isotonic' else 'sigmoid', cv=3).fit(X_final, y_final)
        model = cal
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
    # Final predictions and guardrails
    try:
        p_final = model.predict_proba(X_final)[:,1]
        if float(np.std(p_final)) < 0.01:
            logger.warning("Degenerate classifier (std(p) < 0.01). Aborting artifact write.")
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
        # Revenue-weighted lift using best available weight feature
        weights = np.ones_like(y_final, dtype=float)
        try:
            if 'rfm__all__gp_sum__12m' in df_final.columns:
                weights = pd.to_numeric(df_final['rfm__all__gp_sum__12m'], errors='coerce').fillna(0.0).values
            elif 'total_gp_all_time' in df_final.columns:
                weights = pd.to_numeric(df_final['total_gp_all_time'], errors='coerce').fillna(0.0).values
        except Exception:
            pass
        weighted_lifts = {f"rev_lift@{k}": _weighted_lift_at_k(y_final, p_final, weights, k) for k in cfg.modeling.top_k_percents} if p_final is not None else {}

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

        # Model card / metrics
        # Model card / metrics
        metrics = {
            "division": division,
            "cutoffs": cut_list,
            "selection": winner,
            "aggregate": agg.to_dict(orient='records'),
            "final": {
                "auc": float(auc_val) if auc_val is not None else None,
                "pr_auc": float(pr_auc) if pr_auc is not None else None,
                "brier": float(brier) if brier is not None else None,
                "cal_mae": float(cal_mae) if cal_mae is not None else None,
                **lifts,
                **weighted_lifts,
            },
            "seed": cfg.modeling.seed,
            "window_months": int(window_months),
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




