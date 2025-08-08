from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

from gosales.utils.config import load_config
from gosales.utils.db import get_db_connection
from gosales.features.engine import create_feature_matrix
from gosales.utils.paths import OUTPUTS_DIR, MODELS_DIR
from gosales.utils.logger import get_logger


logger = get_logger(__name__)


def _lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k_percent: int) -> float:
    n = len(y_true)
    k = max(1, int(n * (k_percent / 100.0)))
    idx = np.argsort(-y_score)[:k]
    topk_rate = float(np.mean(y_true[idx]))
    base_rate = float(np.mean(y_true)) if np.mean(y_true) > 0 else 1e-9
    return topk_rate / base_rate


def _train_test_split_time_aware(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Simple stratified split for now; can be upgraded to time-aware using order_date if available
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def _calibrate(clf, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, method: str):
    calibrated = CalibratedClassifierCV(base_estimator=clf, method="sigmoid" if method == "platt" else "isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


@click.command()
@click.option("--division", required=True)
@click.option("--cutoffs", required=True, help="comma-separated cutoffs")
@click.option("--window-months", default=6, type=int)
@click.option("--models", default="logreg,lgbm")
@click.option("--calibration", default="platt,isotonic")
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
def main(division: str, cutoffs: str, window_months: int, models: str, calibration: str, config: str) -> None:
    cfg = load_config(config)
    cut_list = [c.strip() for c in cutoffs.split(",") if c.strip()]
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_db_connection()

    # Accumulate metrics across cutoffs per model
    model_names = [m.strip() for m in models.split(",") if m.strip()]
    cal_methods = [c.strip() for c in calibration.split(",") if c.strip()]

    results = []
    for cutoff in cut_list:
        fm = create_feature_matrix(engine, division, cutoff, window_months)
        if fm.is_empty():
            logger.warning(f"Empty feature matrix for cutoff {cutoff}")
            continue
        df = fm.to_pandas()
        y = df['bought_in_division'].astype(int).values
        X = df.drop(columns=['customer_id','bought_in_division'])
        X_train, X_valid, y_train, y_valid = _train_test_split_time_aware(X, y, cfg.modeling.seed)

        # Baseline LR (elastic-net via saga)
        if 'logreg' in model_names:
            scaler = StandardScaler(with_mean=False)
            X_train_lr = scaler.fit_transform(X_train)
            X_valid_lr = scaler.transform(X_valid)
            lr_params = cfg.modeling.lr_grid
            best_lr = None
            best_lr_auc = -1
            for l1_ratio in lr_params.get('l1_ratio', [0.0, 0.5]):
                for C in lr_params.get('C', [1.0]):
                    lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=l1_ratio, C=C, max_iter=2000, class_weight='balanced', random_state=cfg.modeling.seed)
                    lr.fit(X_train_lr, y_train)
                    p = lr.predict_proba(X_valid_lr)[:,1]
                    auc_lr = roc_auc_score(y_valid, p)
                    if auc_lr > best_lr_auc:
                        best_lr_auc = auc_lr
                        best_lr = (lr, scaler)
            if best_lr is not None:
                lr, scaler = best_lr
                # Calibration
                best_cal = None
                best_brier = 1e9
                for m in cal_methods:
                    cal = _calibrate(lr, X_train_lr, y_train, X_valid_lr, m)
                    p = cal.predict_proba(X_valid_lr)[:,1]
                    brier = brier_score_loss(y_valid, p)
                    if brier < best_brier:
                        best_brier = brier
                        best_cal = cal
                p = best_cal.predict_proba(X_valid_lr)[:,1]
                lift10 = _lift_at_k(y_valid, p, 10)
                results.append({"cutoff": cutoff, "model": "logreg", "auc": float(best_lr_auc), "lift10": float(lift10), "brier": float(best_brier), "calibration": best_cal.method})

        # LGBM challenger
        if 'lgbm' in model_names:
            # scale_pos_weight
            pos = int(np.sum(y_train))
            neg = int(len(y_train) - pos)
            spw = (neg / max(1, pos))
            grid = cfg.modeling.lgbm_grid
            best_lgbm = None
            best_auc = -1
            for num_leaves in grid.get('num_leaves', [31]):
                for min_data_in_leaf in grid.get('min_data_in_leaf', [50]):
                    for learning_rate in grid.get('learning_rate', [0.05]):
                        for feature_fraction in grid.get('feature_fraction', [0.9]):
                            for bagging_fraction in grid.get('bagging_fraction', [0.9]):
                                clf = LGBMClassifier(
                                    random_state=cfg.modeling.seed,
                                    n_estimators=400,
                                    learning_rate=learning_rate,
                                    num_leaves=num_leaves,
                                    min_data_in_leaf=min_data_in_leaf,
                                    feature_fraction=feature_fraction,
                                    bagging_fraction=bagging_fraction,
                                    scale_pos_weight=spw,
                                )
                                clf.fit(X_train, y_train)
                                p = clf.predict_proba(X_valid)[:,1]
                                auc_lgbm = roc_auc_score(y_valid, p)
                                if auc_lgbm > best_auc:
                                    best_auc = auc_lgbm
                                    best_lgbm = clf
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
    fm_final = create_feature_matrix(engine, division, last_cut, window_months)
    df_final = fm_final.to_pandas()
    y_final = df_final['bought_in_division'].astype(int).values
    X_final = df_final.drop(columns=['customer_id','bought_in_division'])
    # Minimal: refit winner without hyper search for brevity (could repeat best params)
    if winner == 'logreg':
        scaler = StandardScaler(with_mean=False)
        X_tr = scaler.fit_transform(X_final)
        lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.2, C=1.0, max_iter=2000, class_weight='balanced', random_state=cfg.modeling.seed)
        lr.fit(X_tr, y_final)
        # Calibrate
        cal = CalibratedClassifierCV(lr, method='sigmoid', cv=3).fit(X_tr, y_final)
        model = cal
        feature_names = list(X_final.columns)
    else:
        clf = LGBMClassifier(random_state=cfg.modeling.seed, n_estimators=400, learning_rate=0.05)
        clf.fit(X_final, y_final)
        cal = CalibratedClassifierCV(clf, method='sigmoid', cv=3).fit(X_final, y_final)
        model = cal
        feature_names = list(X_final.columns)

    # Save artifacts
    out_dir = MODELS_DIR / f"{division.lower()}_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save model pickle via joblib
    try:
        import joblib
        joblib.dump(model, out_dir / "model.pkl")
    except Exception:
        pass
    with open(out_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f)
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
        auc_val = roc_auc_score(y_final, p_final) if p_final is not None else None
        pr_prec, pr_rec, _ = precision_recall_curve(y_final, p_final) if p_final is not None else (None, None, None)
        pr_auc = auc(pr_rec, pr_prec) if pr_prec is not None else None
        brier = brier_score_loss(y_final, p_final) if p_final is not None else None
        lifts = {f"lift@{k}": _lift_at_k(y_final, p_final, k) for k in cfg.modeling.top_k_percents} if p_final is not None else {}

        # Gains (deciles)
        gains_df = pd.DataFrame({"y": y_final, "p": p_final})
        gains_df = gains_df.sort_values("p", ascending=False).reset_index(drop=True)
        gains_df["decile"] = (np.floor((gains_df.index / max(1, len(gains_df)-1)) * 10) + 1).clip(1, 10).astype(int)
        gains = gains_df.groupby("decile").agg(bought_in_division_mean=("y","mean"), count=("y","size"), p_mean=("p","mean")).reset_index()
        gains.to_csv(OUTPUTS_DIR / f"gains_{division.lower()}.csv", index=False)

        # Calibration bins (quantile bins)
        try:
            bins = pd.qcut(gains_df["p"], q=10, duplicates="drop")
            calib = gains_df.assign(bin=bins).groupby("bin").agg(mean_predicted=("p","mean"), fraction_positives=("y","mean"), count=("y","size")).reset_index()
            calib.to_csv(OUTPUTS_DIR / f"calibration_{division.lower()}.csv", index=False)
        except Exception:
            pass

        # Thresholds for top-K percents
        thr_rows = []
        for k in cfg.modeling.top_k_percents:
            if p_final is None or len(p_final) == 0:
                continue
            cutoff_idx = max(1, int(len(p_final) * (k/100.0)))
            thr = float(np.sort(p_final)[-cutoff_idx])
            thr_rows.append({"k_percent": k, "threshold": thr, "count": cutoff_idx})
        pd.DataFrame(thr_rows).to_csv(OUTPUTS_DIR / f"thresholds_{division.lower()}.csv", index=False)

        # LR coefficients (if available)
        try:
            from sklearn.linear_model import LogisticRegression
            base = getattr(model, "base_estimator", None)
            if base is None and hasattr(model, "estimator"):
                base = model.estimator
            if isinstance(base, LogisticRegression) and hasattr(base, "coef_"):
                coef = pd.DataFrame({"feature": feature_names, "coef": base.coef_.ravel().tolist()})
                coef.to_csv(OUTPUTS_DIR / f"coef_{division.lower()}.csv", index=False)
        except Exception:
            pass

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
                **lifts,
            },
            "seed": cfg.modeling.seed,
            "window_months": int(window_months),
        }
        with open(OUTPUTS_DIR / f"metrics_{division.lower()}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed writing Phase 3 artifacts: {e}")

    logger.info(f"Training complete for {division}")


if __name__ == "__main__":
    main()


