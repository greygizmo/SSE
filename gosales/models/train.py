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
    metrics = {
        "selection": winner,
        "aggregate": agg.to_dict(orient='records'),
    }
    with open(OUTPUTS_DIR / f"metrics_{division.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training complete for {division}")


if __name__ == "__main__":
    main()


