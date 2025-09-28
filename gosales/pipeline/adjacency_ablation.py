"""Run the SAFE adjacency ablation experiment for a single division.

This script trains multiple feature subsets at a historical cutoff and evaluates
them on a holdout month to quantify how aggressive SAFE policies impact model
quality. Four variants are compared:

1. ``full`` – the untouched feature set
2. ``no_recency_short`` – removes explicit recency and short window aggregates
3. ``safe`` – mimics the full SAFE policy by excluding adjacency-heavy families
4. ``safe_lite`` – a compromise that keeps embeddings but drops near-cutoff cues

For each variant the script trains a light-weight model (logistic regression and
LightGBM, picking whichever scores best on the holdout AUC) and records AUC,
Lift@10, and Brier Score. Outputs land under
``gosales/outputs/ablation/adjacency/<division>/<train>_<holdout>/`` and include a
JSON summary plus a CSV table. These artifacts drive SAFE decision meetings and
feed ``auto_safe_from_ablation`` which auto-updates the configuration.
"""

from __future__ import annotations

from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier

from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.config import load_config
from gosales.utils.db import get_curated_connection, get_db_connection
from gosales.features.engine import create_feature_matrix
from gosales.models.metrics import compute_lift_at_k


def _drop_noop(X: pd.DataFrame) -> pd.DataFrame:
    return X


def _drop_safe(X: pd.DataFrame) -> pd.DataFrame:
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
        if s in ('gp_2024','gp_2023'):
            continue
        if s.startswith('xdiv__div__gp_share__'):
            continue
        if s.startswith('sku_gp_12m_') or s.startswith('sku_qty_12m_') or s.startswith('sku_gp_per_unit_12m_'):
            continue
        cols.append(c)
    return X[cols] if cols else X


def _drop_no_recency_short(X: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in X.columns:
        s = str(c).lower()
        if 'days_since_last' in s or 'recency' in s:
            continue
        if '__3m' in s or s.endswith('_last_3m') or '__6m' in s or s.endswith('_last_6m') or '__12m' in s or s.endswith('_last_12m'):
            continue
        cols.append(c)
    return X[cols] if cols else X


def _drop_safe_lite(X: pd.DataFrame) -> pd.DataFrame:
    """SAFE‑lite: drop clear adjacency/near-boundary families but retain embeddings and longer-term aggregates.

    Removes:
      - expiring assets windows and subs share ratios
      - explicit recency/days_since_last
      - short windows (<=12m)
    Keeps ALS embeddings and 12m+ aggregates otherwise.
    """
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
        cols.append(c)
    return X[cols] if cols else X

def _train_and_eval(df_train: pd.DataFrame, df_hold: pd.DataFrame) -> dict:
    y_tr = df_train['bought_in_division'].astype(int).values
    X_tr = df_train.drop(columns=['customer_id','bought_in_division'])
    y_ho = df_hold['bought_in_division'].astype(int).values
    X_ho = df_hold.drop(columns=['customer_id','bought_in_division'])

    # Two simple models: LR and LGBM; choose better by holdout AUC
    models = []
    try:
        lr = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('lr', LogisticRegression(max_iter=5000, solver='liblinear', class_weight='balanced')),
        ])
        lr.fit(X_tr, y_tr)
        p_lr = lr.predict_proba(X_ho)[:,1]
        auc_lr = float(roc_auc_score(y_ho, p_lr)) if np.any(y_ho) and not np.all(y_ho == 1) else -1
        models.append(('logreg', auc_lr, p_lr))
    except Exception:
        pass
    try:
        lgbm = LGBMClassifier(random_state=42, n_estimators=400, learning_rate=0.05, deterministic=True, n_jobs=1)
        lgbm.fit(X_tr, y_tr)
        p_lgbm = lgbm.predict_proba(X_ho)[:,1]
        auc_lgbm = float(roc_auc_score(y_ho, p_lgbm)) if np.any(y_ho) and not np.all(y_ho == 1) else -1
        models.append(('lgbm', auc_lgbm, p_lgbm))
    except Exception:
        pass

    if not models:
        raise RuntimeError('No model trained')
    best = max(models, key=lambda t: t[1])
    p = best[2]
    auc = float(roc_auc_score(y_ho, p)) if np.any(y_ho) and not np.all(y_ho == 1) else None
    lift10 = float(compute_lift_at_k(y_ho, p, 10)) if len(y_ho) > 0 else None
    brier = float(brier_score_loss(y_ho, p)) if len(y_ho) > 0 else None
    return {'model': best[0], 'auc': auc, 'lift@10': lift10, 'brier': brier}


def run_ablation(division: str, train_cutoff: str, holdout_cutoff: str, window_months: int) -> dict:
    out_dir = OUTPUTS_DIR / 'ablation' / 'adjacency' / division / f"{train_cutoff}_{holdout_cutoff}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()

    # Build train/holdout frames
    fm_tr = create_feature_matrix(engine, division, train_cutoff, window_months)
    fm_ho = create_feature_matrix(engine, division, holdout_cutoff, window_months)
    df_tr = fm_tr.to_pandas(); df_ho = fm_ho.to_pandas()

    # Variants
    variants = {
        'full': (df_tr.drop(columns=['customer_id','bought_in_division'], errors='ignore'), _drop_noop),
        'no_recency_short': (None, _drop_no_recency_short),
        'safe': (None, _drop_safe),
        'safe_lite': (None, _drop_safe_lite),
    }

    results = {}
    rows = []
    for name, (preset_X, dropper) in variants.items():
        Xtr = preset_X if preset_X is not None else df_tr.drop(columns=['customer_id','bought_in_division'], errors='ignore')
        Xho = df_ho.drop(columns=['customer_id','bought_in_division'], errors='ignore')
        Xtr2 = dropper(Xtr)
        # align holdout to same columns
        keep_cols = list(Xtr2.columns)
        Xho2 = Xho.copy()
        missing = [c for c in keep_cols if c not in Xho2.columns]
        for m in missing:
            Xho2[m] = 0.0
        Xho2 = Xho2[keep_cols].copy()
        # Re-attach IDs/labels for function
        dtr = pd.concat([df_tr[['customer_id','bought_in_division']], Xtr2], axis=1)
        dho = pd.concat([df_ho[['customer_id','bought_in_division']], Xho2], axis=1)
        res = _train_and_eval(dtr, dho)
        results[name] = res
        rows.append({'variant': name, **res})

    # Compare Full vs SAFE on holdout
    try:
        delta_auc = None
        if results.get('full', {}).get('auc') is not None and results.get('safe', {}).get('auc') is not None:
            delta_auc = float(results['full']['auc'] - results['safe']['auc'])
    except Exception:
        delta_auc = None

    out = {
        'division': division,
        'train_cutoff': train_cutoff,
        'holdout_cutoff': holdout_cutoff,
        'window_months': int(window_months),
        'results': results,
        'delta_auc_full_minus_safe': delta_auc,
    }
    js = out_dir / f'adjacency_ablation_{division}_{train_cutoff}_{holdout_cutoff}.json'
    js.write_text(json.dumps(out, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(out_dir / f'adjacency_ablation_{division}_{train_cutoff}_{holdout_cutoff}.csv', index=False)
    return {'ablation_json': str(js)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--division', required=True)
    ap.add_argument('--train-cutoff', required=True)
    ap.add_argument('--holdout-cutoff', required=True)
    ap.add_argument('--window-months', type=int, default=6)
    args = ap.parse_args()
    run_ablation(args.division, args.train_cutoff, args.holdout_cutoff, args.window_months)


if __name__ == '__main__':
    main()


