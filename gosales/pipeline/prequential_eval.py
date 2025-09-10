from __future__ import annotations

"""
Prequential forward-month evaluation.

Trains (or reuses) a model at a fixed training cutoff, then evaluates month-by-month
forward cutoffs, computing AUC, Lift@K, and Brier. Writes JSON, CSV, and PNG curves
under gosales/outputs/prequential/<division>/<train_cutoff>/.
"""

from pathlib import Path
import json
import pandas as pd
import argparse
from datetime import datetime

import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, brier_score_loss

from gosales.utils.paths import OUTPUTS_DIR, MODELS_DIR
from gosales.utils.config import load_config
from gosales.utils.db import get_curated_connection, get_db_connection, validate_connection
from gosales.features.engine import create_feature_matrix
from gosales.models.metrics import compute_lift_at_k


def _ensure_model(division: str, cutoff: str, window_months: int) -> Path:
    div_key = division.lower()
    out_dir = MODELS_DIR / f"{div_key}_model"
    pkl = out_dir / "model.pkl"
    if pkl.exists():
        return pkl
    # Train a model at the given cutoff
    import sys, subprocess
    cfg = load_config()
    cmd = [
        sys.executable, "-m", "gosales.models.train",
        "--division", division,
        "--cutoffs", cutoff,
        "--window-months", str(int(window_months)),
    ]
    # Use production-like mode (no SAFE flags) for forward evaluation
    subprocess.run(cmd, check=True)
    if not pkl.exists():
        raise RuntimeError("Model training did not produce a pickle.")
    return pkl


def _align_features(X: pd.DataFrame, division: str) -> pd.DataFrame:
    """Align columns to the training feature order, filling missing with 0.0 and dropping extras."""
    div_key = division.lower()
    flist = MODELS_DIR / f"{div_key}_model" / 'feature_list.json'
    if not flist.exists():
        return X
    try:
        names = json.loads(flist.read_text(encoding='utf-8'))
        cols = [c for c in names if c in X.columns]
        missing = [c for c in names if c not in X.columns]
        extra = [c for c in X.columns if c not in names]
        X2 = X.drop(columns=extra, errors='ignore').copy()
        for m in missing:
            X2[m] = 0.0
        # Reindex to exact order
        X2 = X2.reindex(columns=names)
        return X2
    except Exception:
        return X


def _sanitize_features_pd(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric floats, replace infs/NaNs with 0.0.

    Mirrors scoring pipeline's sanitization to prevent dtype errors at predict time.
    """
    Xc = X.copy()
    for c in Xc.columns:
        Xc[c] = pd.to_numeric(Xc[c], errors='coerce')
    Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
    return Xc.fillna(0.0)


def _month_ends(start: str, end: str) -> list[str]:
    # Accept YYYY-MM or YYYY-MM-DD; normalize to YYYY-MM for period_range
    def to_period(s: str) -> str:
        try:
            dt = pd.to_datetime(s)
            return dt.strftime("%Y-%m")
        except Exception:
            return s
    p_start = to_period(start)
    p_end = to_period(end)
    periods = pd.period_range(start=p_start, end=p_end, freq='M')
    dates = [pd.Period(p, freq='M').end_time.date().isoformat() for p in periods]
    return dates


def run_prequential(division: str, train_cutoff: str, start: str, end: str, window_months: int, k_percent: int = 10) -> dict:
    # Outputs dir
    out_dir = OUTPUTS_DIR / 'prequential' / division / train_cutoff
    out_dir.mkdir(parents=True, exist_ok=True)

    # DB engine
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()
    validate_connection(engine)

    # Ensure model exists
    model_path = _ensure_model(division, train_cutoff, window_months)
    model = joblib.load(model_path)

    # Build list of monthly cutoffs
    months = _month_ends(start, end)
    # Clamp to months where labels are fully observable: (cutoff + window_months) <= today
    try:
        today = pd.Timestamp.utcnow().normalize()
    except Exception:
        today = pd.Timestamp.today().normalize()
    # Ensure tz-naive for safe comparisons
    try:
        if getattr(today, 'tzinfo', None) is not None:
            today = today.tz_localize(None)
    except Exception:
        pass
    kept = []
    for cut in months:
        try:
            cut_dt = pd.to_datetime(cut)
            pred_end = cut_dt + pd.DateOffset(months=int(window_months))
            if pred_end <= today:
                kept.append(cut)
        except Exception:
            continue
    months = kept

    # Evaluate per month
    rows = []
    for cut in months:
        fm = create_feature_matrix(engine, division, cut, window_months)
        if fm.is_empty():
            rows.append({
                'cutoff': cut,
                'n': 0,
                'prevalence': None,
                'auc': None,
                'lift@10': None,
                'brier': None,
            })
            continue
        df = fm.to_pandas()
        y = df['bought_in_division'].astype(int).values
        X = df.drop(columns=['customer_id','bought_in_division'])
        try:
            X_aligned = _align_features(X, division)
            X_aligned = _sanitize_features_pd(X_aligned)
            p = model.predict_proba(X_aligned)[:,1]
        except Exception:
            # Try unwrap calibrated estimator
            base = getattr(model, 'base_estimator', None)
            if base is None and hasattr(model, 'estimator'):
                base = model.estimator
            m = base if base is not None else model
            X_aligned = _align_features(X, division)
            X_aligned = _sanitize_features_pd(X_aligned)
            p = m.predict_proba(X_aligned)[:,1]
        auc = float(roc_auc_score(y, p)) if np.any(y) and not np.all(y == 1) else None
        lift10 = float(compute_lift_at_k(y, p, k_percent)) if len(y) > 0 else None
        brier = float(brier_score_loss(y, p)) if len(y) > 0 else None
        rows.append({
            'cutoff': cut,
            'n': int(len(y)),
            'prevalence': float(np.mean(y)) if len(y) else None,
            'auc': auc,
            'lift@10': lift10,
            'brier': brier,
        })

    # Write JSON/CSV
    js = out_dir / f'prequential_{division}_{train_cutoff}.json'
    js.write_text(json.dumps({'division': division, 'train_cutoff': train_cutoff, 'window_months': int(window_months), 'k_percent': int(k_percent), 'results': rows}, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(out_dir / f'prequential_{division}_{train_cutoff}.csv', index=False)

    # Plot curves
    try:
        import matplotlib.pyplot as plt
        dfc = pd.DataFrame(rows)
        if not dfc.empty:
            dfc['t'] = pd.to_datetime(dfc['cutoff'])
            dfc = dfc.sort_values('t')
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax1.plot(dfc['t'], dfc['auc'], label='AUC', color='#4C78A8')
            ax1.set_ylabel('AUC', color='#4C78A8')
            ax1.tick_params(axis='y', labelcolor='#4C78A8')
            ax2 = ax1.twinx()
            ax2.plot(dfc['t'], dfc['lift@10'], label='Lift@10', color='#F58518')
            ax2.plot(dfc['t'], dfc['brier'], label='Brier', color='#54A24B')
            ax2.set_ylabel('Lift@10 / Brier', color='#333333')
            ax1.set_xlabel('Cutoff Month')
            ax1.set_title(f'Prequential Curves ({division}) – Train @ {train_cutoff}')
            fig.autofmt_xdate()
            fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
            fig.tight_layout()
            fig.savefig(out_dir / f'prequential_curves_{division}_{train_cutoff}.png')
            plt.close(fig)
    except Exception:
        pass

    return {'prequential_json': str(js)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--division', required=True)
    ap.add_argument('--train-cutoff', default='2024-06-30')
    ap.add_argument('--start', default='2025-01')
    ap.add_argument('--end', default='2025-12')
    ap.add_argument('--window-months', type=int, default=6)
    ap.add_argument('--k-percent', type=int, default=10)
    args = ap.parse_args()
    run_prequential(args.division, args.train_cutoff, args.start, args.end, args.window_months, k_percent=args.k_percent)


if __name__ == '__main__':
    main()

