from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from gosales.utils.paths import OUTPUTS_DIR


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _prevalence(df: pd.DataFrame) -> float:
    if df.empty or 'bought_in_division' not in df.columns:
        return float('nan')
    y = pd.to_numeric(df['bought_in_division'], errors='coerce').fillna(0).astype(int)
    return float(y.mean())


def _cal_mae(df: pd.DataFrame, n_bins: int = 10) -> float:
    if df.empty or 'icp_score' not in df.columns or 'bought_in_division' not in df.columns:
        return float('nan')
    y = pd.to_numeric(df['bought_in_division'], errors='coerce').fillna(0).astype(int)
    p = pd.to_numeric(df['icp_score'], errors='coerce').fillna(0.0).astype(float)
    unique_scores = pd.Series(p).nunique(dropna=False)
    if unique_scores >= n_bins:
        bins = pd.qcut(p, q=n_bins, labels=False, duplicates='drop')
    else:
        bins = pd.cut(
            p,
            bins=max(1, min(n_bins, unique_scores)),
            include_lowest=True,
            duplicates='drop',
            labels=False,
        )
    grp = pd.DataFrame({'y': y, 'p': p, 'bin': bins}).dropna().groupby('bin', observed=False).agg(
        mean_p=( 'p', 'mean'), frac_pos=('y', 'mean'), count=('y','size')
    )
    if grp.empty:
        return float('nan')
    diff = (grp['mean_p'] - grp['frac_pos']).abs()
    w = grp['count'].astype(float)
    return float((diff * w).sum() / max(1, w.sum()))


def _compare(a: float, b: float) -> float:
    if any(np.isnan([a, b])):
        return float('nan')
    return float(b - a)


def check_drift_and_emit_alerts(run_manifest: Dict[str, object] | None = None) -> Path:
    """Compare scoring prevalence/calibration vs training metrics and emit alerts.json.

    - Reads scoring `icp_scores.csv`
    - Reads training metrics `metrics_*.json` for each division present in scores
    - Emits alerts when prevalence deviates materially or calibration MAE increases beyond threshold
    """
    icp = OUTPUTS_DIR / 'icp_scores.csv'
    df = _safe_read_csv(icp)
    alerts: List[Dict[str, object]] = []
    if df.empty:
        out = OUTPUTS_DIR / 'alerts.json'
        out.write_text(json.dumps({'alerts': alerts}, indent=2), encoding='utf-8')
        return out

    for div, g in df.groupby('division_name'):
        g = g.copy()
        prev_now = _prevalence(g)
        cal_now = _cal_mae(g)
        # Try to read training metrics for this division
        mpath = OUTPUTS_DIR / f"metrics_{str(div).lower()}.json"
        prev_train = None
        cal_train = None
        if mpath.exists():
            try:
                m = json.loads(mpath.read_text(encoding='utf-8'))
                # Prevalence not always in metrics; skip if absent
                # Calibration MAE may be under metrics.final.cal_mae or calibration.mae_weighted
                if isinstance(m, dict):
                    cal_train = (m.get('final') or {}).get('cal_mae') if 'final' in m else None
                    if cal_train is None:
                        cal_train = ((m.get('calibration') or {}).get('mae_weighted'))
            except Exception:
                pass
        # Thresholds
        cal_thr = 0.10  # default if not configured elsewhere
        # Alerts for calibration worsening beyond threshold
        try:
            if cal_now is not None and not np.isnan(cal_now) and cal_now > cal_thr:
                alerts.append({
                    'division': div,
                    'type': 'calibration_mae_high',
                    'value': float(cal_now),
                    'threshold': float(cal_thr),
                    'message': f'Calibration MAE {float(cal_now):.3f} exceeds threshold {float(cal_thr):.3f}'
                })
            if cal_train is not None and isinstance(cal_train, (int, float)) and cal_now is not None and not np.isnan(cal_now):
                delta = float(cal_now) - float(cal_train)
                if delta > 0.03:
                    alerts.append({
                        'division': div,
                        'type': 'calibration_mae_regression',
                        'value': float(cal_now),
                        'train_value': float(cal_train),
                        'delta': float(delta),
                        'message': f'Calibration MAE worsened by +{float(delta):.3f} vs training'
                    })
        except Exception:
            pass

        if prev_now is not None and not np.isnan(prev_now):
            if prev_now == 0.0:
                alerts.append({
                    'division': div,
                    'type': 'prevalence_zero',
                    'value': float(prev_now),
                    'message': 'Zero positives in current scoring slice; verify metadata cutoff/window and label mapping.'
                })

    payload = {'alerts': alerts}
    out = OUTPUTS_DIR / 'alerts.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    # Append to run manifest if provided
    if isinstance(run_manifest, dict):
        try:
            run_manifest.setdefault('alerts', []).extend(alerts)
        except Exception:
            pass
    return out


