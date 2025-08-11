from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.config import load_config


def discover_validation_runs() -> List[Tuple[str, str, Path]]:
    base = OUTPUTS_DIR / 'validation'
    if not base.exists():
        return []
    rows: List[Tuple[str, str, Path]] = []
    for div_dir in base.iterdir():
        if not div_dir.is_dir():
            continue
        for cut_dir in div_dir.iterdir():
            if cut_dir.is_dir():
                rows.append((div_dir.name, cut_dir.name, cut_dir))
    return rows


def load_alerts(run_dir: Path) -> List[Dict[str, object]]:
    alerts_path = run_dir / 'alerts.json'
    if alerts_path.exists():
        try:
            payload = json.loads(alerts_path.read_text(encoding='utf-8'))
            return list(payload.get('alerts', []))
        except Exception:
            return []
    return []


def load_thresholds() -> Dict[str, float]:
    cfg = load_config()
    thr = {
        'psi_threshold': float(getattr(cfg.validation, 'psi_threshold', 0.25)),
        'ks_threshold': float(getattr(cfg.validation, 'ks_threshold', 0.15)),
        'cal_mae_threshold': float(getattr(cfg.validation, 'cal_mae_threshold', 0.03)),
    }
    return thr


def compute_validation_badges(run_dir: Path, thresholds: Dict[str, float] | None = None) -> Dict[str, Dict[str, object]]:
    thr = thresholds or load_thresholds()
    # Defaults
    out: Dict[str, Dict[str, object]] = {
        'cal_mae': {'value': None, 'threshold': thr['cal_mae_threshold'], 'status': 'unknown'},
        'psi_ev_vs_gp': {'value': None, 'threshold': thr['psi_threshold'], 'status': 'unknown'},
        'ks_phat_train_holdout': {'value': None, 'threshold': thr['ks_threshold'], 'status': 'unknown'},
    }
    # metrics.json
    metrics_path = run_dir / 'metrics.json'
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            cal_mae = None
            if isinstance(metrics, dict):
                if 'metrics' in metrics and isinstance(metrics['metrics'], dict):
                    cal_mae = metrics['metrics'].get('cal_mae', None)
            if cal_mae is not None:
                v = float(cal_mae)
                out['cal_mae']['value'] = v
                out['cal_mae']['status'] = 'ok' if v < thr['cal_mae_threshold'] else 'alert'
        except Exception:
            pass
    # drift.json
    drift_path = run_dir / 'drift.json'
    if drift_path.exists():
        try:
            drift = json.loads(drift_path.read_text(encoding='utf-8'))
            # Prefer the clearer key name; fallback to legacy if present
            psi_ev = drift.get('psi_holdout_ev_vs_holdout_gp', drift.get('psi_ev_vs_holdout_gp', None))
            ks_th = drift.get('ks_phat_train_holdout', None)
            if psi_ev is not None:
                v = float(psi_ev)
                out['psi_ev_vs_gp']['value'] = v
                out['psi_ev_vs_gp']['status'] = 'ok' if v < thr['psi_threshold'] else 'alert'
            if ks_th is not None:
                v = float(ks_th)
                out['ks_phat_train_holdout']['value'] = v
                out['ks_phat_train_holdout']['status'] = 'ok' if v < thr['ks_threshold'] else 'alert'
        except Exception:
            pass
    return out


def compute_default_validation_index(runs: List[Tuple[str, str, Path]], preferred: Optional[Dict[str, str]] = None) -> int:
    """Select a default run index.

    Preference order:
    1) If preferred {division, cutoff} matches, return its index
    2) Otherwise, pick the run with the latest (max) cutoff (ISO date string)
    3) Fallback to 0
    """
    if not runs:
        return 0
    if isinstance(preferred, dict):
        try:
            for i, (div, cut, _) in enumerate(runs):
                if div == preferred.get("division") and cut == preferred.get("cutoff"):
                    return i
        except Exception:
            pass
    try:
        # Choose by max cutoff (ISO date string comparison works)
        max_i, _ = max(enumerate(runs), key=lambda t: t[1][1])
        return int(max_i)
    except Exception:
        return 0


def read_runs_registry(base_outputs: Path | None = None) -> List[Dict[str, object]]:
    """Read runs registry JSONL from outputs/runs/runs.jsonl"""
    out_dir = base_outputs or OUTPUTS_DIR
    reg_path = out_dir / 'runs' / 'runs.jsonl'
    if not reg_path.exists():
        return []
    try:
        return [json.loads(line) for line in reg_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except Exception:
        return []

