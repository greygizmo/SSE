import json
from pathlib import Path

import pandas as pd

from gosales.ui.utils import compute_validation_badges, load_thresholds


def _make_run(tmp_path: Path, metrics: dict, drift: dict, alerts: dict | None = None) -> Path:
    run_dir = tmp_path / 'outputs' / 'validation' / 'solidworks' / '2025-06-30'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'metrics.json').write_text(pd.Series(metrics).to_json(indent=2), encoding='utf-8')
    (run_dir / 'drift.json').write_text(json.dumps(drift, indent=2), encoding='utf-8')
    if alerts is not None:
        (run_dir / 'alerts.json').write_text(json.dumps(alerts, indent=2), encoding='utf-8')
    return run_dir


def test_compute_validation_badges_ok(tmp_path: Path):
    thr = {'psi_threshold': 0.25, 'ks_threshold': 0.15, 'cal_mae_threshold': 0.03}
    metrics = {
        'metrics': {
            'cal_mae': 0.02,
        }
    }
    drift = {
        'psi_holdout_ev_vs_holdout_gp': 0.10,
        'ks_phat_train_holdout': 0.05,
    }
    run_dir = _make_run(tmp_path, metrics, drift)
    badges = compute_validation_badges(run_dir, thresholds=thr)
    assert badges['cal_mae']['status'] == 'ok'
    assert badges['psi_ev_vs_gp']['status'] == 'ok'
    assert badges['ks_phat_train_holdout']['status'] == 'ok'


def test_compute_validation_badges_alerts(tmp_path: Path):
    thr = {'psi_threshold': 0.25, 'ks_threshold': 0.15, 'cal_mae_threshold': 0.03}
    metrics = {
        'metrics': {
            'cal_mae': 0.05,
        }
    }
    drift = {
        'psi_holdout_ev_vs_holdout_gp': 0.40,
        'ks_phat_train_holdout': 0.20,
    }
    alerts = {
        'alerts': [
            {'type': 'cal_mae', 'value': 0.05, 'threshold': 0.03},
            {'type': 'psi_ev_vs_gp', 'value': 0.40, 'threshold': 0.25},
            {'type': 'ks_phat_train_holdout', 'value': 0.20, 'threshold': 0.15},
        ]
    }
    run_dir = _make_run(tmp_path, metrics, drift, alerts)
    badges = compute_validation_badges(run_dir, thresholds=thr)
    assert badges['cal_mae']['status'] == 'alert'
    assert badges['psi_ev_vs_gp']['status'] == 'alert'
    assert badges['ks_phat_train_holdout']['status'] == 'alert'
    # alerts.json loader
    from gosales.ui.utils import load_alerts
    loaded_alerts = load_alerts(run_dir)
    assert len(loaded_alerts) == 3

def test_streamlit_app_import_smoke(monkeypatch, tmp_path: Path):
    # Prepare a minimal outputs directory structure to satisfy app imports
    fake_outputs = tmp_path / 'outputs'
    fake_outputs.mkdir(parents=True, exist_ok=True)
    # Minimal artifacts to exercise discovery without heavy rendering
    (fake_outputs / 'metrics_solidworks.json').write_text(json.dumps({'ok': True}), encoding='utf-8')
    (fake_outputs / 'gains_solidworks.csv').write_text('decile,bought_in_division_mean\n1,0.1', encoding='utf-8')
    (fake_outputs / 'calibration_solidworks.csv').write_text('bin,mean_predicted,fraction_positives\n1,0.1,0.08', encoding='utf-8')
    (fake_outputs / 'thresholds_solidworks.csv').write_text('k_percent,threshold,count\n10,0.7,100', encoding='utf-8')
    (fake_outputs / 'shap_global_solidworks.csv').write_text('feature,mean_abs_shap\nf1,0.1', encoding='utf-8')
    (fake_outputs / 'whitespace_20240630.csv').write_text('customer_id,division,score\n1,Solidworks,0.5', encoding='utf-8')
    (fake_outputs / 'whitespace_metrics_20240630.json').write_text(json.dumps({'capture_at_10': 0.6}), encoding='utf-8')
    (fake_outputs / 'thresholds_whitespace_20240630.csv').write_text('mode,k,threshold\ntop_percent,10,0.7', encoding='utf-8')

    # Point OUTPUTS_DIR used by the app to the fake path
    import gosales.utils.paths as paths
    monkeypatch.setattr(paths, 'OUTPUTS_DIR', fake_outputs)

    # Import app module to ensure it doesn't crash on import
    import importlib
    import gosales.ui.app as app
    importlib.reload(app)

    from gosales.ui.app import _discover_divisions, _discover_whitespace_cutoffs
    assert 'solidworks' in [d.lower() for d in _discover_divisions()]
    assert '20240630' in _discover_whitespace_cutoffs()

