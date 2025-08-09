import json
import numpy as np
import pandas as pd

from click.testing import CliRunner
from gosales.utils.paths import OUTPUTS_DIR
import gosales.validation.forward as forward


class _DummyModel:
    def __init__(self, p_hold: np.ndarray):
        self._p = p_hold

    def predict_proba(self, X: pd.DataFrame):
        p = self._p[: len(X)]
        return np.column_stack([1 - p, p])


def test_drift_psi_smoke(monkeypatch):
    division = "Solidworks"
    cutoff = "2098-12-31"
    n = 200
    # Create EV increasing, holdout GP decreasing (strong drift)
    ev = np.linspace(0, 100, n)
    hold_gp = np.linspace(100, 0, n)
    feats = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'f1': np.random.RandomState(0).rand(n),
        'f2': np.random.RandomState(1).rand(n),
        'rfm__all__gp_sum__12m': ev,
        'EV_norm': (ev - ev.min()) / (ev.ptp() + 1e-9),
        'bought_in_division': np.random.RandomState(2).randint(0, 2, size=n),
        'holdout_gp': hold_gp,
    })
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / f"features_{division.lower()}_{cutoff}.parquet").write_bytes(feats.to_parquet(index=False))

    # Dummy p_hat constant
    p_hold = np.full(n, 0.5)
    monkeypatch.setattr(forward, "_load_model_and_features", lambda d: (_DummyModel(p_hold), ['f1','f2']))

    runner = CliRunner()
    result = runner.invoke(forward.main, [
        "--division", division,
        "--cutoff", cutoff,
        "--window-months", "6",
        "--capacity-grid", "10",
        "--accounts-per-rep-grid", "10",
        "--bootstrap", "10",
    ])
    assert result.exit_code == 0
    drift_path = OUTPUTS_DIR / 'validation' / division.lower() / cutoff / 'drift.json'
    drift = json.loads(drift_path.read_text(encoding='utf-8'))
    assert float(drift.get('psi_ev_vs_holdout_gp', 0.0)) > 0.25


def test_calibration_sanity(monkeypatch):
    division = "Solidworks"
    cutoff = "2097-12-31"
    n = 400
    rng = np.random.RandomState(123)
    # p close to 0 or 1 to get small Brier and low cal-MAE
    p = np.where(rng.rand(n) < 0.5, 0.95, 0.05)
    y = (rng.rand(n) < p).astype(int)
    feats = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'f1': rng.rand(n),
        'f2': rng.rand(n),
        'EV_norm': rng.rand(n),
        'bought_in_division': y,
    })
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUTPUTS_DIR / f"features_{division.lower()}_{cutoff}.parquet", index=False)

    monkeypatch.setattr(forward, "_load_model_and_features", lambda d: (_DummyModel(p), ['f1','f2']))
    runner = CliRunner()
    result = runner.invoke(forward.main, [
        "--division", division,
        "--cutoff", cutoff,
        "--window-months", "6",
        "--capacity-grid", "10",
        "--accounts-per-rep-grid", "10",
        "--bootstrap", "10",
    ])
    assert result.exit_code == 0
    metrics_path = OUTPUTS_DIR / 'validation' / division.lower() / cutoff / 'metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    brier = float(metrics['metrics']['brier'])
    cal_mae = float(metrics['metrics']['cal_mae'])
    assert brier < 0.08
    assert cal_mae < 0.05


