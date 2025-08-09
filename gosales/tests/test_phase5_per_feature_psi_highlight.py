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


def test_per_feature_psi_highlight(monkeypatch):
    division = "Solidworks"
    cutoff = "2095-12-31"
    n = 1000

    # Train feature sample: drift_feature ~ N(0,1), f1 uniform
    rng = np.random.RandomState(0)
    train_feat = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'drift_feature': rng.normal(loc=0.0, scale=1.0, size=n),
        'f1': rng.rand(n),
    })
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    train_feat.to_parquet(OUTPUTS_DIR / f"train_feature_sample_{division.lower()}_{cutoff}.parquet", index=False)

    # Holdout features: drift_feature shifted N(3,1), f1 similar
    feats = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'drift_feature': rng.normal(loc=3.0, scale=1.0, size=n),
        'f1': rng.rand(n),
        'EV_norm': rng.rand(n),
        'bought_in_division': rng.randint(0, 2, size=n),
    })
    feats.to_parquet(OUTPUTS_DIR / f"features_{division.lower()}_{cutoff}.parquet", index=False)

    # Dummy model: constant p_hat
    p_hold = np.full(n, 0.5)
    monkeypatch.setattr(forward, "_load_model_and_features", lambda d: (_DummyModel(p_hold), ['f1','drift_feature']))

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
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    drift_highlights = metrics.get('drift_highlights', {})
    psi_thr = float(drift_highlights.get('psi_threshold', 0.25))
    flagged = drift_highlights.get('psi_flagged_top', [])
    assert any(item.get('feature') == 'drift_feature' and float(item.get('psi', 0.0)) >= psi_thr for item in flagged)


