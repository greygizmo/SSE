import json
import numpy as np
import pandas as pd
from pathlib import Path

import pytest
from click.testing import CliRunner

from gosales.utils.paths import OUTPUTS_DIR
import gosales.validation.forward as forward


class _DummyModel:
    def __init__(self, p_hold: np.ndarray):
        self._p = p_hold

    def predict_proba(self, X: pd.DataFrame):
        # Return [1-p, p] as scikit-like output
        p = self._p[: len(X)]
        return np.column_stack([1 - p, p])


def test_ks_train_vs_holdout_computed(monkeypatch):
    division = "Solidworks"
    cutoff = "2024-12-31"
    n = 100

    # Prepare features parquet
    feats = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'f1': np.linspace(0, 1, n),
        'f2': np.linspace(1, 0, n),
    })
    (OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    feat_path = OUTPUTS_DIR / f"features_{division.lower()}_{cutoff}.parquet"
    feats.to_parquet(feat_path, index=False)

    # Prepare train-time p_hat snapshot skewed low
    p_train = np.clip(np.random.RandomState(0).beta(2, 8, size=n), 0.0, 1.0)
    train_scores_path = OUTPUTS_DIR / f"train_scores_{division.lower()}_{cutoff}.csv"
    pd.DataFrame({'customer_id': feats['customer_id'], 'p_hat': p_train}).to_csv(train_scores_path, index=False)

    # Dummy holdout p_hat skewed high
    p_hold = np.clip(np.random.RandomState(1).beta(8, 2, size=n), 0.0, 1.0)

    # Monkeypatch model loader to return dummy model and feature list
    def _fake_loader(div: str):
        assert div == division
        return _DummyModel(p_hold), ['f1', 'f2']

    monkeypatch.setattr(forward, "_load_model_and_features", _fake_loader)

    # Run CLI
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

    # Assert drift.json contains KS(train vs holdout)
    drift_path = OUTPUTS_DIR / 'validation' / division.lower() / cutoff / 'drift.json'
    assert drift_path.exists()
    drift = json.loads(drift_path.read_text(encoding='utf-8'))
    ks_val = drift.get('ks_phat_train_holdout', None)
    assert ks_val is not None
    assert float(ks_val) > 0.05


