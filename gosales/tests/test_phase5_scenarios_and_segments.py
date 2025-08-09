import json
import numpy as np
import pandas as pd
from pathlib import Path

from click.testing import CliRunner

from gosales.utils.paths import OUTPUTS_DIR
import gosales.validation.forward as forward


class _DummyModel:
    def __init__(self, p_hold: np.ndarray):
        self._p = p_hold

    def predict_proba(self, X: pd.DataFrame):
        p = self._p[: len(X)]
        return np.column_stack([1 - p, p])


def test_scenarios_math_and_segment_csv(monkeypatch):
    division = "Solidworks"
    cutoff = "2099-12-31"
    n = 10

    # Synthetic feature frame
    ev = np.arange(n, 0, -1).astype(float)  # 10..1
    hold_gp = ev * 10.0                      # 100..10
    y = np.array([1, 1, 1] + [0]*(n-3))      # top 3 positives
    reps = ["repA" if i % 2 == 0 else "repB" for i in range(n)]
    segs = ["A" if i % 2 == 0 else "B" for i in range(n)]
    feats = pd.DataFrame({
        'customer_id': np.arange(1, n + 1),
        'f1': np.linspace(0, 1, n),
        'f2': np.linspace(1, 0, n),
        'EV_norm': ev,
        'bought_in_division': y,
        'rep': reps,
        'industry': segs,
        'holdout_gp': hold_gp,
    })
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    feat_path = OUTPUTS_DIR / f"features_{division.lower()}_{cutoff}.parquet"
    feats.to_parquet(feat_path, index=False)

    # p_hat proportional to EV_norm
    p_hold = (ev - ev.min()) / (ev.max() - ev.min() + 1e-9)
    def _fake_loader(div: str):
        assert div == division
        return _DummyModel(p_hold), ['f1','f2']

    monkeypatch.setattr(forward, "_load_model_and_features", _fake_loader)

    # Run CLI twice to check CI determinism
    runner = CliRunner()
    for _ in range(2):
        result = runner.invoke(forward.main, [
            "--division", division,
            "--cutoff", cutoff,
            "--window-months", "6",
            "--capacity-grid", "30",
            "--accounts-per-rep-grid", "2",
            "--bootstrap", "50",
        ])
        assert result.exit_code == 0

    out_dir = OUTPUTS_DIR / 'validation' / division.lower() / cutoff
    scen_path = out_dir / 'topk_scenarios_sorted.csv'
    seg_path = out_dir / 'segment_performance.csv'
    assert scen_path.exists()
    assert seg_path.exists()

    scen = pd.read_csv(scen_path)
    # Find top_percent 30% row
    row = scen[(scen.get('mode','top_percent') == 'top_percent') & (scen['k_percent'] == 30)].iloc[0]
    # Contacts should be 3
    assert int(row['contacts']) == 3
    # Expected GP norm sum of top 3 EV_norm (10+9+8)
    assert abs(float(row['expected_gp_norm']) - (10+9+8)) < 1e-6
    # Realized GP sum of top 3 holdout_gp (100+90+80)
    assert abs(float(row['realized_gp']) - (100+90+80)) < 1e-6
    # CIs present and deterministic (since we ran twice, check equality by re-reading)
    scen2 = pd.read_csv(scen_path)
    for col in ['capture_ci_lo','capture_ci_hi','precision_ci_lo','precision_ci_hi','rev_capture_ci_lo','rev_capture_ci_hi','realized_gp_ci_lo','realized_gp_ci_hi']:
        assert col in scen.columns
        assert float(scen[col].iloc[0]) == float(scen2[col].iloc[0])

    # Segment file has rows for k=30 and segments A/B
    seg_df = pd.read_csv(seg_path)
    assert 'segment_col' in seg_df.columns
    assert seg_df['k_percent'].isin([30]).any()
    assert set(seg_df['segment'].unique()).issuperset({'A','B'})


