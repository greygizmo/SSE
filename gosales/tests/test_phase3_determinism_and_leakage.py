import numpy as np
import pandas as pd

from gosales.models.metrics import drop_leaky_features


def test_drop_leaky_features_by_name_and_auc():
    # Build synthetic matrix with a leaky column strongly tied to target
    rng = np.random.RandomState(123)
    n = 1000
    y = (rng.rand(n) < 0.2).astype(int)
    # Non-leaky numeric col
    x1 = rng.randn(n)
    # Leaky: exact copy of target (or near-copy with noise)
    x_leak = y.astype(float) + 1e-6 * rng.randn(n)
    # Name hints leakage
    df = pd.DataFrame({
        'safe_feature': x1,
        'future_gp_sum': x_leak,  # name contains 'future'
        'label_hint': y,          # name contains 'label'
    })
    X_new, dropped = drop_leaky_features(df, y, auc_threshold=0.99)
    # Should drop at least the two name-based columns and also the high-AUC col
    assert 'future_gp_sum' in dropped
    assert 'label_hint' in dropped
    assert 'safe_feature' not in dropped
    # Returned matrix should not contain the leaky cols
    assert 'future_gp_sum' not in X_new.columns
    assert 'label_hint' not in X_new.columns


def test_drop_leaky_features_robust_to_constant():
    y = np.array([0, 1, 0, 1])
    df = pd.DataFrame({
        'constant': np.ones(4),
        'vary': np.array([0.1, 0.2, 0.3, 0.4])
    })
    X_new, dropped = drop_leaky_features(df, y, auc_threshold=0.99)
    # Constant should not crash; may or may not be dropped by name; ensure pipeline returns
    assert set(X_new.columns).issubset({'constant', 'vary'})

