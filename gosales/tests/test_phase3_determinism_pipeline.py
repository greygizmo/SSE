import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from gosales.models.metrics import drop_leaky_features


def _train_lr_calibrated(X: pd.DataFrame, y: np.ndarray, seed: int = 7):
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.2, C=1.0,
        max_iter=2000, class_weight='balanced', random_state=seed
    )
    lr.fit(Xs, y)
    cal = CalibratedClassifierCV(lr, method='sigmoid', cv=3)
    cal.fit(Xs, y)
    p = cal.predict_proba(Xs)[:, 1]
    return p


def test_determinism_same_seed_same_probs():
    rng = np.random.RandomState(123)
    n = 1200
    X = pd.DataFrame({
        'x1': rng.randn(n),
        'x2': rng.randn(n),
        'x3': rng.randn(n),
    })
    # Generate target with logistic link
    logits = 0.8 * X['x1'].values - 0.5 * X['x2'].values + 0.2 * X['x3'].values
    p_true = 1 / (1 + np.exp(-logits))
    y = (rng.rand(n) < p_true).astype(int)

    p1 = _train_lr_calibrated(X, y, seed=777)
    p2 = _train_lr_calibrated(X, y, seed=777)
    assert np.allclose(p1, p2, atol=1e-10)


def test_leakage_probe_no_gain_after_guard():
    rng = np.random.RandomState(321)
    n = 2000
    X = pd.DataFrame({
        'x1': rng.randn(n),
        'x2': rng.randn(n),
        'x3': rng.randn(n),
    })
    logits = 0.7 * X['x1'].values - 0.4 * X['x2'].values + 0.1 * X['x3'].values
    p_true = 1 / (1 + np.exp(-logits))
    y = (rng.rand(n) < p_true).astype(int)

    # Baseline AUC
    p_base = _train_lr_calibrated(X, y, seed=11)
    auc_base = roc_auc_score(y, p_base)

    # Inject future/leaky feature correlated with y
    X_leaky = X.copy()
    X_leaky['future_flag'] = y.astype(float) + 1e-6 * rng.randn(n)
    p_leak = _train_lr_calibrated(X_leaky, y, seed=11)
    auc_leak = roc_auc_score(y, p_leak)
    assert auc_leak - auc_base > 0.05  # leakage should inflate AUC noticeably

    # Guard: drop leaky features, retrain
    X_guarded, dropped = drop_leaky_features(X_leaky, y, auc_threshold=0.99)
    assert 'future_flag' in dropped
    p_guard = _train_lr_calibrated(X_guarded, y, seed=11)
    auc_guard = roc_auc_score(y, p_guard)
    # After guard, AUC should be close to baseline
    assert abs(auc_guard - auc_base) < 0.01


