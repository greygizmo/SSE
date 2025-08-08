import numpy as np
import pandas as pd

from gosales.models.metrics import (
    compute_topk_threshold,
    compute_lift_at_k,
    calibration_bins,
    calibration_mae,
)


def test_threshold_math_correctness():
    rng = np.random.RandomState(0)
    scores = rng.rand(1000)
    # ensure deterministic result by sorting
    thr10 = compute_topk_threshold(scores, 10)
    # Exactly 10% should be >= threshold (allow ties handling to include at least k)
    k = max(1, int(len(scores) * 0.10))
    assert (scores >= thr10).sum() >= k


def test_calibration_bins_and_mae():
    # Perfectly calibrated synthetic: y ~ Bernoulli(p)
    rng = np.random.RandomState(42)
    p = rng.rand(5000)
    y = (rng.rand(5000) < p).astype(int)
    bins = calibration_bins(y, p, n_bins=10)
    mae = calibration_mae(bins, weighted=True)
    assert mae < 0.02  # near-perfect calibration should have very small MAE


def test_lift_at_k_monotonic():
    # Higher scores correspond to higher y_prob → lift should be > 1
    y = np.array([0]*90 + [1]*10)
    scores = np.concatenate([np.linspace(0, 0.2, 90), np.linspace(0.8, 1.0, 10)])
    lift10 = compute_lift_at_k(y, scores, 10)
    assert lift10 > 1.0


