import numpy as np
import pytest

from gosales.models.metrics import (
    compute_topk_threshold,
    compute_lift_at_k,
    compute_weighted_lift_at_k,
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


def test_calibration_bins_constant_scores():
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0.5] * 5)
    bins = calibration_bins(y, p, n_bins=10)
    # With constant scores we should fall back to a single bin
    assert len(bins) == 1
    assert bins['count'].iloc[0] == 5


def test_lift_at_k_monotonic():
    # Higher scores correspond to higher y_prob → lift should be > 1
    y = np.array([0]*90 + [1]*10)
    scores = np.concatenate([np.linspace(0, 0.2, 90), np.linspace(0.8, 1.0, 10)])
    lift10 = compute_lift_at_k(y, scores, 10)
    assert lift10 > 1.0


def test_lift_at_k_zero_base_nan_default():
    y = np.zeros(50)
    scores = np.linspace(0, 1, 50)
    result = compute_lift_at_k(y, scores, 10)
    assert np.isnan(result)


def test_lift_at_k_zero_base_custom_default():
    y = np.zeros(30)
    scores = np.linspace(0, 1, 30)
    result = compute_lift_at_k(y, scores, 10, zero_division=0.0)
    assert result == 0.0


def test_lift_at_k_sanitizes_nan_scores():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, np.nan, 0.2, 0.3])
    result = compute_lift_at_k(y, scores, 50)
    assert not np.isnan(result)


def test_lift_at_k_invalid_k_percent():
    with pytest.raises(ValueError):
        compute_lift_at_k(np.array([0, 1]), np.array([0.1, 0.2]), 120)


def test_weighted_lift_handles_nan_and_zero_base():
    y = np.zeros(4)
    scores = np.array([0.1, 0.2, np.nan, 0.4])
    weights = np.array([1.0, np.nan, 2.0, 1.0])
    result = compute_weighted_lift_at_k(y, scores, weights, 50)
    assert np.isnan(result)


