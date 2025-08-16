import numpy as np
import pandas as pd
import pytest

from gosales.pipeline.rank_whitespace import (
    _scale_weights_by_coverage,
    _compute_als_norm,
    _compute_affinity_lift,
    _apply_eligibility,
)


def test_scale_weights_by_coverage_scales_and_normalizes():
    base = [0.6, 0.2, 0.1, 0.1]  # [p, lift, als, ev]
    lift = pd.Series([0.0]*8 + [0.9, 0.8])     # 20% coverage
    als = pd.Series([0.0]*9 + [0.7])           # 10% coverage
    w_adj, adj = _scale_weights_by_coverage(base, als, lift, threshold=0.3)
    assert abs(sum(w_adj) - 1.0) < 1e-6
    # Both lift and als should be scaled down (coverage below threshold)
    assert adj.get('als_weight_factor', 1.0) < 1.0
    assert adj.get('aff_weight_factor', 1.0) < 1.0


def test_als_norm_uses_owned_centroid_after_eligibility():
    # Build a tiny embedding space where owned centroid is near (1,1)
    df = pd.DataFrame({
        'als_f0': [1.0, 0.9, 0.1, 0.0],
        'als_f1': [1.0, 0.8, 0.1, 0.0],
        'owned_division_pre_cutoff': [True, True, False, False],
    })
    eligible, centroid = _apply_eligibility(df)
    s = _compute_als_norm(eligible, owner_centroid=centroid)
    # First eligible row (0.1,0.1) should score higher than (0,0)
    assert s.iloc[0] > s.iloc[1]


def test_als_scores_non_zero_when_candidates_exist():
    df = pd.DataFrame({
        'als_f0': [0.9, 0.2],
        'als_f1': [0.9, 0.1],
        'owned_division_pre_cutoff': [True, False],
    })
    eligible, centroid = _apply_eligibility(df)
    s = _compute_als_norm(eligible, owner_centroid=centroid)
    assert (s > 0).all()


def test_affinity_lift_consumption_prefers_higher_values():
    df = pd.DataFrame({'mb_lift_max': [0.1, 0.5, 0.9]})
    norm = _compute_affinity_lift(df)
    # Monotonic with respect to input ordering after normalization
    assert norm.iloc[0] < norm.iloc[1] < norm.iloc[2]


