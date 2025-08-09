import numpy as np
import pandas as pd

from gosales.pipeline.rank_whitespace import _scale_weights_by_coverage, _compute_als_norm, _compute_affinity_lift


def test_scale_weights_by_coverage_scales_and_normalizes():
    base = [0.6, 0.2, 0.1, 0.1]  # [p, lift, als, ev]
    lift = pd.Series([0.0]*8 + [0.9, 0.8])     # 20% coverage
    als = pd.Series([0.0]*9 + [0.7])           # 10% coverage
    w_adj, adj = _scale_weights_by_coverage(base, als, lift, threshold=0.3)
    assert sum(w_adj) == pytest.approx(1.0, rel=1e-6)
    # Both lift and als should be scaled down (coverage below threshold)
    assert adj.get('als_weight_factor', 1.0) < 1.0
    assert adj.get('aff_weight_factor', 1.0) < 1.0


def test_als_norm_fallback_centroid_prefers_owned_centroid():
    # Build a tiny embedding space where owned centroid is near (1,1)
    df = pd.DataFrame({
        'als_f0': [1.0, 0.9, 0.1, 0.0],
        'als_f1': [1.0, 0.8, 0.1, 0.0],
        'owned_division_pre_cutoff': [True, True, False, False],
    })
    s = _compute_als_norm(df, cfg=None)
    # The first two (near centroid) should have higher normalized scores
    assert s.iloc[0] > s.iloc[2]
    assert s.iloc[1] > s.iloc[3]


def test_affinity_lift_consumption_prefers_higher_values():
    df = pd.DataFrame({'mb_lift_max': [0.1, 0.5, 0.9]})
    norm = _compute_affinity_lift(df)
    # Monotonic with respect to input ordering after normalization
    assert norm.iloc[0] < norm.iloc[1] < norm.iloc[2]


