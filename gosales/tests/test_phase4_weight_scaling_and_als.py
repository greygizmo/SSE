import pandas as pd

from gosales.pipeline.rank_whitespace import (
    _scale_weights_by_coverage,
    _compute_als_norm,
    _compute_affinity_lift,
    RankInputs,
    rank_whitespace,
)


def test_scale_weights_by_coverage_scales_and_normalizes():
    base = [0.6, 0.2, 0.1, 0.1]  # [p, lift, als, ev]
    lift = pd.Series([0.0]*8 + [0.9, 0.8])     # 20% coverage
    als = pd.Series([0.0]*9 + [0.7])           # 10% normalized coverage
    als_signal = pd.Series([0.0]*9 + [1.0])    # Raw embedding availability
    w_adj, adj = _scale_weights_by_coverage(base, als, lift, threshold=0.3, als_signal=als_signal)
    assert abs(sum(w_adj) - 1.0) < 1e-6
    # Both lift and als should be scaled down (coverage below threshold)
    assert adj.get('als_weight_factor', 1.0) < 1.0
    assert adj.get('aff_weight_factor', 1.0) < 1.0


def test_als_norm_fallback_centroid_prefers_owned_centroid():
    # Build a tiny embedding space where owned centroid is near (1,1)
    df = pd.DataFrame({
        'als_f0': [1.0, 0.9, -0.5, 0.0],
        'als_f1': [1.0, 0.8, -0.5, 0.0],
        'owned_division_pre_cutoff': [True, True, False, False],
    })
    s, raw = _compute_als_norm(df, cfg=None)
    # The first two (near centroid) should have higher normalized scores
    assert s.iloc[0] > s.iloc[2]
    assert s.iloc[1] > s.iloc[3]
    # Owned rows should register raw signal
    assert raw.iloc[0] > 0
    assert raw.iloc[1] > 0


def test_affinity_lift_consumption_prefers_higher_values():
    df = pd.DataFrame({'mb_lift_max': [0.1, 0.5, 0.9]})
    norm = _compute_affinity_lift(df)
    # Monotonic with respect to input ordering after normalization
    assert norm.iloc[0] < norm.iloc[1] < norm.iloc[2]


def test_scores_nonzero_when_signals_zero_coverage():
    df = pd.DataFrame(
        {
            "division_name": ["A", "A", "A"],
            "customer_id": ["c1", "c2", "c3"],
            "icp_score": [0.2, 0.4, 0.6],
        }
    )
    inputs = RankInputs(scores=df)
    result = rank_whitespace(inputs, weights=(0.0, 0.5, 0.5, 0.0))
    assert result["score"].sum() > 0


def test_rank_whitespace_assets_fallback_for_sparse_als():
    df = pd.DataFrame(
        {
            "division_name": ["A"] * 4,
            "customer_id": ["c1", "c2", "c3", "c4"],
            "icp_score": [0.2, 0.3, 0.4, 0.5],
            "als_f0": [0.9, 0.0, 0.0, 0.0],
            "als_f1": [0.8, 0.0, 0.0, 0.0],
            "als_assets_f0": [0.5, 0.2, 0.1, 0.3],
            "als_assets_f1": [0.4, 0.3, 0.2, 0.35],
        }
    )
    _, raw = _compute_als_norm(df, cfg=None)
    # Only the first row has ALS signal strength
    assert raw.iloc[0] > 0 and raw.iloc[1:].eq(0).all()

    inputs = RankInputs(scores=df)
    result = rank_whitespace(inputs, weights=(0.0, 0.0, 1.0, 0.0))
    res = result.set_index("customer_id")
    # Rows without ALS embeddings should receive fallback scores
    assert res.loc["c2", "als_norm"] > 0
    assert res.loc["c3", "als_norm"] > 0
    assert res.loc["c4", "als_norm"] > 0
    # The true ALS row should remain competitive
    assert res.loc["c1", "als_norm"] >= res.loc["c2", "als_norm"]


