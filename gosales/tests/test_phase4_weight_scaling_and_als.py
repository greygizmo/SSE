import numpy as np
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
    als = pd.Series([0.0]*9 + [0.7])           # 10% coverage
    w_adj, adj = _scale_weights_by_coverage(base, als, lift, threshold=0.3)
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
    s = _compute_als_norm(df, cfg=None)
    # The first two (near centroid) should have higher normalized scores
    assert s.iloc[0] > s.iloc[2]
    assert s.iloc[1] > s.iloc[3]


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


def test_ranker_uses_division_specific_centroids(tmp_path, monkeypatch):
    import gosales.pipeline.rank_whitespace as rw

    # Redirect centroid persistence to a temp directory for isolation
    monkeypatch.setattr(rw, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(rw, "ALS_CENTROID_PATH", tmp_path / "als_owner_centroid.npy")
    monkeypatch.setattr(
        rw, "ASSETS_ALS_CENTROID_PATH", tmp_path / "assets_als_owner_centroid.npy"
    )

    alpha_scores = pd.DataFrame(
        {
            "division_name": ["Alpha"] * 4,
            "customer_id": ["a1", "a2", "a3", "a4"],
            "icp_score": [0.8, 0.7, 0.4, 0.2],
            "als_f0": [0.9, 0.95, 0.4, -0.2],
            "als_f1": [0.85, 1.0, 0.35, -0.3],
            "owned_division_pre_cutoff": [True, True, False, False],
        }
    )
    beta_scores = pd.DataFrame(
        {
            "division_name": ["Beta"] * 3,
            "customer_id": ["b1", "b2", "b3"],
            "icp_score": [0.9, 0.5, 0.3],
            "als_f0": [-0.9, -1.1, -0.8],
            "als_f1": [-1.05, -0.9, -1.2],
            "owned_division_pre_cutoff": [False, False, False],
        }
    )

    alpha_result = rw.rank_whitespace(rw.RankInputs(scores=alpha_scores))
    assert not alpha_result.empty
    alpha_path = rw._als_centroid_path_for_div("Alpha")
    assert alpha_path.exists()
    alpha_centroid = np.load(alpha_path)

    beta_filtered, beta_centroid, beta_key = rw._apply_eligibility_and_centroid(
        beta_scores.copy()
    )
    assert beta_centroid is None
    beta_result = rw.rank_whitespace(rw.RankInputs(scores=beta_scores))
    assert not beta_result.empty

    expected_beta_centroid = (
        beta_filtered[["als_f0", "als_f1"]].astype(float).mean(axis=0).to_numpy()
    )
    assert not np.allclose(alpha_centroid, expected_beta_centroid)

    beta_norm = rw._compute_als_norm(
        beta_filtered, owner_centroid=None, centroid_division=beta_key
    )
    wrong_norm = rw._compute_als_norm(
        beta_filtered, owner_centroid=alpha_centroid, centroid_division="Alpha"
    )

    beta_actual_norm = beta_result["als_norm"].reset_index(drop=True)
    assert np.allclose(beta_actual_norm.to_numpy(), beta_norm.to_numpy())
    assert not np.allclose(beta_actual_norm.to_numpy(), wrong_norm.to_numpy())

    beta_actual_scores = beta_result["score"].reset_index(drop=True)
    wrong_scores = 0.6 * beta_result["p_icp_pct"].reset_index(drop=True) + 0.1 * wrong_norm.reset_index(drop=True)
    assert not np.allclose(beta_actual_scores.to_numpy(), wrong_scores.to_numpy())

