from types import SimpleNamespace

import pandas as pd
import pytest

from gosales.pipeline.rank_whitespace import (
    _scale_weights_by_coverage,
    _compute_als_norm,
    _compute_affinity_lift,
    _compute_assets_als_norm,
    _compute_item2vec_norm,
    RankInputs,
    rank_whitespace,
)
from gosales.pipeline.rank_whitespace import _als_centroid_path_for_div
import gosales.pipeline.rank_whitespace as rw


def test_scale_weights_by_coverage_scales_and_normalizes():
    base = [0.6, 0.2, 0.1, 0.1]  # [p, lift, als, ev]
    lift = pd.Series([0.0] * 8 + [0.9, 0.8])  # 20% coverage
    als = pd.Series([0.0] * 9 + [0.7])  # 10% normalized coverage
    als_signal = pd.Series([0.0] * 9 + [1.0])  # Raw embedding availability
    w_adj, adj = _scale_weights_by_coverage(
        base, als, lift, threshold=0.3, als_signal=als_signal
    )
    assert abs(sum(w_adj) - 1.0) < 1e-6
    # Both lift and als should be scaled down (coverage below threshold)
    assert adj.get("als_weight_factor", 1.0) < 1.0
    assert adj.get("aff_weight_factor", 1.0) < 1.0


def test_rank_whitespace_emits_metadata():
    df = pd.DataFrame(
        {
            "division_name": ["A", "A", "B"],
            "customer_id": ["c1", "c2", "c3"],
            "icp_score": [0.2, 0.5, 0.4],
            "owned_division_pre_cutoff": [False, True, False],
            "mb_lift_max": [0.1, 0.0, 0.6],
            "als_f0": [0.0, 0.0, 0.2],
            "als_f1": [0.0, 0.0, 0.1],
        }
    )
    inputs = RankInputs(scores=df)
    res = rank_whitespace(inputs)
    meta = res.attrs.get("weight_adjustments")
    assert meta and "global" in meta
    coverage = res.attrs.get("coverage")
    assert coverage and "als" in coverage and "affinity" in coverage
    elig = res.attrs.get("eligibility_counts")
    assert elig and elig["overall"]["start_rows"] == len(df)
    assert "A" in elig.get("per_division", {})


def test_als_norm_fallback_centroid_prefers_owned_centroid():
    # Build a tiny embedding space where owned centroid is near (1,1)
    df = pd.DataFrame(
        {
            "als_f0": [1.0, 0.9, -0.5, 0.0],
            "als_f1": [1.0, 0.8, -0.5, 0.0],
            "owned_division_pre_cutoff": [True, True, False, False],
        }
    )
    s, raw = _compute_als_norm(df, cfg=None)
    # The first two (near centroid) should have higher normalized scores
    assert s.iloc[0] > s.iloc[2]
    assert s.iloc[1] > s.iloc[3]
    # Owned rows should register raw signal
    assert raw.iloc[0] > 0
    assert raw.iloc[1] > 0


def test_affinity_lift_consumption_prefers_higher_values():
    df = pd.DataFrame({"mb_lift_max": [0.1, 0.5, 0.9]})
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


def test_assets_only_rows_are_capped_when_coverage_high():
    df = pd.DataFrame(
        {
            "division_name": ["A"] * 6,
            "customer_id": ["t1", "t2", "t3", "t4", "t5", "asset"],
            "icp_score": [0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
            "als_f0": [0.9, 0.8, 0.7, 0.6, 0.5, 0.0],
            "als_f1": [0.85, 0.7, 0.65, 0.55, 0.45, 0.0],
            "als_assets_f0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.95],
            "als_assets_f1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.92],
        }
    )
    inputs = RankInputs(scores=df)
    result = rank_whitespace(inputs, weights=(0.0, 0.0, 1.0, 0.0)).set_index(
        "customer_id"
    )

    txn_ids = ["t1", "t2", "t3", "t4", "t5"]
    txn_max = float(result.loc[txn_ids, "als_norm"].max())
    asset_val = float(result.loc["asset", "als_norm"])

    assert txn_max > 0
    assert asset_val > 0  # fallback still surfaces
    assert asset_val < txn_max  # capped below strongest transaction-driven ALS


def test_rank_whitespace_item2vec_only_fills_remaining_zero_rows(monkeypatch):
    captured: dict[str, pd.Series | None] = {}

    original_scale = rw._scale_weights_by_coverage

    def _capture_scale(*args, **kwargs):
        captured["als_signal"] = kwargs.get("als_signal")
        return original_scale(*args, **kwargs)

    monkeypatch.setattr(rw, "_scale_weights_by_coverage", _capture_scale)

    cfg = SimpleNamespace(
        whitespace=SimpleNamespace(
            als_coverage_threshold=0.9,
            als_blend_weights=[0.5, 0.5],
            segment_columns=[],
            segment_min_rows=250,
            challenger_enabled=False,
            challenger_model="lr",
        ),
        features=SimpleNamespace(use_item2vec=True),
        run=SimpleNamespace(cutoff_date=None, prediction_window_months=6),
    )
    monkeypatch.setattr("gosales.utils.config.load_config", lambda: cfg)

    df = pd.DataFrame(
        {
            "division_name": ["A"] * 5,
            "customer_id": ["c1", "c2", "c3", "c4", "c5"],
            "icp_score": [0.2, 0.3, 0.4, 0.1, 0.05],
            "als_f0": [0.9, 0.0, 0.0, 0.0, 0.0],
            "als_f1": [0.8, 0.0, 0.0, 0.0, 0.0],
            "als_assets_f0": [0.1, 0.6, 0.0, 0.0, 0.0],
            "als_assets_f1": [0.2, 0.6, 0.0, 0.0, 0.0],
            "i2v_f0": [0.2, 0.1, 0.5, 0.0, 0.0],
            "i2v_f1": [0.2, 0.1, 0.5, 0.0, 0.0],
        }
    )
    inputs = RankInputs(scores=df)
    result = rank_whitespace(inputs, weights=(0.0, 0.0, 1.0, 0.0))
    res = result.set_index("customer_id")

    assets_norm = _compute_assets_als_norm(df, owner_centroid=None)
    i2v_norm = _compute_item2vec_norm(df, owner_centroid=None)

    assets_norm.index = df["customer_id"]
    i2v_norm.index = df["customer_id"]

    assert res.loc["c2", "als_norm"] <= assets_norm.loc["c2"]
    assert res.loc["c2", "als_norm"] < res.loc["c1", "als_norm"]
    assert res.loc["c3", "als_norm"] == pytest.approx(i2v_norm.loc["c3"], abs=1e-5)
    # Asset fallback should not underperform i2v for row c2
    assert res.loc["c2", "als_norm"] >= res.loc["c3", "als_norm"]
    # Item2vec fallback should register signal coverage
    assert captured.get("als_signal") is not None
    als_signal = captured["als_signal"]
    assert float(als_signal.iloc[2]) > 0  # c3 index
    assert res.attrs["coverage"]["als"] > 0.25
    assert res.attrs["coverage"].get("als_i2v", 0.0) > 0


def test_division_specific_owner_als_centroid_no_leakage(tmp_path, monkeypatch):
    # Ensure centroids persist under a temp outputs dir
    monkeypatch.setattr("gosales.pipeline.rank_whitespace.OUTPUTS_DIR", tmp_path)

    # Division A: has owners and ALS vectors near (+1, +1)
    df_a = pd.DataFrame(
        {
            "division_name": ["A"] * 3,
            "customer_id": ["a1", "a2", "a3"],
            "icp_score": [0.1, 0.2, 0.3],
            "als_f0": [1.0, 0.9, 0.8],
            "als_f1": [1.0, 0.8, 0.9],
            "owned_division_pre_cutoff": [True, True, False],
        }
    )

    # Division B: no owners, ALS vectors in opposite quadrant; includes a zero vector
    df_b = pd.DataFrame(
        {
            "division_name": ["B"] * 3,
            "customer_id": ["b1", "b2", "b3"],
            "icp_score": [0.1, 0.2, 0.3],
            "als_f0": [-1.0, -0.5, 0.0],
            "als_f1": [-1.0, -0.5, 0.0],
            "owned_division_pre_cutoff": [False, False, False],
        }
    )

    df_all = pd.concat([df_a, df_b], ignore_index=True)
    inputs = RankInputs(scores=df_all)
    res = rank_whitespace(inputs, weights=(0.0, 0.0, 1.0, 0.0))

    # A's centroid should be persisted; B has no owners so no persisted centroid
    path_a = _als_centroid_path_for_div("A")
    path_b = _als_centroid_path_for_div("B")
    assert path_a.exists()
    assert not path_b.exists()

    # The persisted A centroid should differ from B's in-group mean (fallback)
    import numpy as np

    a_centroid = np.load(path_a)
    b_mean = df_b[["als_f0", "als_f1"]].astype(float).mean(axis=0).to_numpy()
    assert not np.allclose(a_centroid, b_mean)

    # Within division B, ALS norms should reflect signal (non-zero for valid rows, zero for zero-vector row)
    res_b = res[res["division_name"] == "B"].set_index("customer_id")
    assert float(res_b.loc["b3", "als_norm"]) == 0.0  # zero vector yields zero norm
    # At least one valid row should have positive ALS norm
    assert (res_b.loc[["b1", "b2"], "als_norm"] > 0).any()


def test_segment_weighting_populates_challenger_without_champion(monkeypatch):
    cfg = SimpleNamespace(
        whitespace=SimpleNamespace(
            als_coverage_threshold=0.3,
            als_blend_weights=[1.0, 0.0],
            segment_columns=["region"],
            segment_min_rows=1,
            challenger_enabled=False,
            challenger_model="lr",
        ),
        features=SimpleNamespace(use_item2vec=False),
        run=SimpleNamespace(cutoff_date=None, prediction_window_months=6),
    )
    monkeypatch.setattr("gosales.utils.config.load_config", lambda: cfg)

    df = pd.DataFrame(
        {
            "division_name": ["A", "A", "A"],
            "customer_id": ["c1", "c2", "c3"],
            "icp_score": [0.9, 0.7, 0.5],
            "region": ["west", "east", "east"],
            "als_f0": [0.4, 0.2, 0.1],
            "als_f1": [0.3, 0.1, 0.0],
            "mb_lift_max": [0.1, 0.2, 0.0],
        }
    )

    res = rank_whitespace(RankInputs(scores=df))
    assert "score" in res.columns
    assert "score_challenger" in res.columns
    assert res["score_challenger"].equals(res["score"])
