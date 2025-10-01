import json
import types

import pandas as pd
import polars as pl
import pytest

import gosales.pipeline.score_customers as sc
import gosales.pipeline.rank_whitespace as rw
from gosales.utils.run_context import default_manifest


def test_emit_capacity_and_logs_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "OUTPUTS_DIR", tmp_path)

    ranked = pd.DataFrame(
        {
            "division_name": ["A", "A", "B"],
            "customer_id": ["c1", "c2", "c3"],
        }
    )
    ranked.attrs["eligibility_counts"] = {
        "overall": {
            "start_rows": 3,
            "owned_excluded": 1,
            "recent_contact_excluded": 0,
            "open_deal_excluded": 0,
            "region_mismatch_excluded": 0,
            "kept_rows": 2,
        },
        "per_division": {
            "A": {
                "start_rows": 2,
                "kept_rows": 1,
                "owned_excluded": 1,
                "recent_contact_excluded": 0,
                "open_deal_excluded": 0,
                "region_mismatch_excluded": 0,
            },
            "B": {
                "start_rows": 1,
                "kept_rows": 1,
                "owned_excluded": 0,
                "recent_contact_excluded": 0,
                "open_deal_excluded": 0,
                "region_mismatch_excluded": 0,
            },
        },
    }
    ranked.attrs["coverage"] = {"als": 0.5, "affinity": 0.8}
    ranked.attrs["weight_adjustments"] = {
        "base": {"p_icp_pct": 0.6, "lift_norm": 0.2, "als_norm": 0.1, "EV_norm": 0.1},
        "global": {
            "normalized_weights": {
                "p_icp_pct": 0.6,
                "lift_norm": 0.2,
                "als_norm": 0.1,
                "EV_norm": 0.1,
            },
            "coverage": {"affinity": 0.8, "als": 0.5},
            "context": "global",
            "segment_size": 3,
        },
        "segments": [
            {
                "segment_key": {"segment": "warm"},
                "normalized_weights": {
                    "p_icp_pct": 0.5,
                    "lift_norm": 0.3,
                    "als_norm": 0.1,
                    "EV_norm": 0.1,
                },
                "coverage": {"affinity": 0.9, "als": 0.4},
                "segment_size": 2,
                "uses_global_weights": False,
            }
        ],
        "segment_columns": ["segment"],
        "als_coverage_threshold": 0.3,
    }

    selected = pd.DataFrame(
        {
            "division_name": ["A", "A", "B"],
            "segment": ["warm", "cold", "warm"],
            "customer_id": ["c1", "c2", "c3"],
        }
    )

    cap_df = sc._emit_capacity_and_logs(ranked, selected, cutoff_tag="20240630")
    assert not cap_df.empty

    csv_path = tmp_path / "capacity_summary_20240630.csv"
    assert csv_path.exists()
    cap_read = pd.read_csv(csv_path)
    assert {"eligibility_start_rows", "eligibility_kept_rows", "eligibility_owned_excluded"}.issubset(
        set(cap_read.columns)
    )
    assert {"segment_warm_selected", "segment_cold_selected"}.issubset(set(cap_read.columns))

    rec_a = cap_read.set_index("division_name").loc["A"]
    assert int(rec_a["eligibility_owned_excluded"]) == 1
    assert int(rec_a["segment_warm_selected"]) == 1

    log_path = tmp_path / "whitespace_log_20240630.jsonl"
    assert log_path.exists()
    with open(log_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert any(rec["division_name"] == "A" for rec in records)
    rec_b = next(rec for rec in records if rec["division_name"] == "B")
    assert rec_b["eligibility"]["kept_rows"] == 1
    assert rec_b["coverage"]["als"] == 0.5
    assert rec_b["weights"]["global"]["normalized_weights"]["p_icp_pct"] == 0.6
    assert rec_b["selection"]["segments"]["warm"] == 1


def test_generate_scoring_outputs_records_whitespace_artifact(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    monkeypatch.setattr(sc, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(rw, "OUTPUTS_DIR", outputs_dir)

    run_manifest = default_manifest()
    run_manifest["run_id"] = "testrun"
    run_manifest["cutoff"] = "2024-06-30"

    model_dir = tmp_path / "A_model"
    model_dir.mkdir()

    monkeypatch.setattr(sc, "discover_available_models", lambda: {"A": model_dir})
    monkeypatch.setattr(sc, "_filter_models_by_targets", lambda available, targets: available)

    def fake_score_customers(*_args, **_kwargs):
        return pl.DataFrame(
            {
                "customer_id": [1],
                "division_name": ["A"],
                "customer_name": ["Acme"],
                "icp_score": [0.8],
                "rfm__all__gp_sum__12m": [1.0],
                "affinity__div__lift_topk__12m": [0.2],
                "bought_in_division": [0],
                "rfm__all__tx_n__12m": [0],
                "assets_active_total": [0],
                "assets_on_subs_total": [0],
            }
        )

    monkeypatch.setattr(sc, "score_customers_for_division", fake_score_customers)
    monkeypatch.setattr(sc, "generate_whitespace_opportunities", lambda engine: pl.DataFrame())
    monkeypatch.setattr(sc, "validate_whitespace_schema", lambda path: {})
    monkeypatch.setattr(sc, "write_schema_report", lambda report, path: None)

    cfg = types.SimpleNamespace(
        whitespace=types.SimpleNamespace(
            shadow_mode=False,
            capacity_mode="top_percent",
            segment_allocation={},
            segment_columns=[],
            segment_min_rows=250,
            bias_division_max_share_topN=1.0,
        ),
        modeling=types.SimpleNamespace(capacity_percent=10),
    )
    monkeypatch.setattr(sc, "load_config", lambda: cfg)

    def fake_rank(inputs):
        base = inputs.scores.copy()
        return base.assign(
            score=0.9,
            score_challenger=0.8,
            p_icp=0.7,
            p_icp_pct=0.6,
            lift_norm=0.5,
            als_norm=0.4,
            EV_norm=0.3,
            nba_reason="Reason",
        )

    monkeypatch.setattr(sc, "rank_whitespace", fake_rank)

    def fake_emit(ranked, selected, cutoff_tag=None):
        return pd.DataFrame({"division_name": ["A"], "selected_count": [len(selected)]})

    monkeypatch.setattr(sc, "_emit_capacity_and_logs", fake_emit)

    sc.generate_scoring_outputs(engine=None, run_manifest=run_manifest)

    whitespace_path = outputs_dir / "whitespace_20240630.csv"
    assert whitespace_path.exists()
    assert run_manifest["whitespace_artifact"] == str(whitespace_path)


def test_generate_scoring_outputs_raises_when_no_models(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    monkeypatch.setattr(sc, "OUTPUTS_DIR", outputs_dir)

    run_manifest = default_manifest()
    run_manifest["run_id"] = "testrun"
    run_manifest["cutoff"] = "2024-06-30"

    monkeypatch.setattr(sc, "discover_available_models", lambda: {})
    monkeypatch.setattr(sc, "_filter_models_by_targets", lambda available, targets: available)

    with pytest.raises(RuntimeError, match="No models were available for scoring"):
        sc.generate_scoring_outputs(engine=None, run_manifest=run_manifest)


def test_generate_scoring_outputs_applies_segment_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(rw, "OUTPUTS_DIR", tmp_path)

    run_manifest = default_manifest()
    run_manifest["run_id"] = "testrun"
    run_manifest["cutoff"] = "2024-06-30"

    model_dir = tmp_path / "A_model"
    model_dir.mkdir()
    monkeypatch.setattr(sc, "discover_available_models", lambda: {"A": model_dir})
    monkeypatch.setattr(sc, "_filter_models_by_targets", lambda available, targets: available)

    base_rows = {
        "division_name": ["A"] * 6,
        "customer_id": [f"c{i}" for i in range(1, 7)],
        "customer_name": [f"Customer {i}" for i in range(1, 7)],
        "icp_score": [0.9, 0.88, 0.87, 0.86, 0.85, 0.84],
        "rfm__all__gp_sum__12m": [1.0] * 6,
        "affinity__div__lift_topk__12m": [0.2] * 6,
        "bought_in_division": [0] * 6,
        "rfm__all__tx_n__12m": [1, 1, 1, 0, 0, 0],
        "assets_active_total": [0, 0, 0, 1, 1, 1],
        "assets_on_subs_total": [0, 0, 0, 0, 0, 0],
    }

    score_frame = pl.DataFrame(base_rows)
    monkeypatch.setattr(
        sc,
        "score_customers_for_division",
        lambda engine, division_name, model_path, **kwargs: score_frame.clone(),
    )
    monkeypatch.setattr(sc, "generate_whitespace_opportunities", lambda engine: pl.DataFrame())
    monkeypatch.setattr(sc, "validate_whitespace_schema", lambda path: {})
    monkeypatch.setattr(sc, "write_schema_report", lambda report, path: None)

    cfg = types.SimpleNamespace(
        population=types.SimpleNamespace(include_prospects=True),
        whitespace=types.SimpleNamespace(
            shadow_mode=False,
            capacity_mode="top_percent",
            segment_allocation={"Warm": 0.6, "COLD": 0.4},
            segment_columns=["segment"],
            segment_min_rows=250,
            bias_division_max_share_topN=1.0,
        ),
        modeling=types.SimpleNamespace(capacity_percent=50),
    )
    monkeypatch.setattr(sc, "load_config", lambda: cfg)

    ranked_rows = pd.DataFrame(
        {
            "division_name": ["A"] * 6,
            "customer_id": [f"c{i}" for i in range(1, 7)],
            "customer_name": [f"Customer {i}" for i in range(1, 7)],
            "score": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
            "p_icp": [0.9, 0.89, 0.88, 0.87, 0.86, 0.85],
            "p_icp_pct": [0.9, 0.89, 0.88, 0.87, 0.86, 0.85],
            "lift_norm": [0.5, 0.49, 0.48, 0.47, 0.46, 0.45],
            "als_norm": [0.4, 0.39, 0.38, 0.37, 0.36, 0.35],
            "EV_norm": [0.3, 0.29, 0.28, 0.27, 0.26, 0.25],
            "nba_reason": ["Reason"] * 6,
        }
    )

    monkeypatch.setattr(sc, "rank_whitespace", lambda inputs: ranked_rows.copy())

    captured: dict[str, pd.DataFrame] = {}

    def fake_emit(ranked, selected, cutoff_tag=None):
        captured["selected"] = selected.copy()
        return pd.DataFrame()

    monkeypatch.setattr(sc, "_emit_capacity_and_logs", fake_emit)

    sc.generate_scoring_outputs(engine=None, run_manifest=run_manifest)

    assert "selected" in captured
    selected = captured["selected"]
    assert len(selected) == 3
    # Segment allocation should ensure at least one cold account is retained
    assert any(selected["customer_id"] == "c4")
    assert (selected["segment"].str.lower() == "cold").sum() >= 1
    assert (selected["segment"].str.lower() == "warm").sum() >= 1
