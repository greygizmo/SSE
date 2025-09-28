import json

import pandas as pd

import gosales.pipeline.score_customers as sc


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
