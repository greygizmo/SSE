import pandas as pd
import polars as pl
from pathlib import Path
from unittest.mock import MagicMock

from gosales.pipeline import score_customers as sc


def test_read_sql_invoked_once(monkeypatch, tmp_path):
    # Patch read_sql to track calls and return dummy customer names
    customer_df = pd.DataFrame({"customer_id": [1], "customer_name": ["Acme"]})
    read_sql_mock = MagicMock(return_value=customer_df)
    monkeypatch.setattr(pd, "read_sql", read_sql_mock)

    # Provide two fake model directories
    model1 = tmp_path / "div1_model"
    model1.mkdir()
    model2 = tmp_path / "div2_model"
    model2.mkdir()
    monkeypatch.setattr(
        sc,
        "discover_available_models",
        lambda: {"div1": model1, "div2": model2},
    )

    # Capture the customer_names objects passed to scorer
    passed = []

    def fake_score(engine, division_name, model_path, *, customer_names, run_manifest=None):
        passed.append(customer_names)
        # Minimal dataframe with required columns
        return pl.DataFrame(
            {
                "customer_id": [1],
                "division_name": [division_name],
                "bought_in_division": [0],
                "icp_score": [0.1],
            }
        )

    monkeypatch.setattr(sc, "score_customers_for_division", fake_score)

    # Stub downstream heavy functions
    dummy_rank = pd.DataFrame(
        {
            "customer_id": [1],
            "division_name": ["div1"],
            "score": [0.1],
            "p_icp": [0.1],
            "p_icp_pct": [0.1],
            "lift_norm": [0.1],
            "als_norm": [0.1],
            "EV_norm": [0.1],
        }
    )
    monkeypatch.setattr(sc, "rank_whitespace", lambda inputs: dummy_rank)
    monkeypatch.setattr(sc, "save_ranked_whitespace", lambda ranked, cutoff_tag=None: tmp_path / "ws.csv")

    class WS:
        shadow_mode = False
        capacity_mode = "top_percent"
        bias_division_max_share_topN = 1.0
        ev_cap_percentile = None

    class Modeling:
        capacity_percent = 10

    class Cfg:
        whitespace = WS()
        modeling = Modeling()

    monkeypatch.setattr(sc, "load_config", lambda: Cfg())
    monkeypatch.setattr(sc, "generate_whitespace_opportunities", lambda engine: pl.DataFrame())
    monkeypatch.setattr(sc, "validate_icp_scores_schema", lambda path: {})
    monkeypatch.setattr(sc, "write_schema_report", lambda report, path: None)
    monkeypatch.setattr(sc, "validate_whitespace_schema", lambda path: {})
    monkeypatch.setattr(sc, "emit_validation_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(sc, "check_drift_and_emit_alerts", lambda *a, **k: None)

    # Use temporary output directory
    monkeypatch.setattr(sc, "OUTPUTS_DIR", tmp_path)

    sc.generate_scoring_outputs(engine=None)

    assert read_sql_mock.call_count == 1
    assert passed[0] is passed[1]
