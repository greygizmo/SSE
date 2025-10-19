import pytest
import polars as pl

from gosales.validation import line_item_parity as lip


def test_prepare_division_metrics_filters_cutoff_and_aggregates():
    frame = pl.DataFrame(
        {
            "division_canonical": ["Solidworks", "Solidworks", "Scanning"],
            "Revenue_usd": [100.0, 50.0, 20.0],
            "GP_usd": [30.0, 15.0, 5.0],
            "Rec_Date": [
                "2025-03-01T00:00:00",
                "2025-07-01T00:00:00",
                "2025-01-15T00:00:00",
            ],
        }
    )

    result = lip._prepare_division_metrics(
        frame,
        division_column="division_canonical",
        revenue_candidates=["Revenue_usd"],
        gp_candidates=["GP_usd"],
        count_column_name="line_count",
        date_column="Rec_Date",
        cutoff="2025-06-30",
    )

    assert result.height == 2
    solidworks_row = result.filter(pl.col("division") == "SOLIDWORKS")
    scanning_row = result.filter(pl.col("division") == "SCANNING")
    assert solidworks_row["revenue"][0] == 100.0
    assert solidworks_row["line_count"][0] == 1
    assert scanning_row["revenue"][0] == 20.0


def test_compare_metrics_computes_deltas_and_status():
    line_metrics = pl.DataFrame(
        {
            "division": ["SOLIDWORKS", "SCANNING"],
            "revenue": [150.0, 80.0],
            "gross_profit": [45.0, 20.0],
            "line_count": [3, 2],
        }
    )
    legacy_metrics = pl.DataFrame(
        {
            "division": ["SOLIDWORKS", "SCANNING"],
            "revenue": [140.0, 100.0],
            "gross_profit": [40.0, 25.0],
            "legacy_count": [2, 3],
        }
    )

    result = lip._compare_metrics(line_metrics, legacy_metrics, delta_threshold=0.25)

    df = result.frame
    solidworks = df.filter(pl.col("division") == "SOLIDWORKS")
    assert pytest.approx(float(solidworks.get_column("delta_revenue").item()), abs=1e-6) == 10.0
    assert pytest.approx(float(solidworks.get_column("delta_revenue_pct").item()), abs=1e-6) == 10.0 / 140.0
    assert result.summary["status"] == "pass"

    # Tighten threshold to force failure
    result_fail = lip._compare_metrics(line_metrics, legacy_metrics, delta_threshold=0.05)
    assert result_fail.summary["status"] == "fail"
