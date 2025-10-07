import polars as pl
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))
from gosales.whitespace.als import build_als


@pytest.mark.skipif(sys.platform.startswith("win"), reason="resource module not available on Windows")
def test_build_als_generates_top_n_recommendations(tmp_path, monkeypatch):
    # Create mock fact_orders table
    fact_orders = pl.DataFrame(
        {
            "customer_id": ["cust-1", "cust-1", "cust-2", "cust-2", "cust-3", "cust-3"],
            "item": ["A", "B", "B", "C", "C", "D"],
        }
    )

    # Mock the database read
    monkeypatch.setattr(pl, "read_database", lambda query, engine: fact_orders)

    output = tmp_path / "als.csv"

    build_als(None, output, top_n=2)

    df = pl.read_csv(output, schema_overrides={"customer_id": pl.Utf8})
    counts = df.group_by("customer_id").len().sort("customer_id")

    # Assert each user has exactly 2 recommendations
    assert counts["len"].to_list() == [2, 2, 2]
    assert set(df["customer_id"].to_list()) == {"cust-1", "cust-2", "cust-3"}
    assert all(isinstance(cid, str) for cid in df["customer_id"].to_list())

    # Memory bound check skipped on Windows where resource is unavailable
