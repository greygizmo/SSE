import polars as pl
import resource
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from gosales.whitespace.als import build_als


def test_build_als_generates_top_n_recommendations(tmp_path, monkeypatch):
    # Create mock fact_orders table
    fact_orders = pl.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3, 3],
            "product_name": ["A", "B", "B", "C", "C", "D"],
        }
    )

    # Mock the database read
    monkeypatch.setattr(pl, "read_database", lambda query, engine: fact_orders)

    output = tmp_path / "als.csv"

    build_als(None, output, top_n=2)

    df = pl.read_csv(output)
    counts = df.group_by("customer_id").len().sort("customer_id")

    # Assert each user has exactly 2 recommendations
    assert counts["len"].to_list() == [2, 2, 2]

    # Ensure memory usage is within a reasonable bound (<200MB)
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 200_000
