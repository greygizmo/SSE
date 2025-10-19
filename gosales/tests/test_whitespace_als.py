import sys
from pathlib import Path

import implicit
import polars as pl
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))
from gosales.whitespace.als import build_als


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="resource module not available on Windows"
)
def test_build_als_generates_top_n_recommendations(tmp_path, monkeypatch):
    # Create mock fact_transactions table
    fact_transactions = pl.DataFrame(
        {
            "customer_id": ["cust-1", "cust-1", "cust-2", "cust-2", "cust-3", "cust-3"],
            "item": ["A", "B", "B", "C", "C", "D"],
        }
    )

    # Mock the database read
    monkeypatch.setattr(pl, "read_database", lambda query, engine: fact_transactions)

    output = tmp_path / "als.csv"

    build_als(None, output, top_n=2)

    df = pl.read_csv(output, schema_overrides={"customer_id": pl.Utf8})
    counts = df.group_by("customer_id").len().sort("customer_id")

    # Assert each user has exactly 2 recommendations
    assert counts["len"].to_list() == [2, 2, 2]
    assert set(df["customer_id"].to_list()) == {"cust-1", "cust-2", "cust-3"}
    assert all(isinstance(cid, str) for cid in df["customer_id"].to_list())

    # Memory bound check skipped on Windows where resource is unavailable


def test_build_als_handles_non_string_items(tmp_path, monkeypatch):
    fact_transactions = pl.DataFrame(
        {
            "customer_id": pl.Series(
                ["cust-1", "cust-1", "cust-2", "cust-2"], dtype=pl.Utf8
            ),
            "item": pl.Series(["A", None, 1001, 1001], dtype=pl.Object),
        }
    )

    monkeypatch.setattr(pl, "read_database", lambda query, engine: fact_transactions)

    class DummyALS:
        def __init__(self, *args, **kwargs):
            self.random_state = kwargs.get("random_state")

        def fit(self, matrix):
            self.num_items = matrix.shape[1]

        def recommend(self, user_idx, user_items, N):
            count = min(N, self.num_items)
            indices = list(range(count))
            scores = [float(count - idx) for idx in indices]
            return indices, scores

    monkeypatch.setattr(implicit.als, "AlternatingLeastSquares", DummyALS)

    output = tmp_path / "als_non_string.csv"
    build_als(None, output, top_n=3)

    df = pl.read_csv(output, schema_overrides={"customer_id": pl.Utf8})
    product_names = df["product_name"].to_list()

    assert "[missing-item]" in product_names
    assert "1001" in product_names
    assert set(df["customer_id"].to_list()) <= {"cust-1", "cust-2"}
