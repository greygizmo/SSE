import pandas as pd
import polars as pl
from sqlalchemy import create_engine

from gosales.whitespace.build_lift import build_lift


def test_build_lift_writes_rules_csv(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'lift_fixture.db'}")

    transactions = pd.DataFrame(
        {
            "customer_id": [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "product_sku": [
                "CAD_A",
                "CAD_B",
                "CAD_A",
                "CAD_B",
                "CAD_A",
                "CAD_B",
                "CAD_A",
                "CAD_B",
                "CAD_C",
                "CAD_C",
                "CAD_C",
                "CAD_C",
                "CAD_C",
                "CAD_C",
            ],
        }
    )

    transactions.to_sql("fact_transactions", engine, index=False, if_exists="replace")

    output_path = tmp_path / "lift.csv"
    build_lift(engine, output_path)

    assert output_path.exists(), "The lift CSV should be created"

    rules = pl.read_csv(output_path)
    assert not rules.is_empty(), "Expected lift rules to be generated"
    assert {"antecedents", "consequents", "lift"}.issubset(rules.columns)
