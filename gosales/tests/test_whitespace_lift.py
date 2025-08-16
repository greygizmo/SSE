import numpy as np
import pandas as pd
import polars as pl
from sqlalchemy import create_engine

import gosales.whitespace.build_lift as build_lift_module


def test_basket_plus_binary_and_lift_finite(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    fact_orders = pl.DataFrame(
        {
            "customer_id": [
                1,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "product_name": [
                "A",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "A",
                "A",
                "C",
                "C",
                "C",
                "C",
            ],
        }
    )
    fact_orders.write_database("fact_orders", engine)

    captured: dict[str, pd.DataFrame] = {}

    from mlxtend.frequent_patterns import apriori as apriori_orig

    def apriori_capture(df, *args, **kwargs):
        captured["df"] = df
        return apriori_orig(df, *args, **kwargs)

    monkeypatch.setattr(build_lift_module, "apriori", apriori_capture)

    output_path = tmp_path / "lift.csv"
    build_lift_module.build_lift(engine, output_path)

    passed_df = captured["df"]
    assert set(np.unique(passed_df.to_numpy().ravel())) <= {0, 1}

    rules = pd.read_csv(output_path)
    assert not rules.empty
    assert np.isfinite(rules["lift"]).all()

