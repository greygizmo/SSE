import numpy as np
import pandas as pd
import tracemalloc
from sqlalchemy import create_engine

from gosales.features.engine import create_feature_matrix


def test_feature_matrix_memory_smoke(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/large.db")
    n_rows = 200_000
    divisions = ["Solidworks", "Simulation", "Services", "Hardware"]
    sku_map = {
        "Solidworks": "SWX_Core",
        "Simulation": "Simulation",
        "Services": "Training",
        "Hardware": "Supplies",
    }
    rng = np.random.default_rng(0)
    divs = rng.choice(divisions, size=n_rows)
    skus = [sku_map[d] for d in divs]
    data = pd.DataFrame(
        {
            "customer_id": rng.integers(1, 1000, size=n_rows),
            "order_date": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 90, size=n_rows), unit="D"),
            "product_division": divs,
            "product_sku": skus,
            "gross_profit": rng.random(size=n_rows) * 100,
        }
    )
    data.to_sql("fact_transactions", eng, index=False)
    pd.DataFrame({"customer_id": range(1, 1000)}).to_sql(
        "dim_customer", eng, index=False
    )

    tracemalloc.start()
    fm = create_feature_matrix(
        eng, "Solidworks", cutoff_date="2024-02-01", prediction_window_months=1
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert not fm.is_empty()
    assert peak < 200 * 1024 * 1024

