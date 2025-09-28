import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from gosales.validation.utils import bootstrap_ci


def test_bootstrap_ci_respects_customer_multiplicity():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "value": [1.0, 2.0, 3.0],
        }
    )

    n_boot = 25
    seed = 123
    sample_sizes = []
    sample_sums = []

    def metric_fn(sample_df: pd.DataFrame) -> float:
        sample_sizes.append(len(sample_df))
        total = float(sample_df["value"].sum())
        sample_sums.append(total)
        return total

    bootstrap_ci(metric_fn, df, n=n_boot, seed=seed)

    assert set(sample_sizes) == {len(df)}

    rng = np.random.RandomState(seed)
    customers = df["customer_id"].unique()
    value_map = df.set_index("customer_id")["value"].to_dict()
    expected_sums = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(customers, size=len(customers), replace=True)
        counts = pd.Series(sampled_ids).value_counts()
        total = float(sum(value_map[cust_id] * count for cust_id, count in counts.items()))
        expected_sums.append(total)

    assert sample_sums == expected_sums
