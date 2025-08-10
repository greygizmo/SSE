import numpy as np
import pandas as pd

from gosales.pipeline.rank_whitespace import _compute_expected_value


class DummyCfg:
    class WS:
        ev_cap_percentile = 0.95
    whitespace = WS()


def test_ev_cap_applies():
    df = pd.DataFrame({
        'rfm__all__gp_sum__12m': list(range(100)) + [10000],
    })
    ev_norm, _ = _compute_expected_value(df, DummyCfg)
    # The outlier should be capped; normalized value should be <= 1
    assert float(ev_norm.max()) <= 1.0


