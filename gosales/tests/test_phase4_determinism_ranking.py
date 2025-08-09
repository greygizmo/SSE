import numpy as np
import pandas as pd

from gosales.pipeline.rank_whitespace import _percentile_normalize


def test_score_determinism_sort_stable():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'customer_id': np.arange(1000),
        'p_icp': rng.rand(1000),
        'mb_lift_max': rng.rand(1000),
        'als_sim_division': rng.rand(1000),
        'rfm__all__gp_sum__12m': rng.gamma(2.0, 100.0, size=1000),
    })
    # Build normalized components and score twice; expect same order
    p_pct = _percentile_normalize(df['p_icp'])
    lift_norm = _percentile_normalize(df['mb_lift_max'])
    als_norm = _percentile_normalize(df['als_sim_division'])
    ev = _percentile_normalize(df['rfm__all__gp_sum__12m'])
    w = [0.6, 0.2, 0.1, 0.1]
    score1 = w[0]*p_pct + w[1]*lift_norm + w[2]*als_norm + w[3]*ev
    score2 = w[0]*p_pct + w[1]*lift_norm + w[2]*als_norm + w[3]*ev
    order1 = score1.sort_values(ascending=False).index.values
    order2 = score2.sort_values(ascending=False).index.values
    assert np.array_equal(order1, order2)


