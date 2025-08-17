import numpy as np
import pandas as pd

from gosales.pipeline.rank_whitespace import _percentile_normalize


def test_percentile_normalize_uniform_like():
    # Construct a simple vector with unique values per division and check percentile spreads
    rng = np.random.RandomState(0)
    s1 = pd.Series(rng.rand(100))
    s2 = pd.Series(rng.rand(200))
    p1 = _percentile_normalize(s1)
    p2 = _percentile_normalize(s2)
    # Means should be around ~0.5, std non-zero
    assert 0.4 < p1.mean() < 0.6
    assert 0.4 < p2.mean() < 0.6
    assert p1.std() > 0.15
    assert p2.std() > 0.15


def test_percentile_normalize_constant_values():
    s = pd.Series([5.0] * 10)
    p = _percentile_normalize(s)
    assert (p == 0).all()


