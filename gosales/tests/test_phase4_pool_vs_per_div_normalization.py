import numpy as np
import pandas as pd

from gosales.pipeline.rank_whitespace import _percentile_normalize


def test_pooled_vs_per_division_normalization_behaviors():
    # Create two divisions with different distributions
    a = pd.Series(np.linspace(0, 1, 100))
    b = pd.Series(np.concatenate([np.zeros(90), np.ones(10)]))
    a_norm = _percentile_normalize(a)
    b_norm = _percentile_normalize(b)
    # Means around 0.5 for each
    assert 0.4 < a_norm.mean() < 0.6
    assert 0.4 < b_norm.mean() < 0.6


