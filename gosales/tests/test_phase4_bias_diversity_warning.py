import pandas as pd
import numpy as np

from gosales.pipeline.rank_whitespace import _percentile_normalize


def test_bias_warning_logic_share_calc():
    # Simulate selected set: 80% Solidworks, 20% Services
    selected = pd.DataFrame({
        'division': ['Solidworks']*80 + ['Services']*20,
        'customer_id': list(range(100)),
    })
    shares = selected.groupby('division')['customer_id'].size().sort_values(ascending=False)
    total_sel = max(1, int(len(selected)))
    share_map = {div: float(cnt) / total_sel for div, cnt in shares.items()}
    assert share_map['Solidworks'] > 0.6


