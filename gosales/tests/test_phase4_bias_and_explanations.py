import pandas as pd
import numpy as np

from gosales.pipeline.rank_whitespace import _explain


def test_explain_short_and_tokens():
    row = pd.Series({
        'p_icp': 0.83,
        'lift_norm': 0.8,
        'als_norm': 0.2,
        'EV_norm': 0.9,
    })
    txt = _explain(row)
    assert len(txt) <= 140
    assert 'High p=' in txt
    assert ('affinity' in txt) or ('ALS' in txt) or ('EV' in txt)


