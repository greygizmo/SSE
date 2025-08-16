import pandas as pd

from gosales.pipeline.rank_whitespace import RankInputs, rank_whitespace


def test_cooldown_resorts_order():
    df = pd.DataFrame({
        'division_name': ['A'] * 5,
        'customer_id': [1, 2, 3, 4, 5],
        'icp_score': [0.9, 0.8, 0.7, 0.6, 0.5],
        'days_since_last_surfaced': [5, 100, 100, 100, 100],
    })
    out = rank_whitespace(RankInputs(scores=df))
    assert list(out['customer_id'].head(2)) == [2, 1]
