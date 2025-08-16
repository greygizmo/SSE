import pandas as pd


def test_capacity_selection_with_ties_returns_exact_k():
    ranked = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5, 6],
        'score': [0.9, 0.8, 0.8, 0.8, 0.8, 0.8],
        'p_icp': [0.5, 0.4, 0.4, 0.4, 0.4, 0.4],
        'EV_norm': [0.2, 0.3, 0.3, 0.3, 0.3, 0.3],
    })
    sort_cols = [c for c in ['score', 'p_icp', 'EV_norm', 'customer_id'] if c in ranked.columns]
    selected = ranked.nlargest(3, sort_cols)
    assert len(selected) == 3
    assert selected['customer_id'].tolist() == [1, 6, 5]

