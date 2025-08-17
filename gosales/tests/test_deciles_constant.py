import numpy as np
import pandas as pd

from gosales.validation.deciles import gains_and_capture


def test_gains_and_capture_decile_counts(tmp_path):
    # Constant scores should yield a single decile
    df = pd.DataFrame({
        'division_name': ['A'] * 5,
        'icp_score': [0.5] * 5,
        'bought_in_division': [0, 1, 0, 1, 0],
    })
    path = tmp_path / 'const.csv'
    df.to_csv(path, index=False)
    gains, _ = gains_and_capture(path)
    assert gains['decile'].nunique() == 1
    assert len(gains) == 1

    # Diverse scores should yield 10 deciles
    df2 = pd.DataFrame({
        'division_name': ['A'] * 20,
        'icp_score': np.linspace(0, 1, 20),
        'bought_in_division': [0, 1] * 10,
    })
    path2 = tmp_path / 'var.csv'
    df2.to_csv(path2, index=False)
    gains2, _ = gains_and_capture(path2)
    assert gains2['decile'].nunique() == 10
    assert len(gains2) == 10
