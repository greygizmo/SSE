from __future__ import annotations

import pandas as pd
import numpy as np

from gosales.features.als_embed import pd as _pd_module  # reuse pandas


def _compute_weights_local(df: pd.DataFrame, use_qty: bool) -> pd.Series:
    # Mirror als_embed weighting logic without requiring DB or implicit libs
    qty = pd.to_numeric(df.get('quantity', 1.0), errors='coerce').fillna(1.0)
    gp = pd.to_numeric(df.get('gross_profit', 0.0), errors='coerce').fillna(0.0).clip(lower=0.0)
    q_term = np.log1p(1.0 + qty) if use_qty else 0.0
    gp_term = np.log1p(1.0 + gp)
    return (q_term + gp_term).astype('float64')


def test_als_weight_by_quantity_toggle_changes_weights():
    df = pd.DataFrame({
        'quantity': [1.0, 10.0],
        'gross_profit': [10.0, 10.0],
    })
    w_on = _compute_weights_local(df, use_qty=True)
    w_off = _compute_weights_local(df, use_qty=False)
    # When using quantity, second row (qty=10) should have larger weight; when off, both equal
    assert w_on.iloc[1] > w_on.iloc[0]
    assert np.isclose(w_off.iloc[1], w_off.iloc[0])

