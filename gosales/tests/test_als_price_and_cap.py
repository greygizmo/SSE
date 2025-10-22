from __future__ import annotations

import pandas as pd
import numpy as np


def _compute_weights_local(df: pd.DataFrame, use_qty: bool, include_revenue: bool, price_factor: float, cap: float | None):
    qty = pd.to_numeric(df.get('quantity', 1.0), errors='coerce').fillna(1.0)
    gp = pd.to_numeric(df.get('gross_profit', 0.0), errors='coerce').fillna(0.0).clip(lower=0.0)
    rev = pd.to_numeric(df.get('revenue', 0.0), errors='coerce').fillna(0.0).clip(lower=0.0)
    q_term = np.log1p(1.0 + qty) if use_qty else 0.0
    gp_term = np.log1p(1.0 + gp)
    price_term = price_factor * (np.log1p(1.0 + rev) if include_revenue else 0.0)
    w = (q_term + gp_term + price_term).astype('float64')
    if cap is not None:
        w = np.minimum(w, float(cap))
    return w


def test_als_include_revenue_increases_weight():
    df = pd.DataFrame({
        'quantity': [1.0, 1.0],
        'gross_profit': [10.0, 10.0],
        'revenue': [1.0, 1000.0],
    })
    w_no_price = _compute_weights_local(df, use_qty=True, include_revenue=False, price_factor=1.0, cap=None)
    w_with_price = _compute_weights_local(df, use_qty=True, include_revenue=True, price_factor=1.0, cap=None)
    assert (w_with_price > w_no_price).any()


def test_als_weight_cap_applies():
    df = pd.DataFrame({
        'quantity': [1000.0],
        'gross_profit': [1e6],
        'revenue': [1e7],
    })
    w_uncapped = _compute_weights_local(df, use_qty=True, include_revenue=True, price_factor=1.0, cap=None)
    w_capped = _compute_weights_local(df, use_qty=True, include_revenue=True, price_factor=1.0, cap=5.0)
    assert w_capped.iloc[0] <= 5.0
    assert w_uncapped.iloc[0] >= w_capped.iloc[0]

