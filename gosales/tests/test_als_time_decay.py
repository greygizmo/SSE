from __future__ import annotations

import pandas as pd
import numpy as np


def _compute_time_decay_weight(qty: float, gp: float, rev: float, age_days: int, use_qty: bool = True, include_rev: bool = True, price_factor: float = 1.0, half_life_days: int = 180):
    q_term = np.log1p(1.0 + qty) if use_qty else 0.0
    gp_term = np.log1p(1.0 + max(gp, 0.0))
    price_term = price_factor * (np.log1p(1.0 + max(rev, 0.0)) if include_rev else 0.0)
    w = q_term + gp_term + price_term
    lam = np.log(2.0) / float(max(1, int(half_life_days)))
    decay = np.exp(-lam * float(max(0, int(age_days))))
    return float(w * decay)


def test_time_decay_reduces_older_weights():
    w_new = _compute_time_decay_weight(qty=1.0, gp=10.0, rev=100.0, age_days=0, half_life_days=180)
    w_old = _compute_time_decay_weight(qty=1.0, gp=10.0, rev=100.0, age_days=180, half_life_days=180)
    # At half-life, weight should be roughly half
    assert w_old < w_new
    assert abs((w_old / w_new) - 0.5) < 0.1

