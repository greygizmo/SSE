from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Tuple


def bootstrap_ci(metric_fn: Callable[[pd.DataFrame], float], df: pd.DataFrame, n: int = 1000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    customers = df['customer_id'].unique()
    stats = []
    for _ in range(n):
        sample_ids = rng.choice(customers, size=len(customers), replace=True)
        sample = df[df['customer_id'].isin(sample_ids)]
        stats.append(metric_fn(sample))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def psi(train: pd.Series, holdout: pd.Series, bins: int = 10) -> float:
    # Population Stability Index
    t = pd.to_numeric(train, errors='coerce').dropna()
    h = pd.to_numeric(holdout, errors='coerce').dropna()
    if t.empty or h.empty:
        return 0.0
    edges = np.quantile(t, np.linspace(0.0, 1.0, bins+1))
    t_hist, _ = np.histogram(t, bins=edges)
    h_hist, _ = np.histogram(h, bins=edges)
    t_pct = (t_hist / max(1, t_hist.sum())).clip(1e-9)
    h_pct = (h_hist / max(1, h_hist.sum())).clip(1e-9)
    return float(np.sum((h_pct - t_pct) * np.log(h_pct / t_pct)))


def ks_statistic(train: pd.Series, holdout: pd.Series) -> float:
    from scipy.stats import ks_2samp
    t = pd.to_numeric(train, errors='coerce').dropna()
    h = pd.to_numeric(holdout, errors='coerce').dropna()
    if t.empty or h.empty:
        return 0.0
    return float(ks_2samp(t, h).statistic)


