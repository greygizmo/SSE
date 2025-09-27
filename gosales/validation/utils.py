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
        sampled_counts = pd.Series(sample_ids).value_counts()
        selected = df[df['customer_id'].isin(sampled_counts.index)].copy()
        # Repeat each customer's rows according to how many times they were sampled
        repeat_counts = selected['customer_id'].map(sampled_counts)
        sample = selected.loc[selected.index.repeat(repeat_counts)].reset_index(drop=True)
        stats.append(metric_fn(sample))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def psi(train: pd.Series, holdout: pd.Series, bins: int = 10) -> float:
    # Population Stability Index
    t = pd.to_numeric(train, errors='coerce').dropna()
    h = pd.to_numeric(holdout, errors='coerce').dropna()
    if t.empty or h.empty:
        return 0.0
    # Use quantiles from the combined distribution for robust bin edges
    combined = pd.concat([t, h])
    edges = np.unique(np.quantile(combined, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0
    t_hist, _ = np.histogram(t, bins=edges)
    h_hist, _ = np.histogram(h, bins=edges)
    # Normalize to probabilities, then clip to epsilon and re-normalize to avoid zeros without biasing totals
    t_pct = t_hist.astype(float) / max(1, t_hist.sum())
    h_pct = h_hist.astype(float) / max(1, h_hist.sum())
    eps = 1e-10
    t_pct = np.clip(t_pct, eps, None)
    h_pct = np.clip(h_pct, eps, None)
    t_pct = t_pct / t_pct.sum()
    h_pct = h_pct / h_pct.sum()
    return float(np.sum((h_pct - t_pct) * np.log(h_pct / t_pct)))


def ks_statistic(train: pd.Series, holdout: pd.Series) -> float:
    from scipy.stats import ks_2samp
    t = pd.to_numeric(train, errors='coerce').dropna()
    h = pd.to_numeric(holdout, errors='coerce').dropna()
    if t.empty or h.empty:
        return 0.0
    return float(ks_2samp(t, h).statistic)


