from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import roc_auc_score


def compute_lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k_percent: int) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(1, int(n * (k_percent / 100.0)))
    idx = np.argsort(-y_score, kind="stable")[:k]
    topk_rate = float(np.mean(y_true[idx]))
    base_rate = float(np.mean(y_true)) if np.mean(y_true) > 0 else 1e-9
    return topk_rate / base_rate


def compute_weighted_lift_at_k(y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray, k_percent: int) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(1, int(n * (k_percent / 100.0)))
    idx = np.argsort(-y_score, kind="stable")[:k]
    top_y = y_true[idx]
    top_w = weights[idx]
    base = (y_true * weights).sum() / max(1e-9, weights.sum())
    top = (top_y * top_w).sum() / max(1e-9, top_w.sum())
    return float(top / max(1e-9, base))


def compute_topk_threshold(y_score: np.ndarray, k_percent: int) -> float:
    n = len(y_score)
    if n == 0:
        return float("nan")
    k = max(1, int(n * (k_percent / 100.0)))
    sorted_scores = np.sort(y_score)
    return float(sorted_scores[-k])


def calibration_bins(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    if len(y_true) == 0:
        return pd.DataFrame(columns=["mean_predicted", "fraction_positives", "count"])  # empty
    df = pd.DataFrame({"y": y_true, "p": y_score})
    try:
        bins = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except Exception:
        # If not enough unique values, fall back to equal-width bins
        bins = pd.cut(df["p"], bins=n_bins, include_lowest=True, duplicates="drop")
    grouped = df.assign(bin=bins).groupby("bin", observed=False).agg(
        mean_predicted=("p", "mean"),
        fraction_positives=("y", "mean"),
        count=("y", "size"),
    ).reset_index(drop=True)
    return grouped


def calibration_mae(bins_df: pd.DataFrame, weighted: bool = True) -> float:
    if bins_df.empty:
        return float("nan")
    diff = (bins_df["mean_predicted"].astype(float) - bins_df["fraction_positives"].astype(float)).abs()
    if weighted:
        w = bins_df["count"].astype(float)
        return float((diff * w).sum() / max(1e-9, w.sum()))
    return float(diff.mean())


def drop_leaky_features(
    X: pd.DataFrame,
    y: np.ndarray,
    auc_threshold: float = 0.995,
    name_patterns: Tuple[str, ...] = ("future", "label", "bought_in_division", "target"),
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove features that appear to leak the target.

    - Drops columns whose name suggests leakage (contains any of name_patterns).
    - Drops numeric columns whose single-variable AUC vs target exceeds auc_threshold.
    Returns a new DataFrame and the list of columns dropped.
    """
    drop_cols: List[str] = []
    cols = list(X.columns)
    # Name-based drop
    for c in cols:
        lc = str(c).lower()
        if any(pat in lc for pat in name_patterns):
            drop_cols.append(c)

    # Score-based drop for numeric columns
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    for c in num_cols:
        if c in drop_cols:
            continue
        s = X[c].to_numpy()
        if np.allclose(s, s[0]):
            continue
        try:
            auc1 = roc_auc_score(y, s)
            auc2 = roc_auc_score(y, -s)
            if max(auc1, auc2) >= auc_threshold:
                drop_cols.append(c)
        except Exception:
            # Non-finite or unsuitable vector; skip
            continue

    if drop_cols:
        X = X.drop(columns=list(set(drop_cols)), errors="ignore")
    return X, drop_cols


