from __future__ import annotations

import numpy as np
import pandas as pd


def compute_lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k_percent: int) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(1, int(n * (k_percent / 100.0)))
    idx = np.argsort(-y_score)[:k]
    topk_rate = float(np.mean(y_true[idx]))
    base_rate = float(np.mean(y_true)) if np.mean(y_true) > 0 else 1e-9
    return topk_rate / base_rate


def compute_weighted_lift_at_k(y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray, k_percent: int) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(1, int(n * (k_percent / 100.0)))
    idx = np.argsort(-y_score)[:k]
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
    grouped = df.assign(bin=bins).groupby("bin").agg(
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


