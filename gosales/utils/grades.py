from __future__ import annotations

import pandas as pd


def percentile_rank_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
    """Return rank-based percentiles per group in [0,1].

    Uses average ranks to be stable across ties. Higher values => higher percentile.
    """
    if value_col not in df.columns or group_col not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index, dtype=float)
    # pct=True maps to [0,1]; method='average' stabilizes ties
    return df.groupby(group_col)[value_col].rank(method="average", pct=True).astype(float)


def letter_grade_from_percentile(p: float) -> str:
    """Map percentile in [0,1] to A/B/C/D/F using exec-friendly bins.

    Default bins:
      - A: >= 0.90
      - B: >= 0.70 and < 0.90
      - C: >= 0.40 and < 0.70
      - D: >= 0.20 and < 0.40
      - F: < 0.20
    """
    try:
        x = float(p)
    except Exception:
        x = 0.0
    if x >= 0.90:
        return "A"
    if x >= 0.70:
        return "B"
    if x >= 0.40:
        return "C"
    if x >= 0.20:
        return "D"
    return "F"


def assign_letter_grades_from_percentiles(s: pd.Series) -> pd.Series:
    """Vectorized grade assignment for a percentile Series in [0,1]."""
    return s.map(letter_grade_from_percentile).astype(str)

