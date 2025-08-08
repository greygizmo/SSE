from __future__ import annotations

from typing import Tuple
import pandas as pd
import numpy as np


def filter_to_cutoff(df: pd.DataFrame, date_col: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out[out[date_col] <= cutoff]


def winsorize_series(s: pd.Series, p: float) -> Tuple[pd.Series, float, float]:
    s_num = pd.to_numeric(s, errors="coerce")
    lower = float(s_num.quantile(1.0 - p)) if p > 0.5 else float(s_num.quantile(0.0))
    upper = float(s_num.quantile(p))
    capped = s_num.clip(lower=lower, upper=upper)
    return capped, lower, upper


