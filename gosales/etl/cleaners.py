from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def clean_currency_value(value: Any) -> float:
    """Convert currency-like inputs to float.

    Handles strings with commas, leading "$", and negative parentheticals.
    None/NaN becomes 0.0.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return 0.0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        # Handle negative in parentheses
        is_negative = text.startswith("(") and text.endswith(")")
        text = text.replace("$", "").replace(",", "").replace("(", "").replace(")", "")
        try:
            number = float(text)
            return -number if is_negative else number
        except Exception:
            return 0.0
    # Fallback
    try:
        return float(value)
    except Exception:
        return 0.0


def coerce_datetime(series: pd.Series) -> pd.Series:
    """Coerce a pandas Series to datetime (UTC-naive), preserving NaT on errors."""
    return pd.to_datetime(series, errors="coerce")


def summarise_dataframe_schema(df: pd.DataFrame) -> Dict[str, str]:
    """Return a simple mapping of column name → dtype string for auditing."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}






