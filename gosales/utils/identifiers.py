from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

_DECIMAL_TRAIL_RE = re.compile(r"\.0+$")


def _strip_decimal_suffix(text: str) -> str:
    """Remove trailing .0 sequences from numeric identifiers."""
    if '.' not in text:
        return text
    return _DECIMAL_TRAIL_RE.sub('', text)


def normalize_identifier_value(value: Any) -> str | None:
    """Normalize raw identifier values to clean strings or None."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.lower() == 'nan':
            return None
        return _strip_decimal_suffix(text)
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if math.isnan(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        text = format(float(value), '.15g')
        return _strip_decimal_suffix(text)
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    if text.lower() == 'nan':
        return None
    return _strip_decimal_suffix(text)


def normalize_identifier_series(series: pd.Series) -> pd.Series:
    """Vectorized normalization for pandas Series."""
    return series.map(normalize_identifier_value)


def normalize_identifier_expr(expr: pl.Expr) -> pl.Expr:
    """Polars expression helper to normalize identifiers lazily."""
    return expr.map_elements(normalize_identifier_value, return_dtype=pl.Utf8)
