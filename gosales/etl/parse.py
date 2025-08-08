from __future__ import annotations

import datetime as dt
import re
from decimal import Decimal, InvalidOperation
from typing import Any


def parse_currency(value: Any) -> float:
    """Parse currency-like strings/numbers into float.

    Handles symbols, commas, parentheses negatives, and common European formats.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if text == "":
        return 0.0

    # Parentheses indicate negative
    is_negative = text.startswith("(") and text.endswith(")")
    text = text.replace("$", "").replace("€", "").replace("£", "")
    text = text.replace("(", "").replace(")", "")

    # Heuristic: if there are both '.' and ',' and the last separator is ',', treat ',' as decimal
    if "," in text and "." in text and text.rfind(",") > text.rfind("."):
        # European style: 1.234,56 -> 1234.56
        text = text.replace(".", "").replace(",", ".")
    else:
        # US style: remove thousands commas
        text = text.replace(",", "")

    try:
        number = Decimal(text)
    except InvalidOperation:
        return 0.0
    return float(-number if is_negative else number)


def parse_date(value: Any) -> dt.date | None:
    """Parse many date formats to date (UTC-naive), returning None on failure."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    # Try ISO first
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    # Fallback to pandas if available (optional)
    try:
        import pandas as pd

        ts = pd.to_datetime(text, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    # Collapse whitespace and normalize unicode dashes to ASCII hyphen
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE).strip()
    text = text.replace("–", "-").replace("—", "-")
    return text


def normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


