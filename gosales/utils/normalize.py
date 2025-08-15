from __future__ import annotations


def normalize_division(text: str | None) -> str:
    """Return a canonical division string for comparisons.

    Currently minimal: trims surrounding whitespace and handles None safely.
    """
    return (text or "").strip()


