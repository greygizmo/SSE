from __future__ import annotations


def normalize_division(text: str | None) -> str:
    """Return a canonical division string for comparisons.

    The canonical form trims surrounding whitespace and applies ``casefold``
    so downstream comparisons can be performed in a case-insensitive manner.
    ``None`` inputs are converted to the empty string.
    """

    return (text or "").strip().casefold()


