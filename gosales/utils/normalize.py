from __future__ import annotations


def normalize_division(text: str | None) -> str:
    """Return a canonical division string for comparisons.

    The canonical form trims surrounding whitespace and applies ``casefold``
    so downstream comparisons can be performed in a case-insensitive manner.
    ``None`` inputs are converted to the empty string.
    """
    return (text or "").strip().casefold()


def normalize_model_key(text: str | None) -> str:
    """Normalize model/target keys for robust matching.

    Rules:
    - Safe on ``None`` (treated as empty string)
    - Use ``normalize_division`` for baseline trimming and casefolding
    - Treat underscores and hyphens as spaces
    - Collapse multiple spaces into a single space
    """
    base = normalize_division(text)
    base = base.replace("_", " ").replace("-", " ")
    return " ".join(base.split())

