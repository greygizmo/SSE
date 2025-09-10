import re
from typing import Iterable, Optional

_IDENT_RE = re.compile(r"^[A-Za-z0-9_\[\]\.\s]+$")


def validate_identifier(name: str) -> None:
    """Validate a schema/table/view identifier to mitigate SQL injection risk.

    Allows only alphanumerics, underscore, brackets, dot, and spaces. Disallows
    semicolons, quotes, comments and other special characters.
    Raises ValueError if invalid.
    """
    if name is None:
        raise ValueError("identifier is None")
    s = str(name).strip()
    if not s or len(s) > 256:
        raise ValueError("identifier empty or too long")
    if not _IDENT_RE.match(s):
        raise ValueError(f"invalid identifier characters: {name!r}")
    banned = [";", "--", "/*", "*/", "'", '"']
    if any(b in s for b in banned):
        raise ValueError(f"invalid identifier tokens: {name!r}")
    # Basic bracket balance sanity
    if s.count('[') != s.count(']'):
        raise ValueError(f"unbalanced brackets in identifier: {name!r}")


def ensure_allowed_identifier(name: str, allowlist: Optional[Iterable[str]] = None) -> str:
    """Validate identifier and, if an allowlist is provided, enforce membership.

    Returns the normalized identifier string on success; raises ValueError on failure.
    Membership check is case-insensitive; surrounding whitespace is ignored.
    """
    validate_identifier(name)
    s = str(name).strip()
    if allowlist:
        allowed_norm = {str(x).strip().lower() for x in allowlist if str(x).strip()}
        if s.lower() not in allowed_norm:
            raise ValueError(f"identifier not in allow-list: {name!r}")
    return s
