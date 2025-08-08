from __future__ import annotations

import hashlib
from typing import Optional


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def txn_key(order_id: str, order_line: int) -> str:
    normalized = f"{str(order_id).strip().upper()}|{int(order_line)}"
    return _sha(normalized)


def customer_key(customer_id: Optional[str], customer_name: str) -> str:
    base = (customer_id if (customer_id and str(customer_id).strip()) else customer_name).strip().upper()
    return _sha(base)


def date_key(date_str: str) -> str:
    # Expect YYYY-MM-DD; strip non-digits and keep yyyymmdd
    digits = "".join(ch for ch in str(date_str) if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return _sha(digits)


