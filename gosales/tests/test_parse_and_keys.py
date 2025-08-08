from __future__ import annotations

from gosales.etl.parse import parse_currency, parse_date, clean_string, normalize_bool
from gosales.etl.keys import txn_key, customer_key, date_key


def test_parse_currency_cases():
    assert parse_currency("$1,234.50") == 1234.5
    assert parse_currency("(2,000)") == -2000.0
    assert parse_currency("1.234,50") == 1234.5
    assert parse_currency(None) == 0.0


def test_parse_date_and_bool_and_clean():
    d = parse_date("2024-01-31")
    assert d is not None and d.year == 2024 and d.month == 1 and d.day == 31
    assert normalize_bool("Yes") is True
    assert normalize_bool("no") is False
    assert normalize_bool("maybe") is None
    assert clean_string(" A\nB\t  C ") == "A B C"


def test_keys_deterministic():
    k1 = txn_key("abc", 1)
    k2 = txn_key(" ABC ", 1)
    assert k1 == k2
    ck1 = customer_key(None, "Acme Co")
    ck2 = customer_key(" ", " acme co ")
    assert ck1 == ck2
    assert date_key("2024-02-15") == "20240215"


