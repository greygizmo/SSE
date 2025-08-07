import pandas as pd

from gosales.etl.cleaners import clean_currency_value, coerce_datetime


def test_clean_currency_value():
    assert clean_currency_value("$1,234.50") == 1234.5
    assert clean_currency_value("(2,000)") == -2000.0
    assert clean_currency_value(15) == 15.0
    assert clean_currency_value(None) == 0.0


def test_coerce_datetime():
    s = pd.Series(["2024-01-01", "bad", None])
    out = coerce_datetime(s)
    assert pd.notna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])



