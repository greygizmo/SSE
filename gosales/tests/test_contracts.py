import pandas as pd

from gosales.etl.contracts import (
    check_required_columns,
    check_primary_key_not_null,
    check_no_duplicate_pk,
    check_date_parse_and_bounds,
)


def test_contract_required_columns_and_pk():
    df = pd.DataFrame(
        {
            "CustomerId": [1, 2, 2],
            "Rec Date": ["2024-01-01", "2024-01-02", "2024-01-02"],
            "Division": ["Solidworks", "Solidworks", "Solidworks"],
        }
    )

    req = ["CustomerId", "Rec Date", "Division", "SWX_Core", "SWX_Core_Qty"]
    v1 = check_required_columns(df, "sales_log", req)
    assert any(v.violation_type == "missing_column" for v in v1)

    v2 = check_primary_key_not_null(df, "sales_log", ("CustomerId", "Rec Date"))
    assert all(v.violation_type != "null_in_pk" for v in v2)

    v3 = check_no_duplicate_pk(df, "sales_log", ("CustomerId", "Rec Date"))
    assert any(v.violation_type == "duplicate_pk" for v in v3)


def test_contract_date_bounds():
    df = pd.DataFrame({"Rec Date": ["2024-01-01", "2050-01-01", "bad"]})
    v = check_date_parse_and_bounds(df, "sales_log", "Rec Date", pd.Timestamp("2024-12-31"))
    types = {vi.violation_type for vi in v}
    assert "invalid_date" in types
    assert "date_after_max" in types






