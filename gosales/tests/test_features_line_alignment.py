import pandas as pd

from gosales.features.engine import (
    _compute_flags_from_line_meta,
    _prepare_branch_source,
)


def test_compute_flags_from_line_meta_derives_new_and_acr():
    line_meta = pd.DataFrame(
        {
            "customer_id": ["1", "1", "2", "3"],
            "order_date": pd.to_datetime(
                ["2023-12-01", "2024-01-05", "2023-11-10", "2025-02-01"]
            ),
            "new_flag": [1, 0, None, None],
            "new_business_flag": ["existing", "New Business", None, "New Customer"],
            "referral_ns_field": ["ACR", None, "Account Coverage Review", None],
        }
    )

    agg = _compute_flags_from_line_meta(line_meta, cutoff_date="2024-01-31")
    assert not agg.empty

    agg = agg.set_index("customer_id")
    assert agg.loc["1", "ever_new_customer"] == 1
    assert agg.loc["1", "ever_acr"] == 1
    assert agg.loc["2", "ever_new_customer"] == 0
    assert agg.loc["2", "ever_acr"] == 1
    # Customer 3 row is beyond cutoff, expect defaults of 0
    assert "3" not in agg.index or agg.loc["3"].fillna(0).eq(0).all()


def test_compute_flags_from_line_meta_returns_empty_without_sources():
    line_meta = pd.DataFrame(
        {"customer_id": ["1", "2"], "order_date": pd.to_datetime(["2024-01-01", "2024-01-02"])}
    )
    agg = _compute_flags_from_line_meta(line_meta, cutoff_date="2024-01-31")
    assert set(agg.columns) >= {"customer_id", "ever_new_customer", "ever_acr"}
    assert agg["ever_new_customer"].eq(0).all()
    assert agg["ever_acr"].eq(0).all()


def test_prepare_branch_source_requires_branch_or_rep():
    populated = pd.DataFrame(
        {
            "customer_id": ["1", "2"],
            "order_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "branch": ["North", None],
            "rep": ["Alice", "Bob"],
        }
    )
    empty = pd.DataFrame(
        {
            "customer_id": ["1", "2"],
            "order_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "branch": [None, None],
            "rep": [None, None],
        }
    )

    prepared = _prepare_branch_source(populated)
    assert not prepared.empty
    assert set(prepared.columns) >= {"customer_id"}

    assert _prepare_branch_source(empty).empty
