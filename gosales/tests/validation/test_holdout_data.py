from types import SimpleNamespace

import pandas as pd
import pytest

from gosales.validation import holdout_data as hd


def _make_cfg(*, use_line_items: bool, holdout_source: str = "db", holdout_db_object: str | None = None):
    return SimpleNamespace(
        validation=SimpleNamespace(holdout_source=holdout_source, holdout_db_object=holdout_db_object),
        etl=SimpleNamespace(
            line_items=SimpleNamespace(use_line_item_facts=use_line_items),
            source_columns={},
        ),
        database=SimpleNamespace(source_tables={"sales_detail": "dbo.table_saleslog_detail"}),
    )


def test_load_holdout_buyers_prefers_fact_sales_line(monkeypatch):
    cfg = _make_cfg(use_line_items=True)
    cutoff = pd.Timestamp("2024-06-30")

    monkeypatch.setattr(hd, "get_curated_connection", lambda: object())

    def raise_on_db():
        raise AssertionError("get_db_connection should not be called when fact_sales_line is preferred")

    monkeypatch.setattr(hd, "get_db_connection", raise_on_db)
    monkeypatch.setattr(hd, "validate_connection", lambda engine: True)

    captured = {}

    def fake_read_sql(query, engine, params):
        captured["query"] = query
        captured["params"] = params
        return pd.DataFrame(
            {
                "customer_id": [101, 101, 202],
                "rec_date": [
                    pd.Timestamp("2024-07-15"),
                    pd.Timestamp("2024-07-20"),
                    pd.Timestamp("2024-08-01"),
                ],
                "gp_amount": [10.0, -2.0, 5.0],
            }
        )

    monkeypatch.setattr(hd.pd, "read_sql_query", fake_read_sql)

    data = hd.load_holdout_buyers(cfg, "Solidworks", cutoff, 3, source_override="db")

    assert data.source == "db"
    assert list(data.buyers.dropna().astype(int)) == [101, 202]
    assert "fact_sales_line" in captured["query"]
    assert captured["params"]["div_lower"] == "solidworks"
    assert data.realized_gp is not None
    gp_lookup = dict(zip(data.realized_gp["customer_id"].astype(int), data.realized_gp["holdout_gp"]))
    assert pytest.approx(gp_lookup[101], abs=1e-6) == 8.0
    assert pytest.approx(gp_lookup[202], abs=1e-6) == 5.0


def test_load_holdout_buyers_falls_back_to_sales_detail(monkeypatch):
    cfg = _make_cfg(use_line_items=False)
    cutoff = pd.Timestamp("2024-06-30")

    def raise_on_curated():
        raise AssertionError("get_curated_connection should not be called when line items are disabled")

    monkeypatch.setattr(hd, "get_curated_connection", raise_on_curated)
    monkeypatch.setattr(hd, "get_db_connection", lambda: object())
    monkeypatch.setattr(hd, "validate_connection", lambda engine: True)

    captured = {}

    def fake_read_sql(query, engine, params):
        captured["query"] = query
        captured["params"] = params
        return pd.DataFrame(columns=["customer_id", "rec_date"])

    monkeypatch.setattr(hd.pd, "read_sql_query", fake_read_sql)

    data = hd.load_holdout_buyers(cfg, "Scanning", cutoff, 3, source_override="db")

    assert data.source == "db"
    assert (data.buyers is None) or data.buyers.empty
    assert data.realized_gp is None or data.realized_gp.empty
    assert "dbo.table_saleslog_detail" in captured["query"]
    assert captured["params"]["div_lower"] == "scanning"
