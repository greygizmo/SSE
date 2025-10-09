from sqlalchemy import create_engine
import pandas as pd
from gosales.pipeline.score_customers import generate_whitespace_opportunities


def _seed(engine):
    transactions = pd.DataFrame(
        [
            {"customer_id": 1, "order_date": "2024-01-01", "product_division": "A", "gross_profit": 100},
            {"customer_id": 1, "order_date": "2024-02-15", "product_division": "B", "gross_profit": 50},
            {"customer_id": 2, "order_date": "2023-12-01", "product_division": "A", "gross_profit": 20},
            {"customer_id": 3, "order_date": "2024-01-05", "product_division": "B", "gross_profit": 80},
        ]
    )
    transactions.to_sql("fact_transactions", engine, if_exists="replace", index=False)
    pd.DataFrame({"customer_id": [1, 2, 3]}).to_sql("dim_customer", engine, if_exists="replace", index=False)


def test_whitespace_score_is_continuous(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/ws.db")
    _seed(eng)
    df = generate_whitespace_opportunities(eng)
    assert not df.is_empty()
    scores = df["whitespace_score"].to_list()
    assert min(scores) >= 0.0 and max(scores) <= 1.0
    assert not all(round(s, 1) in {0.5, 0.6, 0.8} for s in scores)


def test_whitespace_score_handles_all_null_dates(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/ws_null_dates.db")
    transactions = pd.DataFrame([
        {"customer_id": "1", "order_date": None, "product_division": "A", "gross_profit": 120.0},
        {"customer_id": "2", "order_date": None, "product_division": "B", "gross_profit": 95.0},
    ])
    transactions.to_sql("fact_transactions", eng, if_exists="replace", index=False)
    pd.DataFrame({"customer_id": ["1", "2"]}).to_sql("dim_customer", eng, if_exists="replace", index=False)

    df = generate_whitespace_opportunities(eng)

    assert not df.is_empty()
    assert df["whitespace_score"].is_not_null().all()


def test_whitespace_uses_aggregated_queries(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{tmp_path}/ws_queries.db")
    _seed(eng)

    observed_queries: list[str] = []
    original_read_sql = pd.read_sql

    def _spy_read_sql(sql, con=None, *args, **kwargs):
        if isinstance(sql, str):
            observed_queries.append(sql)
            assert "SELECT *" not in sql.upper()
        return original_read_sql(sql, con, *args, **kwargs)

    monkeypatch.setattr(pd, "read_sql", _spy_read_sql)

    df = generate_whitespace_opportunities(eng)
    assert not df.is_empty()
    assert observed_queries
    assert any("GROUP BY customer_id" in q or "DISTINCT product_division" in q for q in observed_queries)


def test_whitespace_respects_cutoff(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/ws_cutoff.db")
    _seed(eng)

    df_future = generate_whitespace_opportunities(eng)
    df_cut = generate_whitespace_opportunities(eng, cutoff_date="2024-01-31")

    def _has_row(df, customer, division):
        if {"customer_id", "whitespace_division"}.issubset(set(df.columns)):
            pdf = df.to_pandas()
            mask = (pdf["customer_id"].astype(str) == str(customer)) & (
                pdf["whitespace_division"].str.lower() == division.lower()
            )
            return mask.any()
        return False

    assert not _has_row(df_future, 1, "B"), "Future-aware heuristic should treat post-cutoff purchase as owned."
    assert _has_row(df_cut, 1, "B"), "Cutoff-aware heuristic must expose divisions purchased only after cutoff."
