import pandas as pd
from unittest.mock import MagicMock

from gosales.monitoring.data_collector import MonitoringDataCollector


def test_type_consistency_score_penalizes_inconsistent_types(monkeypatch):
    collector = MonitoringDataCollector()

    consistent_df = pd.DataFrame(
        {
            "source_table": ["dim_customer"] * 3 + ["fact_transactions"] * 3,
            "customer_id": [1, 2, 3, 4, 5, 6],
        }
    )
    inconsistent_df = pd.DataFrame(
        {
            "source_table": ["dim_customer", "dim_customer", "fact_transactions", "fact_transactions"],
            "customer_id": [1, 2, 3, "4"],
        }
    )

    monkeypatch.setattr(
        "gosales.monitoring.data_collector.get_db_connection", lambda: object()
    )
    read_sql_mock = MagicMock(side_effect=[consistent_df, inconsistent_df])
    monkeypatch.setattr(
        "gosales.monitoring.data_collector.pd.read_sql_query", read_sql_mock
    )

    consistent_score = collector._calculate_type_consistency_score()
    inconsistent_score = collector._calculate_type_consistency_score()

    assert inconsistent_score < consistent_score
    assert read_sql_mock.call_count == 2
