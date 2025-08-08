from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from gosales.features.engine import create_feature_matrix


def _seed(engine):
    fact = pd.DataFrame([
        {"customer_id": 1, "order_date": "2024-01-01", "product_division": "Solidworks", "product_sku": "SWX_Core", "gross_profit": 100, "quantity": 1},
        {"customer_id": 1, "order_date": "2023-12-15", "product_division": "Services", "product_sku": "Training", "gross_profit": 50, "quantity": 1},
        {"customer_id": 2, "order_date": "2023-11-01", "product_division": "Solidworks", "product_sku": "SWX_Core", "gross_profit": 10, "quantity": 1},
    ])
    fact.to_sql("fact_transactions", engine, if_exists="replace", index=False)
    pd.DataFrame({"customer_id": [1, 2]}).to_sql("dim_customer", engine, if_exists="replace", index=False)


def test_golden_small_features(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{tmp_path}/golden.db")
    _seed(eng)
    # Fix config windows to 3m so only 2023-10-02 to 2024-01-01 range contributes
    from gosales.utils import config as cfgmod
    orig = cfgmod.load_config
    def _fake():
        c = orig()
        c.features.windows_months = [3]
        c.features.gp_winsor_p = 1.0
        return c
    monkeypatch.setattr(cfgmod, "load_config", _fake)

    fm = create_feature_matrix(eng, "Solidworks", cutoff_date="2024-01-01", prediction_window_months=3)
    pdf = fm.to_pandas().sort_values('customer_id')
    # Customer 1: in last 3m, Solidworks(100) + Services(50)
    c1 = pdf[pdf['customer_id']==1].iloc[0]
    assert int(c1['rfm__all__tx_n__3m']) == 2
    assert abs(float(c1['rfm__all__gp_sum__3m']) - 150.0) < 1e-6
    assert abs(float(c1['margin__all__gp_pct__3m']) - (150.0/150.0)) < 1e-6
    # Customer 2: only older SWX transaction is outside 3m? (2023-11-01 within 3m of 2024-01-01)
    c2 = pdf[pdf['customer_id']==2].iloc[0]
    assert int(c2['rfm__all__tx_n__3m']) == 1
    assert abs(float(c2['rfm__all__gp_sum__3m']) - 10.0) < 1e-6
    assert abs(float(c2['margin__all__gp_pct__3m']) - 1.0) < 1e-6


