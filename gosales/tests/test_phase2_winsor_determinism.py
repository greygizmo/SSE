from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from gosales.features.engine import create_feature_matrix


def _seed_two_customers(engine, cutoff: str):
    # Two customers, both within 3m of cutoff
    cutoff_dt = pd.to_datetime(cutoff)
    d1 = (cutoff_dt - pd.DateOffset(days=10)).date().isoformat()
    d2 = (cutoff_dt - pd.DateOffset(days=5)).date().isoformat()
    fact = pd.DataFrame([
        {"customer_id": 1, "order_date": d1, "product_division": "Solidworks", "product_sku": "SWX_Core", "gross_profit": 1000, "quantity": 1},
        {"customer_id": 2, "order_date": d2, "product_division": "Solidworks", "product_sku": "SWX_Core", "gross_profit": 0, "quantity": 1},
    ])
    fact.to_sql("fact_transactions", engine, if_exists="replace", index=False)
    pd.DataFrame({"customer_id": [1, 2]}).to_sql("dim_customer", engine, if_exists="replace", index=False)


def test_determinism_in_memory(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/determinism.db")
    _seed_two_customers(eng, "2024-01-31")
    fm1 = create_feature_matrix(eng, "Solidworks", cutoff_date="2024-01-31", prediction_window_months=3)
    fm2 = create_feature_matrix(eng, "Solidworks", cutoff_date="2024-01-31", prediction_window_months=3)
    p1 = fm1.to_pandas().sort_values(list(fm1.columns)).reset_index(drop=True)
    p2 = fm2.to_pandas().sort_values(list(fm2.columns)).reset_index(drop=True)
    # Exact equality expected
    pd.testing.assert_frame_equal(p1, p2, check_like=False)


def test_winsorization_effect(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{tmp_path}/winsor.db")
    cutoff = "2024-01-31"
    _seed_two_customers(eng, cutoff)

    # Monkeypatch config to set gp_winsor_p to 0.5 and include 3m window
    from gosales.utils import config as cfgmod
    orig = cfgmod.load_config

    def _fake():
        c = orig()
        c.features.gp_winsor_p = 0.5
        c.features.windows_months = [3]
        return c

    monkeypatch.setattr(cfgmod, "load_config", _fake)

    fm = create_feature_matrix(eng, "Solidworks", cutoff_date=cutoff, prediction_window_months=3)
    pdf = fm.to_pandas()
    # rfm__all__gp_sum__3m across customers would be [1000, 0]; winsor at p=0.5 caps upper at median=500
    # So customer 1 should be clipped to ~500 (exact 500.0), customer 2 remains 0
    vals = pdf[["customer_id", "rfm__all__gp_sum__3m"]].set_index("customer_id")["rfm__all__gp_sum__3m"].to_dict()
    assert abs(vals.get(1, 0.0) - 500.0) < 1e-6
    assert abs(vals.get(2, -1.0) - 0.0) < 1e-6


