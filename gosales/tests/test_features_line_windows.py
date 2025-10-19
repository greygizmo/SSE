from __future__ import annotations

import pandas as pd
import pytest

from gosales.features.engine import _compute_line_window_features


def test_line_window_features_basic_canonical_margin_currency():
    line_df = pd.DataFrame(
        {
            "customer_id": ["1", "1", "2", "2"],
            "order_date": pd.to_datetime(["2024-02-01", "2024-03-01", "2024-02-15", "2024-03-05"]),
            "division_canonical": ["SOLIDWORKS", "UNKNOWN", "SERVICES", "SERVICES"],
            "Revenue_usd": [100.0, 50.0, 200.0, -20.0],
            "GP_usd": [30.0, 5.0, 80.0, -10.0],
            "Term_GP_usd": [20.0, 2.0, 70.0, -5.0],
            "SalesOrder_Currency": ["USD", "CAD", "CAD", "CAD"],
            "USD_CAD_Conversion_rate": [None, 1.3, 1.2, 1.2],
            "item_rollup": ["core", "unknown", "services", "services"],
            "Sales_Order": ["SO1", "SO2", "SO3", "SO4"],
            "is_return_line": [0, 0, 0, 1],
        }
    )

    frames = _compute_line_window_features(
        line_df,
        cutoff_dt=pd.Timestamp("2024-03-31"),
        windows=[3],
        target_division="CAD",
        use_usd=True,
        enable_canonical=True,
        enable_margin=True,
        enable_currency=True,
        enable_diversity=True,
        enable_returns=True,
        enable_order_comp=True,
    )

    assert len(frames) == 1
    result = frames[0].set_index("customer_id")

    assert pytest.approx(result.loc["1", "margin__gp_rate__3m"], abs=1e-6) == pytest.approx((30 + 5) / 150, abs=1e-6)
    assert pytest.approx(result.loc["1", "margin__term_gp_rate__3m"], abs=1e-6) == pytest.approx((20 + 2) / 150, abs=1e-6)
    assert pytest.approx(result.loc["1", "currency__cad_share__3m"], abs=1e-6) == pytest.approx(50 / 150, abs=1e-6)
    assert pytest.approx(result.loc["1", "currency__fx_applied_share__3m"], abs=1e-6) == 0.5
    assert pytest.approx(result.loc["1", "xdiv__canon_unknown_share__3m"], abs=1e-6) == pytest.approx(50 / 150, abs=1e-6)
    assert pytest.approx(result.loc["1", "xdiv__canon_top1_share__3m"], abs=1e-6) == pytest.approx(100 / 150, abs=1e-6)
    assert pytest.approx(result.loc["1", "diversity__rollup_hhi__3m"], abs=1e-6) == pytest.approx((100 / 150) ** 2 + (50 / 150) ** 2, abs=1e-6)
    assert result.loc["1", "returns__line_share__3m"] == pytest.approx(0.0)
    assert pytest.approx(result.loc["1", "order__avg_revenue_per_line_usd__3m"], abs=1e-6) == pytest.approx(150 / 2, abs=1e-6)
    assert pytest.approx(result.loc["1", "order__avg_revenue_per_order_usd__3m"], abs=1e-6) == pytest.approx(75.0, abs=1e-6)

    assert pytest.approx(result.loc["2", "margin__gp_rate__3m"], abs=1e-6) == pytest.approx((80 - 10) / 180, abs=1e-6)
    assert pytest.approx(result.loc["2", "currency__cad_share__3m"], abs=1e-6) == pytest.approx(1.0, abs=1e-6)
    assert result.loc["2", "diversity__rollup_nunique__3m"] == pytest.approx(1.0)
    assert result.loc["2", "diversity__rollup_hhi__3m"] == pytest.approx(1.0)
    assert pytest.approx(result.loc["2", "returns__line_share__3m"], abs=1e-6) == pytest.approx(0.5, abs=1e-6)
    assert pytest.approx(result.loc["2", "returns__revenue_share__3m"], abs=1e-6) == pytest.approx(-20 / 180, abs=1e-6)
    assert pytest.approx(result.loc["2", "order__avg_revenue_per_line_usd__3m"], abs=1e-6) == pytest.approx(180 / 2, abs=1e-6)
    assert pytest.approx(result.loc["2", "order__avg_revenue_per_order_usd__3m"], abs=1e-6) == pytest.approx(90.0, abs=1e-6)
