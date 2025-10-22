from __future__ import annotations

import polars as pl

from gosales.etl.sales_line import normalize_line_quantity


def test_normalize_line_quantity_candidates():
    frame = pl.DataFrame(
        {
            "Quantity": [1, 2, None, -3],
            "Revenue": [100.0, 200.0, 0.0, -50.0],
        }
    )

    out = normalize_line_quantity(frame, behavior_cfg=None)

    assert "Line_Qty" in out.columns
    vals = out.select("Line_Qty").to_series().to_list()
    # Absolute and null -> 0.0
    assert vals == [1.0, 2.0, 0.0, 3.0]

