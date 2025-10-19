import polars as pl
import pytest

from gosales.etl import sales_line as sl
from gosales.utils.config import LineItemsBehavior


@pytest.mark.parametrize(
    ("goal", "rollup", "fallback", "expected"),
    [
        ("Draftsight", "Draftsight", None, "Solidworks"),
        ("GeoMagic", "GeoMagic", None, "Scanning"),
        ("CAD", None, None, "Solidworks"),
        (None, None, "Services", "Services"),
        (None, None, None, sl.UNKNOWN_DIVISION),
    ],
)
def test_resolve_canonical_division(goal, rollup, fallback, expected):
    assert sl._resolve_canonical_division(goal, rollup, fallback) == expected


def test_load_product_tag_mapping_produces_expected_overrides(monkeypatch):
    dummy_mapping = pl.DataFrame(
        {
            "product_internal_id": ["1", "2"],
            sl.ITEM_ROLLUP_COLUMN: ["Draftsight", "GeoMagic"],
            sl.DIVISION_GOAL_COLUMN: ["Draftsight", "GeoMagic"],
        }
    )

    def fake_loader(engine, cfg, sources_cfg):
        return dummy_mapping

    monkeypatch.setattr(sl, "_load_product_tag_mapping", fake_loader)

    rollup_goal, rollup_division = sl.get_rollup_goal_mappings(engine="eng", cfg=object())
    draftsight_key = sl._normalize_rollup_label("Draftsight")
    geomagic_key = sl._normalize_rollup_label("GeoMagic")

    assert rollup_goal[draftsight_key] == "Draftsight"
    assert rollup_division[draftsight_key] == "Solidworks"
    assert rollup_goal[geomagic_key] == "GeoMagic"
    assert rollup_division[geomagic_key] == "Scanning"


def test_attach_division_metadata_prefers_order_tags(monkeypatch):
    base = pl.DataFrame(
        {
            "Item_internalid": ["123"],
            "Division": ["Legacy"],
            "Revenue": [100.0],
        }
    )

    order_mapping = pl.DataFrame(
        {
            "product_internal_id": ["123"],
            sl.ORDER_TAG_ROLLUP_COLUMN: ["Draftsight"],
            sl.ORDER_GOAL_COLUMN: ["Draftsight"],
        }
    )
    product_mapping = pl.DataFrame(
        {
            "product_internal_id": ["123"],
            sl.ITEM_ROLLUP_COLUMN: ["GeoMagic"],
            sl.DIVISION_GOAL_COLUMN: ["Scanning"],
        }
    )

    monkeypatch.setattr(sl, "_load_order_tag_mapping", lambda *args, **kwargs: order_mapping)
    monkeypatch.setattr(sl, "_load_product_tag_mapping", lambda *args, **kwargs: product_mapping)

    enriched = sl._attach_division_metadata(base, engine=None, cfg=None, sources_cfg=None)

    assert enriched[sl.ITEM_ROLLUP_COLUMN][0] == "Draftsight"
    assert enriched[sl.DIVISION_GOAL_COLUMN][0] == "Draftsight"
    assert enriched[sl.DIVISION_CANONICAL_COLUMN][0] == "Solidworks"


def test_apply_behavior_config_filters_line_types_and_returns():
    frame = pl.DataFrame(
        {
            "Rev_type": ["Tax", "Sale"],
            "Revenue": [-5.0, 20.0],
        }
    )
    behavior = LineItemsBehavior(
        exclude_line_types=["tax"],
        return_treatment="exclude_returns",
        kit_handling="include_parent",
    )

    filtered = sl._apply_behavior_config(frame, behavior, revenue_column="Revenue")

    assert filtered.height == 1
    assert filtered["Revenue"][0] == 20.0


def test_apply_behavior_config_marks_returns_when_separate_flag():
    frame = pl.DataFrame({"Revenue": [10.0, -2.0, 0.0]})
    behavior = LineItemsBehavior(return_treatment="separate_flag", kit_handling="include_parent")

    flagged = sl._apply_behavior_config(frame, behavior, revenue_column="Revenue")

    assert flagged.height == frame.height
    assert flagged["is_return_line"].to_list() == [False, True, False]


def test_apply_behavior_config_removes_kit_parents(monkeypatch):
    frame = pl.DataFrame(
        {
            "Revenue": [15.0, 10.0],
            "is_kit_parent": [False, True],
        }
    )
    behavior = LineItemsBehavior(kit_handling="prefer_children")

    pruned = sl._apply_behavior_config(frame, behavior, revenue_column="Revenue")

    assert pruned.height == 1
    assert pruned["Revenue"][0] == 15.0
