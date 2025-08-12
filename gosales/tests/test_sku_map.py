from gosales.etl.sku_map import get_sku_mapping


def test_sku_map_basic_contract():
    mapping = get_sku_mapping()
    # Must include core Solidworks keys
    for key in [
        "SWX_Core",
        "SWX_Pro_Prem",
        "Core_New_UAP",
        "Pro_Prem_New_UAP",
        "PDM",
    ]:
        assert key in mapping
        assert "qty_col" in mapping[key]
        assert "division" in mapping[key]


def test_sku_map_extended_divisions_and_aliases():
    m = get_sku_mapping()
    # New divisions present
    assert m.get("CATIA", {}).get("division") == "CPE"
    assert m.get("Delmia_Apriso", {}).get("division") == "CPE"
    assert m.get("HV_Simulation", {}).get("division") == "CPE"
    assert m.get("Post_Processing", {}).get("division") == "Post_Processing"
    # Qty-only plastics captured for Simulation
    assert m.get("SW_Plastics", {}).get("division") == "Simulation"
    # AM software mapping
    assert m.get("AM_Software", {}).get("qty_col") == "AM_Software_Qty"






