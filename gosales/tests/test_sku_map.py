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



