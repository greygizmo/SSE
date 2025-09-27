from gosales.utils.normalize import normalize_division


def test_normalize_division_casefolds_and_strips():
    assert normalize_division("  SolidWorks  ") == "solidworks"
    assert normalize_division("SW ELECTRICAL") == "sw electrical"
    assert normalize_division(None) == ""
