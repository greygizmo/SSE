from gosales.utils.normalize import normalize_model_key


def test_normalize_model_key_variants():
    cases = {
        None: "",
        "": "",
        "SW_Inspection": "sw inspection",
        " SW  Inspection ": "sw inspection",
        "SW-Inspection": "sw inspection",
        "sw inspection": "sw inspection",
    }
    for raw, expected in cases.items():
        assert normalize_model_key(raw) == expected

