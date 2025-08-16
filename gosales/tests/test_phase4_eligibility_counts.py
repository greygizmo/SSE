import pandas as pd
from types import SimpleNamespace

from gosales.pipeline.rank_whitespace import _apply_eligibility
from gosales.utils.config import WhitespaceEligibilityConfig


def test_eligibility_counts_sum_to_dropped_rows():
    df = pd.DataFrame(
        {
            "owned_division_pre_cutoff": [True, False, False, True, False, False],
            "days_since_last_contact": [0, 5, 1000, 5, 1000, 1000],
            "has_open_deal": [False, True, False, True, False, True],
            "region_match": [True, False, True, True, False, True],
        }
    )
    elig_cfg = WhitespaceEligibilityConfig(
        exclude_if_owned_ever=True,
        exclude_if_recent_contact_days=30,
        exclude_if_open_deal=True,
        require_region_match=True,
    )
    cfg = SimpleNamespace(whitespace=SimpleNamespace(eligibility=elig_cfg))

    out, counts = _apply_eligibility(df, cfg)

    assert counts["start_rows"] == len(df)
    dropped = counts["start_rows"] - counts["kept_rows"]
    total_excl = (
        counts["owned_excluded"]
        + counts["recent_contact_excluded"]
        + counts["open_deal_excluded"]
        + counts["region_mismatch_excluded"]
    )
    assert dropped == total_excl
    assert counts["kept_rows"] == len(out)
    assert out["_eligible"].dtype == bool
    assert out["_eligible"].all()
