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


def test_active_assets_and_reinclude_adjust_counts():
    df = pd.DataFrame(
        {
            "division_name": ["A", "A", "A", "B"],
            "owned_division_pre_cutoff": [True, False, False, False],
            "owns_assets_div_a": [False, True, True, False],
            "owns_assets_div_b": [False, False, False, False],
            "former_owner_div_a": [True, False, False, False],
            "assets_days_since_last_expiration_div_a": [120, 10, 45, 0],
        }
    )

    def make_cfg(**elig_kwargs):
        return SimpleNamespace(
            whitespace=SimpleNamespace(
                eligibility=WhitespaceEligibilityConfig(**elig_kwargs)
            )
        )

    base_cfg = make_cfg(exclude_if_owned_ever=True)
    out_base, counts_base = _apply_eligibility(df, base_cfg)
    assert counts_base["owned_excluded"] == 1
    assert len(out_base) == 3

    active_cfg = make_cfg(
        exclude_if_owned_ever=True,
        exclude_if_active_assets=True,
    )
    out_active, counts_active = _apply_eligibility(df, active_cfg)
    assert counts_active["owned_excluded"] == 3
    assert len(out_active) == 1

    reinclude_cfg = make_cfg(
        exclude_if_owned_ever=True,
        exclude_if_active_assets=True,
        reinclude_if_assets_expired_days=90,
    )
    out_reinclude, counts_reinclude = _apply_eligibility(df, reinclude_cfg)

    # Active assets exclusion should drop the two active accounts while the
    # reinclusion window allows the expired owner to return.
    assert counts_reinclude["owned_excluded"] == 2
    assert len(out_reinclude) == 2
    assert 0 in out_reinclude.index
