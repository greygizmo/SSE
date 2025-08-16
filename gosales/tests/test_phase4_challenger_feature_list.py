import numpy as np
import pandas as pd
import types

import gosales.pipeline.rank_whitespace as rw


def test_rank_whitespace_handles_extra_feature(monkeypatch):
    n = 10
    df = pd.DataFrame(
        {
            "division_name": ["A"] * n,
            "customer_id": np.arange(n),
            "icp_score": np.linspace(0.1, 0.9, n),
            "mb_lift_max": np.linspace(0.1, 0.9, n),
            "als_f0": np.linspace(0.1, 0.9, n),
            "rfm__all__gp_sum__12m": np.linspace(10.0, 100.0, n),
        }
    )

    cfg = types.SimpleNamespace(
        whitespace=types.SimpleNamespace(
            challenger_enabled=True, challenger_model="lr", ev_cap_percentile=0.95
        )
    )
    import gosales.utils.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(rw, "CHALLENGER_FEAT_COLS", rw.CHALLENGER_FEAT_COLS + ["extra_col"])

    result = rw.rank_whitespace(rw.RankInputs(scores=df))
    assert "score" in result.columns
    assert "score_challenger" in result.columns
    assert len(result) == n
