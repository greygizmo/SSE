import numpy as np
import pandas as pd

from gosales.pipeline.score_customers import _score_p_icp


class DummyModel:
    def predict_proba(self, X: pd.DataFrame):
        # Ensure all columns are float and finite
        assert all(pd.api.types.is_float_dtype(dt) for dt in X.dtypes)
        assert np.isfinite(X.to_numpy()).all()
        # Return fixed probabilities
        return np.tile([0.2, 0.8], (len(X), 1))


def test_score_p_icp_handles_nan_inf_and_non_numeric():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, np.inf, -np.inf],
            "b": ["1", "two", None, 3],
        }
    )
    probs = _score_p_icp(DummyModel(), df)
    assert probs.shape == (4,)
    assert np.isfinite(probs).all()
