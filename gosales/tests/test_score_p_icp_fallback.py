import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gosales.pipeline.rank_whitespace import _score_p_icp


def test_score_p_icp_ignores_label_and_extra_columns():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(X, y)

    df = pd.DataFrame(X, columns=["f1", "f2"])
    df["extra_numeric"] = 999.0
    df["label"] = y
    df["score"] = 0.1

    preds = _score_p_icp(df, model, feat_cols=None)
    expected = model.predict_proba(df[["f1", "f2"]])[:, 1]
    assert np.allclose(preds, expected)
