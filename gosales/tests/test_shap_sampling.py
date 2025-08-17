import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gosales.models import train as train_mod
from gosales.utils import paths


def test_shap_sampling_controls(monkeypatch, tmp_path, caplog):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(100, 3), columns=list("abc"))
    y = (rng.rand(100) > 0.5).astype(int)
    model = LogisticRegression().fit(X, y)
    df_final = pd.DataFrame({"customer_id": np.arange(len(X))})
    feature_names = list(X.columns)

    # Enabled case
    out_enabled = tmp_path / "enabled"
    monkeypatch.setattr(paths, "OUTPUTS_DIR", out_enabled)
    monkeypatch.setattr(train_mod, "OUTPUTS_DIR", out_enabled)
    caplog.set_level("WARNING")
    artifacts = train_mod._maybe_export_shap(
        model,
        X,
        df_final,
        "Test",
        feature_names,
        shap_sample=5,
        shap_max_rows=200,
        seed=0,
    )
    assert (out_enabled / "shap_global_test.csv").exists()
    assert "shap_global_test.csv" in artifacts

    # Disabled via shap_sample=0
    out_disabled = tmp_path / "disabled"
    monkeypatch.setattr(paths, "OUTPUTS_DIR", out_disabled)
    monkeypatch.setattr(train_mod, "OUTPUTS_DIR", out_disabled)
    caplog.clear()
    artifacts = train_mod._maybe_export_shap(
        model,
        X,
        df_final,
        "Test",
        feature_names,
        shap_sample=0,
        shap_max_rows=200,
        seed=0,
    )
    assert artifacts == {}
    assert "SHAP sample N is zero" in caplog.text
    assert not list(out_disabled.glob("*.csv"))

    # Skip when dataset too large
    out_large = tmp_path / "large"
    monkeypatch.setattr(paths, "OUTPUTS_DIR", out_large)
    monkeypatch.setattr(train_mod, "OUTPUTS_DIR", out_large)
    caplog.clear()
    artifacts = train_mod._maybe_export_shap(
        model,
        X,
        df_final,
        "Test",
        feature_names,
        shap_sample=5,
        shap_max_rows=10,
        seed=0,
    )
    assert artifacts == {}
    assert "exceeding threshold" in caplog.text
    assert not list(out_large.glob("*.csv"))

