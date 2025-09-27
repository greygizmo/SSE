import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gosales.models import train as train_module


class _DummyFeatureMatrix:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def write_parquet(self, path) -> None:  # pragma: no cover - side effects not needed in tests
        """Skip writing parquet during tests."""
        return None

    def is_empty(self) -> bool:
        return self._df.empty

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


class _DummyContext(dict):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.mark.parametrize("cutoffs", [["2023-01-01", "2023-02-01", "2023-03-01"]])
def test_phase3_results_capture_each_cutoff(monkeypatch, tmp_path, cutoffs):
    outputs_dir = tmp_path / "outputs"
    models_dir = tmp_path / "models"
    monkeypatch.setattr(train_module, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(train_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(train_module, "_HAS_SHAP", False)

    def fake_run_context(name: str):
        ctx = _DummyContext()
        ctx["write_manifest"] = lambda artifacts: None
        ctx["append_registry"] = lambda payload: None
        return ctx

    monkeypatch.setattr(train_module, "run_context", fake_run_context)

    modeling_cfg = SimpleNamespace(
        lr_grid={"l1_ratio": [0.0], "C": [1.0]},
        lgbm_grid={
            "num_leaves": [31],
            "min_data_in_leaf": [20],
            "learning_rate": [0.1],
            "feature_fraction": [0.9],
            "bagging_fraction": [0.9],
        },
        class_weight="balanced",
        seed=42,
        folds=3,
        safe_divisions=[],
        use_scale_pos_weight=False,
        scale_pos_weight_cap=10.0,
        shap_max_rows=10000,
        top_k_percents=[10],
        sparse_isotonic_threshold_pos=1000,
    )
    cfg = SimpleNamespace(modeling=modeling_cfg, database=SimpleNamespace(strict_db=False))
    monkeypatch.setattr(train_module, "load_config", lambda _: cfg)
    monkeypatch.setattr(train_module, "get_curated_connection", lambda: object())
    monkeypatch.setattr(train_module, "get_db_connection", lambda: object())
    monkeypatch.setattr(train_module, "validate_connection", lambda engine: True)

    def fake_feature_matrix(engine, division, cutoff, window_months, **kwargs):
        digits = "".join(ch for ch in str(cutoff) if ch.isdigit())
        seed = int(digits or 1)
        rng = np.random.RandomState(seed % (2**32 - 1))
        n = 240
        x1 = rng.normal(loc=0.0, scale=1.0, size=n)
        x2 = rng.normal(loc=0.5, scale=1.2, size=n)
        bias = -0.3 if seed % 2 == 0 else 0.3
        logits = 0.5 * x1 + 0.4 * x2 + bias
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.rand(n) < p).astype(int)
        df = pd.DataFrame(
            {
                "customer_id": [f"{cutoff}-{i}" for i in range(n)],
                "feature_one": x1,
                "feature_two": x2,
                "bought_in_division": y,
            }
        )
        return _DummyFeatureMatrix(df)

    monkeypatch.setattr(train_module, "create_feature_matrix", fake_feature_matrix)

    cutoffs_arg = ",".join(cutoffs)
    train_module.main.callback(
        division="Acme",
        cutoffs=cutoffs_arg,
        window_months=1,
        models="logreg",
        calibration="platt",
        shap_sample=0,
        config="ignored",
        group_cv=False,
        purge_days=0,
        label_buffer_days=0,
        safe_mode=False,
        dry_run=False,
    )

    diag_path = outputs_dir / "diagnostics_acme.json"
    assert diag_path.exists(), "Diagnostics JSON should be emitted"
    diag = json.loads(diag_path.read_text())
    recorded_cutoffs = [row["cutoff"] for row in diag.get("results_grid", []) if row.get("model") == "logreg"]
    assert sorted(recorded_cutoffs) == sorted(cutoffs)
