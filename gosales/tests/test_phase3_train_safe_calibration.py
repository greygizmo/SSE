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

    def write_parquet(self, path) -> None:  # pragma: no cover
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


def _cfg_for_tests():
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
        stability=SimpleNamespace(
            min_positive_rate=0.0,
            min_positive_count=0,
            coverage_floor=0.0,
            penalty_lambda=0.0,
            cv_guard_max=1.0,
            sparse_family_prefixes=[],
            backbone_features=[],
            backbone_prefixes=[],
        ),
    )
    return SimpleNamespace(modeling=modeling_cfg, database=SimpleNamespace(strict_db=False))


def _run_with_matrix(
    monkeypatch,
    tmp_path,
    df: pd.DataFrame,
    models: str = "logreg",
    cutoffs: str = "2023-01-01",
    cfg_mutator=None,
):
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

    cfg = _cfg_for_tests()
    if cfg_mutator is not None:
        cfg_mutator(cfg)
    monkeypatch.setattr(train_module, "load_config", lambda _: cfg)
    monkeypatch.setattr(train_module, "get_curated_connection", lambda: object())
    monkeypatch.setattr(train_module, "get_db_connection", lambda: object())
    monkeypatch.setattr(train_module, "validate_connection", lambda engine: True)

    def fake_feature_matrix(engine, division, cutoff, window_months, **kwargs):
        return _DummyFeatureMatrix(df)

    monkeypatch.setattr(train_module, "create_feature_matrix", fake_feature_matrix)

    train_module.main.callback(
        division="Acme",
        cutoffs=cutoffs,
        window_months=1,
        models=models,
        calibration="platt,isotonic",
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
    return json.loads(diag_path.read_text())


def test_phase3_calibration_skips_records_reason(monkeypatch, tmp_path):
    # Build a tiny imbalance so train has only 1 positive per split → no feasible cv>=2
    n = 200
    rng = np.random.RandomState(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.zeros(n, dtype=int)
    # Force time-aware split and control class placement: 1 pos in valid, 1 in train
    recency = np.arange(1, n + 1)  # ascending → earliest rows go to validation slice
    y[0] = 1      # will fall into validation (first 20%)
    y[41] = 1     # will fall into training (remaining 80%)
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "feature_one": x1,
            "feature_two": x2,
            "rfm__all__recency_days__life": recency,
            "bought_in_division": y,
        }
    )

    diag = _run_with_matrix(monkeypatch, tmp_path, df, models="logreg")
    rows = [r for r in diag.get("results_grid", []) if r.get("model") == "logreg"]
    assert len(rows) == 1
    row = rows[0]
    assert row.get("calibration") == "none"
    # reason could be insufficient per-class or single_class_train depending on split
    assert isinstance(row.get("calibration_reason"), (str, type(None)))
    if row.get("calibration_reason"):
        assert ("insufficient" in row["calibration_reason"]) or ("single_class" in row["calibration_reason"])


def test_stability_objective_penalizes_peaky_models():
    results = [
        {"model": "stable", "cutoff": "c1", "rev_lift@10": 3.0, "lift@10": 3.0},
        {"model": "stable", "cutoff": "c2", "rev_lift@10": 3.2, "lift@10": 3.1},
        {"model": "peaky", "cutoff": "c1", "rev_lift@10": 5.0, "lift@10": 5.0},
        {"model": "peaky", "cutoff": "c2", "rev_lift@10": 1.0, "lift@10": 1.1},
    ]

    agg, winner, summary = train_module._aggregate_stability(results, [10], penalty_lambda=1.0, cv_guard=1.0)

    assert winner == "stable"
    assert summary["stable"]["objective"] > summary["peaky"]["objective"]
    assert any(row["model"] == "peaky" for _, row in agg.iterrows())


def test_model_card_contains_stability_block(monkeypatch, tmp_path):
    rng = np.random.RandomState(123)
    n = 120
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "feature_one": rng.normal(size=n),
            "feature_two": rng.normal(size=n),
            "rfm__all__recency_days__life": np.linspace(1, n, n),
            "rfm__all__gp_sum__12m": rng.gamma(shape=2.0, scale=10.0, size=n),
            "bought_in_division": (rng.rand(n) < 0.25).astype(int),
        }
    )

    def mutate(cfg):
        cfg.modeling.top_k_percents = [5, 10]
        cfg.modeling.stability.penalty_lambda = 0.3
        cfg.modeling.stability.cv_guard_max = 5.0

    _run_with_matrix(
        monkeypatch,
        tmp_path,
        df,
        models="logreg",
        cutoffs="2023-01-01,2023-02-01",
        cfg_mutator=mutate,
    )

    card_path = tmp_path / "outputs" / "model_card_acme.json"
    assert card_path.exists()
    card = json.loads(card_path.read_text(encoding="utf-8"))
    stability = card.get("stability")
    assert stability is not None
    assert "jaccard_topn" in stability
    assert "delta_holdout_rev_lift" in stability
    assert "final_rev_lift" in stability
    assert "final_prevalence" in stability


def test_sparse_family_drop_preserves_backbone(monkeypatch, tmp_path):
    rng = np.random.RandomState(99)
    n = 80
    y = np.zeros(n, dtype=int)
    y[[3, 35, 60]] = 1
    df = pd.DataFrame(
        {
            "customer_id": [f"CX{i:05d}" for i in range(n)],
            "rfm__all__recency_days__life": np.arange(1, n + 1),
            "rfm__all__gp_sum__12m": rng.gamma(shape=1.5, scale=5.0, size=n),
            "als_signal_sparse": rng.normal(size=n),
            "aux_sparse": np.where(rng.rand(n) < 0.1, rng.normal(size=n), np.nan),
            "bought_in_division": y,
        }
    )

    def mutate(cfg):
        cfg.modeling.top_k_percents = [10]
        cfg.modeling.stability.min_positive_count = 100
        cfg.modeling.stability.coverage_floor = 0.8
        cfg.modeling.stability.sparse_family_prefixes = ["als_"]
        cfg.modeling.stability.backbone_features = ["rfm__all__gp_sum__12m"]
        cfg.modeling.stability.backbone_prefixes = ["rfm__"]

    diag = _run_with_matrix(
        monkeypatch,
        tmp_path,
        df,
        models="logreg",
        cutoffs="2023-03-01",
        cfg_mutator=mutate,
    )

    feature_path = tmp_path / "models" / "acme_model" / "feature_list.json"
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    assert "als_signal_sparse" not in features
    assert "rfm__all__gp_sum__12m" in features
    coverage_drop = diag.get("coverage_drops", {})
    sparse_drop = diag.get("sparse_family_drops", {})
    assert any("aux_sparse" in drops for drops in coverage_drop.values())
    assert any("als_signal_sparse" in drops for drops in sparse_drop.values())
