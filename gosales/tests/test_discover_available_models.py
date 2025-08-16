import types
import sys
from pathlib import Path


def setup_score_customers_import():
    mlflow_stub = types.SimpleNamespace(sklearn=types.SimpleNamespace())
    features_engine_stub = types.SimpleNamespace(create_feature_matrix=lambda *a, **k: None)
    rank_whitespace_stub = types.SimpleNamespace(rank_whitespace=None, save_ranked_whitespace=None, RankInputs=None)
    deciles_stub = types.SimpleNamespace(emit_validation_artifacts=lambda *a, **k: None)
    schema_stub = types.SimpleNamespace(
        validate_icp_scores_schema=lambda *a, **k: None,
        validate_whitespace_schema=lambda *a, **k: None,
        write_schema_report=lambda *a, **k: None,
    )
    drift_stub = types.SimpleNamespace(check_drift_and_emit_alerts=lambda *a, **k: None)

    modules = {
        'mlflow': mlflow_stub,
        'mlflow.sklearn': mlflow_stub.sklearn,
        'gosales.features.engine': features_engine_stub,
        'gosales.pipeline.rank_whitespace': rank_whitespace_stub,
        'gosales.validation.deciles': deciles_stub,
        'gosales.validation.schema': schema_stub,
        'gosales.monitoring.drift': drift_stub,
    }
    for name, mod in modules.items():
        sys.modules.setdefault(name, mod)



def test_discover_available_models_preserves_casing(tmp_path):
    setup_score_customers_import()
    from gosales.pipeline.score_customers import discover_available_models

    models_root = tmp_path
    model_dir = models_root / "Multi Word_model"
    model_dir.mkdir()

    available = discover_available_models(models_root)
    assert "Multi Word" in available
    assert available["Multi Word"] == model_dir
