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



def test_discover_available_models_normalizes_division(tmp_path):
    setup_score_customers_import()
    from gosales.pipeline.score_customers import discover_available_models

    models_root = tmp_path
    model_dir = models_root / "Multi Word_model"
    model_dir.mkdir()

    available = discover_available_models(models_root)
    assert "multi word" in available
    assert available["multi word"] == model_dir


def test_model_without_metadata_survives_pruning(tmp_path):
    setup_score_customers_import()
    from gosales.pipeline.score_customers import (
        discover_available_models,
        _filter_models_by_targets,
    )

    models_root = tmp_path
    model_dir = models_root / "SW_Inspection_model"
    model_dir.mkdir()

    available = discover_available_models(models_root)
    # Discovered key should be casefolded but retain underscore
    assert "sw_inspection" in available

    filtered = _filter_models_by_targets(available, {"SW Inspection"})
    assert "sw_inspection" in filtered
    assert filtered["sw_inspection"] == model_dir


def test_filtering_prefers_primary_and_handles_variants(tmp_path):
    setup_score_customers_import()
    from gosales.pipeline.score_customers import (
        discover_available_models,
        _filter_models_by_targets,
    )

    models_root = tmp_path
    # Primary model with hyphen
    primary_dir = models_root / "SW-Inspection_model"
    primary_dir.mkdir()
    # Cold-start variant with underscore
    cold_dir = models_root / "SW_Inspection_cold_model"
    cold_dir.mkdir()

    available = discover_available_models(models_root)
    # Primary entry exists; discovery keeps distinct keys for hyphen/underscore
    assert "sw-inspection" in available
    assert available["sw-inspection"] == primary_dir

    # Filtering should match targets regardless of case/space/hyphen/underscore
    targets = ["SW   Inspection", "", None]
    filtered = _filter_models_by_targets(available, targets)  # type: ignore[arg-type]
    assert "sw-inspection" in filtered and filtered["sw-inspection"] == primary_dir
    # Cold model uses a different discovered key (with "_cold"); it should be pruned
    assert "sw_inspection_cold" not in filtered
