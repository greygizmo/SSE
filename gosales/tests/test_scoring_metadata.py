import json
from pathlib import Path
from unittest.mock import patch

import joblib
import polars as pl
import pytest

import gosales.pipeline.score_customers as sc
from gosales.pipeline.score_customers import (
    MissingModelMetadataError,
    score_customers_for_division,
)


def test_missing_metadata_fields_raises(tmp_path):
    model_dir = tmp_path / "solidworks_model"
    model_dir.mkdir()
    joblib.dump({"dummy": True}, model_dir / "model.pkl")
    (model_dir / "metadata.json").write_text(
        json.dumps({"division": "Solidworks"}), encoding="utf-8"
    )

    with patch(
        "gosales.pipeline.score_customers.mlflow.sklearn.load_model",
        side_effect=Exception("mlflow not used"),
    ):
        with pytest.raises(MissingModelMetadataError):
            score_customers_for_division(None, "Solidworks", model_dir)


def test_missing_metadata_fields_use_fallback_arguments(tmp_path):
    model_dir = tmp_path / "solidworks_model"
    model_dir.mkdir()
    joblib.dump({"dummy": True}, model_dir / "model.pkl")
    (model_dir / "metadata.json").write_text(
        json.dumps({"division": "Solidworks"}), encoding="utf-8"
    )

    fallback_cutoff = "2023-01-31"
    fallback_window = 3
    run_manifest: dict = {}

    empty_matrix = pl.DataFrame(
        {
            "customer_id": pl.Series(name="customer_id", values=[], dtype=pl.Utf8),
            "bought_in_division": pl.Series(
                name="bought_in_division", values=[], dtype=pl.Int64
            ),
        }
    )

    with (
        patch(
            "gosales.pipeline.score_customers.mlflow.sklearn.load_model",
            side_effect=Exception("mlflow not used"),
        ),
        patch(
            "gosales.pipeline.score_customers.create_feature_matrix",
            return_value=empty_matrix,
        ) as mock_feature_matrix,
        patch(
            "gosales.pipeline.score_customers._prepare_customer_names",
            return_value=None,
        ),
        patch(
            "gosales.pipeline.score_customers._should_use_batched_scoring",
            return_value=False,
        ),
    ):
        result = score_customers_for_division(
            None,
            "Solidworks",
            model_dir,
            run_manifest=run_manifest,
            cutoff_date=fallback_cutoff,
            prediction_window_months=fallback_window,
        )

    assert isinstance(result, pl.DataFrame)
    mock_feature_matrix.assert_called_once_with(
        None, "solidworks", fallback_cutoff, fallback_window
    )

    alerts = run_manifest.get("alerts", [])
    assert alerts, "Expected fallback alert to be recorded"
    alert = alerts[0]
    assert alert["severity"] == "warning"
    assert "used fallback arguments" in alert["message"].lower()


def test_generate_scoring_outputs_passes_pipeline_defaults(tmp_path, monkeypatch):
    model_dir = tmp_path / "solidworks_model"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text(
        json.dumps({"division": "Solidworks"}),
        encoding="utf-8",
    )

    fallback_cutoff = "2023-05-31"
    fallback_window = 4
    run_manifest: dict = {}

    monkeypatch.setattr(
        sc,
        "discover_available_models",
        lambda: {"Solidworks": model_dir},
    )
    monkeypatch.setattr(
        sc,
        "_filter_models_by_targets",
        lambda available, targets: available,
    )

    captured: dict[str, object] = {}

    def _fake_score(
        engine,
        division_name,
        model_path,
        *,
        run_manifest,
        cutoff_date,
        prediction_window_months,
    ):
        captured["engine"] = engine
        captured["division_name"] = division_name
        captured["model_path"] = model_path
        captured["run_manifest"] = run_manifest
        captured["cutoff_date"] = cutoff_date
        captured["prediction_window_months"] = prediction_window_months
        return pl.DataFrame({"division_name": []})

    monkeypatch.setattr(sc, "score_customers_for_division", _fake_score)

    sc.generate_scoring_outputs(
        engine=None,
        run_manifest=run_manifest,
        cutoff_date=fallback_cutoff,
        prediction_window_months=fallback_window,
    )

    assert captured["cutoff_date"] == fallback_cutoff
    assert captured["prediction_window_months"] == fallback_window
    assert captured["division_name"].lower() == "solidworks"
    assert run_manifest.get("whitespace_artifact") is None
