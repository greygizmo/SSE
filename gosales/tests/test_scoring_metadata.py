import json
from pathlib import Path
from unittest.mock import patch

import joblib
import pytest

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
