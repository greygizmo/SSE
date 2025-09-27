from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from gosales.utils.logger import get_logger


logger = get_logger(__name__)


def compute_shap_reasons(model, X: pd.DataFrame, feature_cols: Iterable[str], top_k: int = 3) -> pd.DataFrame:
    """Compute per-row top-k SHAP reason codes for a tree model.

    Falls back gracefully if `shap` is unavailable.
    Returns a DataFrame with columns [reason_1, reason_2, reason_3].
    """
    def _empty_reasons() -> pd.DataFrame:
        return pd.DataFrame({f"reason_{i+1}": [None] * len(X) for i in range(top_k)})

    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.warning("SHAP not available: %s; skipping reason codes", exc)
        return _empty_reasons()

    # Unwrap calibrator wrapper if present
    model_for_shap = model
    try:
        from sklearn.calibration import CalibratedClassifierCV  # type: ignore

        if isinstance(model, CalibratedClassifierCV) and hasattr(model, "base_estimator"):
            base = getattr(model, "base_estimator", None) or getattr(model, "estimator", None)
            if base is not None:
                model_for_shap = base
    except Exception:  # pragma: no cover - best effort unwrap
        model_for_shap = model

    try:
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(X[feature_cols])

        # Normalize SHAP output to a (n_rows, n_features) numpy array
        shap_candidate = shap_values
        if isinstance(shap_candidate, list):
            if len(shap_candidate) == 2:
                shap_candidate = shap_candidate[1]
            elif len(shap_candidate) > 0:
                shap_candidate = shap_candidate[-1]
            else:
                return _empty_reasons()

        if hasattr(shap_candidate, "values"):
            shap_arr = np.asarray(shap_candidate.values)
        else:
            shap_arr = np.asarray(shap_candidate)

        if shap_arr.ndim == 3:
            if shap_arr.shape[0] == len(X):
                shap_arr = shap_arr[:, -1, :]
            else:
                shap_arr = shap_arr[-1]
        elif shap_arr.ndim == 1:
            shap_arr = shap_arr.reshape(-1, 1)

        if shap_arr.shape[0] != len(X):
            logger.warning(
                "SHAP values shape mismatch (expected %s rows, got %s); skipping reason codes",
                len(X),
                shap_arr.shape[0],
            )
            return _empty_reasons()

        abs_vals = np.abs(shap_arr)
        top_idx = np.argsort(-abs_vals, axis=1)[:, :top_k]
        reasons = []
        cols = list(feature_cols)
        for row_idx in range(top_idx.shape[0]):
            feats = [cols[j] for j in top_idx[row_idx]]
            reasons.append(feats + [None] * max(0, top_k - len(feats)))
        reasons_df = pd.DataFrame(reasons, columns=[f"reason_{i+1}" for i in range(top_k)])
        return reasons_df
    except Exception as exc:  # pragma: no cover
        logger.warning("SHAP failed: %s; skipping reason codes", exc)
        return _empty_reasons()

