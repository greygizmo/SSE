"""Refresh SHAP artifacts for all trained divisions.

Product and marketing teams rely on lightweight SHAP exports to explain why
customers were prioritized. This script aligns freshly generated feature
matrices to the model's training schema, samples a subset of customers, computes
global mean-absolute SHAP values and per-customer scores, and stores everything
under ``gosales/outputs``. A manifest JSON summarizes which artifacts were
created per division.

Running the command after a new training cut ensures we have interpretable
artifacts that correspond to the latest production models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gosales.utils.paths import MODELS_DIR, OUTPUTS_DIR
from gosales.utils.db import get_curated_connection, get_db_connection, validate_connection
from gosales.features.engine import create_feature_matrix
from gosales.utils.logger import get_logger


logger = get_logger(__name__)


def _align_features(X: pd.DataFrame, model_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Align feature columns to the training feature order stored in feature_list.json.

    Returns the aligned DataFrame and the ordered feature names used for SHAP.
    """
    flist = model_dir / 'feature_list.json'
    if not flist.exists():
        cols = [c for c in X.columns]
        Xc = X.copy()
        for c in cols:
            Xc[c] = pd.to_numeric(Xc[c], errors='coerce')
        Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
        return Xc.fillna(0.0).astype(float), cols
    try:
        names = json.loads(flist.read_text(encoding='utf-8'))
        cols = [c for c in names if c in X.columns]
        missing = [c for c in names if c not in X.columns]
        extra = [c for c in X.columns if c not in names]
        X2 = X.drop(columns=extra, errors='ignore').copy()
        for m in missing:
            X2[m] = 0.0
        X2 = X2.reindex(columns=names)
        for c in X2.columns:
            X2[c] = pd.to_numeric(X2[c], errors='coerce')
        X2.replace([np.inf, -np.inf], np.nan, inplace=True)
        return X2.fillna(0.0).astype(float), list(names)
    except Exception:
        cols = [c for c in X.columns]
        Xc = X.copy()
        for c in cols:
            Xc[c] = pd.to_numeric(Xc[c], errors='coerce')
        Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
        return Xc.fillna(0.0).astype(float), cols


def _unwrap_model(model):
    base = getattr(model, 'base_estimator', None)
    if base is None and hasattr(model, 'estimator'):
        base = model.estimator
    # Unwrap sklearn Pipeline if present
    try:
        from sklearn.pipeline import Pipeline as _SkPipeline
        if isinstance(base, _SkPipeline) and 'model' in getattr(base, 'named_steps', {}):
            base = base.named_steps['model']
    except Exception:
        pass
    return base if base is not None else model


def regenerate_for_division(model_dir: Path, *, cutoff: str, window_months: int, sample_n: int, seed: int = 42) -> dict[str, str]:
    import joblib
    try:
        model = joblib.load(model_dir / 'model.pkl')
    except Exception as e:
        logger.warning(f"Skip {model_dir.name}: failed to load model.pkl: {e}")
        return {}

    # Resolve division from metadata or folder name
    division = model_dir.name.replace('_model', '')
    meta_path = model_dir / 'metadata.json'
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        division = meta.get('division') or division
    except Exception:
        pass

    # Prefer curated engine (facts live there); fallback to primary
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()
    validate_connection(engine)

    # Build feature matrix at cutoff
    fm = create_feature_matrix(engine, division, cutoff, window_months)
    if fm.is_empty():
        logger.warning(f"No feature rows for {division} @ {cutoff}; skipping SHAP")
        return {}
    df = fm.to_pandas()
    X = df.drop(columns=['customer_id', 'bought_in_division'], errors='ignore')
    X_aligned, feature_names = _align_features(X, model_dir)

    # Prepare sample
    n = len(X_aligned)
    if n == 0:
        logger.warning(f"Empty features for {division}; skipping SHAP")
        return {}
    m = min(int(sample_n), n)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=m, replace=False)
    Xs = X_aligned.iloc[idx]
    cust_ids = df.iloc[idx]['customer_id'].values if 'customer_id' in df.columns else np.arange(m)

    # Compute SHAP
    artifacts: dict[str, str] = {}
    try:
        import shap  # type: ignore
        base = _unwrap_model(model)
        if hasattr(base, 'predict_proba'):
            if base.__class__.__name__.lower().startswith('lgbm') or base.__class__.__name__.lower().startswith('lightgbm'):
                explainer = shap.TreeExplainer(base)
                sv = explainer.shap_values(Xs)
                vals = sv[1] if isinstance(sv, list) and len(sv) == 2 else sv
            elif base.__class__.__name__.lower().startswith('logisticregression'):
                explainer = shap.LinearExplainer(base, Xs)
                vals = explainer.shap_values(Xs)
            else:
                logger.warning(f"Unsupported model type for SHAP on {division}; skipping")
                return {}
            mean_abs = np.mean(np.abs(vals), axis=0)
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            key = str(division).lower()
            gpath = OUTPUTS_DIR / f"shap_global_{key}.csv"
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}) \
                .sort_values("mean_abs_shap", ascending=False) \
                .to_csv(gpath, index=False)
            artifacts[gpath.name] = str(gpath)

            spath = OUTPUTS_DIR / f"shap_sample_{key}.csv"
            sdf = pd.DataFrame(vals, columns=feature_names)
            sdf.insert(0, 'customer_id', cust_ids)
            sdf.to_csv(spath, index=False)
            artifacts[spath.name] = str(spath)
        else:
            logger.warning(f"Model for {division} lacks predict_proba; skipping SHAP")
    except Exception as e:  # pragma: no cover
        logger.warning(f"SHAP generation failed for {division}: {e}")
    return artifacts


def main():
    ap = argparse.ArgumentParser(description="Regenerate SHAP artifacts for all trained models.")
    ap.add_argument('--cutoff', default='2024-06-30')
    ap.add_argument('--window-months', type=int, default=6)
    ap.add_argument('--sample', type=int, default=2000)
    ap.add_argument('--only', default=None, help="Comma-separated list of divisions to include (optional)")
    args = ap.parse_args()

    include = None
    if args.only:
        include = {x.strip().lower() for x in str(args.only).split(',') if x.strip()}

    roots = sorted([p for p in MODELS_DIR.glob('*_model') if p.is_dir()])
    all_artifacts: dict[str, dict[str, str]] = {}
    for md in roots:
        div = md.name.replace('_model', '')
        if include and (div.lower() not in include):
            continue
        logger.info(f"Regenerating SHAP for {div} @ {args.cutoff} (n={args.sample})")
        arts = regenerate_for_division(md, cutoff=args.cutoff, window_months=int(args.window_months), sample_n=int(args.sample))
        all_artifacts[div] = arts
    # Write simple manifest
    man = OUTPUTS_DIR / 'shap_manifest.json'
    man.write_text(json.dumps(all_artifacts, indent=2), encoding='utf-8')
    print('Wrote', man)


if __name__ == '__main__':
    main()

