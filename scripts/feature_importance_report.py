from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import joblib
import numpy as np
import pandas as pd

import sys
try:
    import gosales  # noqa: F401
except Exception:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gosales.utils.paths import MODELS_DIR, OUTPUTS_DIR
from gosales.utils.logger import get_logger


logger = get_logger(__name__)


def _unwrap(model):
    base = getattr(model, 'base_estimator', None)
    if base is None and hasattr(model, 'estimator'):
        base = model.estimator
    if base is None:
        base = model
    try:
        from sklearn.pipeline import Pipeline as _SkPipeline  # type: ignore
        if isinstance(base, _SkPipeline) and 'model' in getattr(base, 'named_steps', {}):
            base = base.named_steps['model']
    except Exception:
        pass
    return base


def _feature_names(model_dir: Path) -> Optional[list[str]]:
    try:
        meta = json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8'))
        feats = meta.get('feature_names')
        if feats:
            return [str(x) for x in feats]
    except Exception:
        return None
    return None


def _importance_series(model, names: Optional[list[str]]) -> Optional[pd.Series]:
    base = _unwrap(model)
    try:
        if hasattr(base, 'feature_importances_'):
            arr = np.asarray(getattr(base, 'feature_importances_'))
            n = names if names is not None else [f'f{i}' for i in range(len(arr))]
            return pd.Series(arr, index=n).sort_values(ascending=False)
        if hasattr(base, 'coef_'):
            arr = np.abs(np.ravel(getattr(base, 'coef_')))
            n = names if names is not None else [f'f{i}' for i in range(len(arr))]
            return pd.Series(arr, index=n).sort_values(ascending=False)
    except Exception:
        return None
    return None


def _flag_reason(name: str) -> Optional[str]:
    s = str(name).lower()
    if s.startswith('assets_expiring_'):
        return 'near-cutoff_expiring_window'
    if 'recency' in s or 'days_since_last' in s:
        return 'recency_near_cutoff'
    if s.startswith('assets_subs_share_') or s.startswith('assets_on_subs_share_') or s.startswith('assets_off_subs_share_'):
        return 'subscription_composition'
    return None


@click.command()
@click.option('--divisions', default=None, help='Comma list to subset (e.g., Printers,Solidworks)')
@click.option('--top', default=50, type=int, help='Top N features to export with flags')
def main(divisions: Optional[str], top: int) -> None:
    targets = [p for p in MODELS_DIR.glob('*_model') if p.is_dir()]
    if divisions:
        allow = {d.strip().lower() for d in divisions.split(',') if d.strip()}
        targets = [p for p in targets if p.name.replace('_model', '').lower() in allow]
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for mdir in targets:
        div = mdir.name.replace('_model', '')
        try:
            model = joblib.load(mdir / 'model.pkl')
        except Exception as e:
            logger.warning('Skip %s: cannot load model (%s)', div, e)
            continue
        names = _feature_names(mdir)
        imp = _importance_series(model, names)
        if imp is None:
            logger.warning('No importances for %s', div)
            continue
        df = pd.DataFrame({'feature': imp.index, 'importance': imp.values})
        df['flag'] = df['feature'].apply(lambda x: _flag_reason(str(x)))
        out = OUTPUTS_DIR / f'feature_importance_{div}.csv'
        df.head(int(top)).to_csv(out, index=False)
        logger.info('Wrote %s', out)


if __name__ == '__main__':
    main()

