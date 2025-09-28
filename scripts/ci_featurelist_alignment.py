"""Verify that exported feature catalogs stay aligned with the trained models.

Training jobs persist ``feature_list.json`` alongside the model so scoring and
validation pipelines can reproduce the exact feature order. This CI-oriented
script compares those saved lists with the feature catalog or parquet emitted by
feature generation for the corresponding division and cutoff. It writes a
machine-readable JSON report and returns a non-zero exit code if any division has
missing features, preventing silently drifting schemas from landing in the repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from gosales.utils.paths import MODELS_DIR, OUTPUTS_DIR
from gosales.utils.normalize import normalize_division


def _load_model_meta(model_dir: Path) -> Dict:
    meta_path = model_dir / 'metadata.json'
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _load_feature_list(model_dir: Path, meta: Dict) -> Optional[List[str]]:
    # Prefer explicit feature_list.json
    fl = model_dir / 'feature_list.json'
    if fl.exists():
        try:
            return json.loads(fl.read_text(encoding='utf-8'))
        except Exception:
            pass
    # Fallback to metadata.feature_names if present
    try:
        feats = meta.get('feature_names')
        if isinstance(feats, list) and feats:
            return [str(x) for x in feats]
    except Exception:
        pass
    return None


def _features_from_outputs(division: str, cutoff: str) -> Optional[List[str]]:
    # Prefer features parquet
    p_parquet = OUTPUTS_DIR / f"features_{division.lower()}_{cutoff}.parquet"
    if p_parquet.exists():
        try:
            df = pd.read_parquet(p_parquet)
            cols = [c for c in df.columns if c not in ('customer_id', 'bought_in_division')]
            return [str(c) for c in cols]
        except Exception:
            pass
    # Fallback to feature catalog CSV
    p_catalog = OUTPUTS_DIR / f"feature_catalog_{division.lower()}_{cutoff}.csv"
    if p_catalog.exists():
        try:
            cat = pd.read_csv(p_catalog)
            names = [str(n) for n in cat.get('name', []) if str(n) not in ('customer_id', 'bought_in_division')]
            return names
        except Exception:
            pass
    return None


def check_alignment() -> Path:
    results = []
    failures = 0
    for model_dir in MODELS_DIR.glob('*_model'):
        meta = _load_model_meta(model_dir)
        div_raw = meta.get('division') or model_dir.name.replace('_model', '')
        division = normalize_division(div_raw)
        cutoff = str(meta.get('cutoff_date') or '').strip()
        feat_list = _load_feature_list(model_dir, meta)
        if not feat_list or not cutoff:
            results.append({
                'division': division,
                'cutoff': cutoff or None,
                'status': 'SKIP',
                'reason': 'missing feature_list or cutoff in metadata',
            })
            continue
        avail = _features_from_outputs(division, cutoff)
        if avail is None:
            results.append({
                'division': division,
                'cutoff': cutoff,
                'status': 'SKIP',
                'reason': 'missing features parquet/catalog in outputs',
            })
            continue
        need = set(feat_list)
        have = set(avail)
        missing = sorted(list(need - have))
        extras = sorted(list(have - need))
        status = 'PASS' if not missing else 'FAIL'
        if status == 'FAIL':
            failures += 1
        results.append({
            'division': division,
            'cutoff': cutoff,
            'status': status,
            'missing_count': len(missing),
            'extras_count': len(extras),
            'missing': missing[:50],  # truncate for brevity
            'extras': extras[:50],
        })
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / 'ci_featurelist_alignment.json'
    out.write_text(json.dumps({'results': results}, indent=2), encoding='utf-8')
    if failures:
        print(f'Feature-list alignment FAIL for {failures} target(s). See {out}')
        raise SystemExit(1)
    print('Feature-list alignment PASS. See', out)
    return out


if __name__ == '__main__':
    check_alignment()

