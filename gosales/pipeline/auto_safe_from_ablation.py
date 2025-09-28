"""Synchronize SAFE configuration with the latest ablation evidence.

SAFE decisions are made based on how the ``safe`` variant compares to the full
feature set in ``adjacency_ablation`` runs. This command line helper scans those
artifacts, looks for divisions where SAFE beats the baseline by at least the
configured threshold, and updates ``config.yaml`` so the modeling pipeline knows
to enable SAFE for that division by default. It saves analysts from manually
editing YAML every time a new ablation run finishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from typing import Dict, List

from gosales.utils.paths import ROOT_DIR, OUTPUTS_DIR


def _load_json(p: Path) -> Dict[str, object]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_yaml(p: Path) -> Dict[str, object]:
    import yaml
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _dump_yaml(p: Path, payload: Dict[str, object]) -> None:
    import yaml
    txt = yaml.safe_dump(payload, sort_keys=False)
    p.write_text(txt, encoding="utf-8")


def find_ablation_jsons(div_filter: str | None = None) -> List[Path]:
    root = OUTPUTS_DIR / 'ablation' / 'adjacency'
    outs: List[Path] = []
    if not root.exists():
        return outs
    for div_dir in root.iterdir():
        if not div_dir.is_dir():
            continue
        if div_filter and div_dir.name.lower() != div_filter.lower():
            continue
        for run_dir in div_dir.iterdir():
            if not run_dir.is_dir():
                continue
            for f in run_dir.glob('adjacency_ablation_*.json'):
                outs.append(f)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--division', default=None)
    ap.add_argument('--threshold', type=float, default=0.005)
    args = ap.parse_args()

    cfg_path = ROOT_DIR / 'config.yaml'
    cfg = _load_yaml(cfg_path)
    modeling = cfg.setdefault('modeling', {})
    safe_list = modeling.setdefault('safe_divisions', []) or []
    # Normalize to strings
    safe_set = {str(x) for x in safe_list}

    changed = False
    for js in find_ablation_jsons(args.division):
        payload = _load_json(js)
        div = str(payload.get('division') or '').strip()
        if not div:
            continue
        delta = payload.get('delta_auc_full_minus_safe')
        try:
            if delta is None:
                continue
            delta = float(delta)
        except Exception:
            continue
        # If SAFE >= Full by threshold -> add division to safe_divisions
        if (-delta) >= float(args.threshold):
            if div not in safe_set:
                safe_set.add(div)
                changed = True

    if changed:
        modeling['safe_divisions'] = sorted(list(safe_set))
        _dump_yaml(cfg_path, cfg)
        print(f"Updated config.yaml safe_divisions: {modeling['safe_divisions']}")
    else:
        print("No changes required to safe_divisions")


if __name__ == '__main__':
    main()

