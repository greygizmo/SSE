"""Compile lightweight drift indicators from validation artifacts.

Every validation run writes a ``metrics.json`` file and, when available, a
``validation_frame.parquet`` with the scored holdout data. This script walks the
``gosales/outputs/validation`` tree, extracts the calibration error and label
prevalence for each division/cutoff pair, and saves a compact
``drift_snapshots.csv``. The CSV helps quickly spot divisions whose validation
prevalence or calibration suddenly changes between monthly runs.
"""

import json

from pathlib import Path
import pandas as pd
from gosales.utils.paths import OUTPUTS_DIR


def collect_validation_runs() -> list[tuple[str, str, Path]]:
    root = OUTPUTS_DIR / 'validation'
    items: list[tuple[str, str, Path]] = []
    if not root.exists():
        return items
    for div_dir in root.iterdir():
        if not div_dir.is_dir():
            continue
        for cut_dir in div_dir.iterdir():
            if not cut_dir.is_dir():
                continue
            items.append((div_dir.name, cut_dir.name, cut_dir))
    return items


def main() -> Path | None:
    rows = []
    for division, cutoff, path in collect_validation_runs():
        try:
            metrics = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
        except Exception:
            metrics = {}
        cal_mae = None
        try:
            cal_mae = float((metrics.get('final') or {}).get('cal_mae'))
        except Exception:
            cal_mae = None
        prevalence = None
        try:
            vf_path = path / 'validation_frame.parquet'
            if vf_path.exists():
                vf = pd.read_parquet(vf_path, columns=['bought_in_division'])
                prevalence = float(pd.to_numeric(vf['bought_in_division'], errors='coerce').fillna(0).mean())
        except Exception:
            prevalence = None
        rows.append({
            'division': division,
            'cutoff': cutoff,
            'prevalence': prevalence,
            'cal_mae': cal_mae,
        })
    if not rows:
        print('No validation runs found under', OUTPUTS_DIR / 'validation')
        return None
    df = pd.DataFrame(rows).sort_values(['division','cutoff'])
    out = OUTPUTS_DIR / 'drift_snapshots.csv'
    df.to_csv(out, index=False)
    print('Wrote', out)
    return out


if __name__ == '__main__':
    main()

