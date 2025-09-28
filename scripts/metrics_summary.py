"""Aggregate per-division model metrics into a single CSV for reporting.

Model training jobs emit ``metrics_<division>.json`` files containing the final
evaluation statistics for each target. This script sweeps the ``gosales/outputs``
directory, extracts the key headline metrics (AUC, PR-AUC, lift at K values,
Brier score, calibration MAE), and writes an alphabetized ``metrics_summary.csv``.

The consolidated view is handy for status updates and dashboards where analysts
need to compare how the latest training run performed across multiple divisions
without inspecting each JSON by hand.
"""

import json

from pathlib import Path
import pandas as pd
from gosales.utils.paths import OUTPUTS_DIR


def main():
    rows = []
    for p in OUTPUTS_DIR.glob('metrics_*.json'):
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
            fin = obj.get('final', {}) or {}
            rows.append({
                'division': obj.get('division'),
                'auc': float(fin.get('auc', 0.0)),
                'pr_auc': float(fin.get('pr_auc', 0.0)),
                'lift@5': float(fin.get('lift@5', 0.0)),
                'lift@10': float(fin.get('lift@10', 0.0)),
                'lift@20': float(fin.get('lift@20', 0.0)),
                'brier': float(fin.get('brier', 0.0)),
                'cal_mae': float(fin.get('cal_mae', 0.0)),
            })
        except Exception:
            continue
    if not rows:
        print('No metrics_*.json found under', OUTPUTS_DIR)
        return
    df = pd.DataFrame(rows).sort_values('division')
    out = OUTPUTS_DIR / 'metrics_summary.csv'
    df.to_csv(out, index=False)
    print('Wrote', out)


if __name__ == '__main__':
    main()

