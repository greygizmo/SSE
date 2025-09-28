"""Collect feature importance and capture-at-k baselines for the business deck.

Phase 5 deliverables require a simple set of reference tables that describe the
baseline (pre-ML) customer metrics. This script grabs two sources:

* ``feature_importance.csv`` under each division's model directory
* ``capture_at_k.csv`` exported by the validation job

It normalizes the data into ``gosales/outputs/baseline`` so that analytics and
product stakeholders can plug the outputs directly into slides or Confluence.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'gosales' / 'models'
OUTPUTS_DIR = ROOT / 'gosales' / 'outputs'

DIV_MAP = {
    'Solidworks':'solidworks_model',
    'CPE':'cpe_model',
    'Hardware':'printers_model',
    'Services':'services_model',
    'Simulation':'simulation_model',
    'Scanning':'scanning_model',
    'Training':'training_model',
    'CAMWorks':'camworks_model',
    'Post_Processing':'post_processing_model',
    'PDM':'pdm_seats_model',
    'Success Plan':'success_plan_model',
    'Maintenance':'maintenance_model',
}

def load_feature_importance() -> pd.DataFrame:
    rows = []
    for div, dname in DIV_MAP.items():
        f = MODELS_DIR / dname / 'feature_importance.csv'
        if f.exists():
            try:
                df = pd.read_csv(f)
                df['division']=div
                rows.append(df)
            except Exception:
                pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_capture_at_k() -> pd.DataFrame:
    f = OUTPUTS_DIR / 'capture_at_k.csv'
    if f.exists():
        try:
            return pd.read_csv(f)
        except Exception:
            pass
    return pd.DataFrame()


def main():
    cap = load_capture_at_k()
    fi = load_feature_importance()

    out_dir = OUTPUTS_DIR / 'baseline'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cap.empty:
        cap.to_csv(out_dir / 'capture_at_k_summary.csv', index=False)
    if not fi.empty:
        # Top 25 features per division
        tops = (
            fi.sort_values(['division','gain'], ascending=[True, False])
              .groupby('division')
              .head(25)
        )
        tops.to_csv(out_dir / 'feature_importance_top25.csv', index=False)

    print('Wrote summaries to', out_dir)


if __name__ == '__main__':
    main()

