"""CI guardrail that checks fact_assets hygiene ahead of feature generation.

The assets-based features rely on clean purchase dates and rollup mappings. This
script is intended to run in continuous integration. It executes a handful of
data-quality checks against ``fact_assets`` in the curated warehouse:

* proportion of rows with a valid ``item_rollup`` value
* share of purchase dates that are missing or obviously wrong
* reasonableness of the inferred effective tenure (median days since purchase)

Results are written to ``gosales/outputs/ci_assets_sanity_<cutoff>.json`` and
the process exits with a non-zero code if any threshold is violated so CI can
flag the run. Thresholds and the cutoff date can be overridden via CLI options.
"""

from __future__ import annotations

import sys
from pathlib import Path
import click
import pandas as pd
import numpy as np

from gosales.utils.config import load_config
from gosales.utils.db import get_curated_connection
from gosales.utils.paths import OUTPUTS_DIR


def _effective_purchase_dates(fa: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    fa = fa.copy()
    fa['purchase_date'] = pd.to_datetime(fa['purchase_date'], errors='coerce')
    min_valid = pd.Timestamp('1996-01-01')
    invalid = fa['purchase_date'].isna() | (fa['purchase_date'] < min_valid)
    valid = ~invalid
    fa_valid = fa.loc[valid].copy()
    fa_valid['tenure_days'] = (cutoff - fa_valid['purchase_date']).dt.days
    med_by_rollup = fa_valid.groupby('item_rollup')['tenure_days'].median().rename('median_tenure_days')
    global_med = float(fa_valid['tenure_days'].median()) if len(fa_valid) else 3650.0
    med_map = med_by_rollup.to_dict()
    def eff(row):
        if not invalid.loc[row.name]:
            return row['purchase_date']
        med = float(med_map.get(row.get('item_rollup'), global_med))
        return cutoff - pd.Timedelta(days=int(med))
    eff = fa.apply(eff, axis=1)
    eff = eff.where(eff <= cutoff, cutoff)
    return pd.to_datetime(eff)


@click.command()
@click.option('--cutoff', default=None, help='Cutoff date YYYY-MM-DD; defaults to config.run.cutoff_date')
@click.option('--min-rollup-coverage', default=0.80, type=float, help='Minimum fraction of fact_assets with non-null item_rollup')
@click.option('--max-bad-purchase-share', default=0.60, type=float, help='Maximum fraction of assets with invalid purchase dates')
@click.option('--min-tenure-days', default=30, type=int, help='Minimum reasonable median effective tenure days')
@click.option('--max-tenure-days', default=365*30, type=int, help='Maximum reasonable median effective tenure days')
@click.option('--config', default=str((Path(__file__).parents[1] / 'config.yaml').resolve()))
def main(cutoff: str | None, min_rollup_coverage: float, max_bad_purchase_share: float, min_tenure_days: int, max_tenure_days: int, config: str) -> None:
    cfg = load_config(config)
    cutoff = cutoff or str(getattr(cfg.run, 'cutoff_date'))
    cutoff_ts = pd.to_datetime(cutoff)
    eng = get_curated_connection()
    fa = pd.read_sql('SELECT customer_id, item_rollup, purchase_date, expiration_date, qty FROM fact_assets', eng)
    # Coverage
    coverage = float((~fa['item_rollup'].isna()).mean()) if len(fa) else 0.0
    # Bad purchase share
    min_valid = pd.Timestamp('1996-01-01')
    bad = fa['purchase_date'].isna() | (pd.to_datetime(fa['purchase_date'], errors='coerce') < min_valid)
    bad_share = float(bad.mean()) if len(fa) else 0.0
    # Effective tenure sanity
    eff = _effective_purchase_dates(fa, cutoff_ts)
    med_tenure = float(((cutoff_ts - eff).dt.days.median()) if len(eff) else np.nan)

    failures = []
    if coverage < float(min_rollup_coverage):
        failures.append(f"rollup_coverage {coverage:.3f} < min {min_rollup_coverage:.3f}")
    if bad_share > float(max_bad_purchase_share):
        failures.append(f"bad_purchase_share {bad_share:.3f} > max {max_bad_purchase_share:.3f}")
    try:
        if (not np.isnan(med_tenure)) and (med_tenure < float(min_tenure_days) or med_tenure > float(max_tenure_days)):
            failures.append(f"median_effective_tenure_days {med_tenure:.1f} outside [{min_tenure_days},{max_tenure_days}]")
    except Exception:
        pass

    report = {
        'cutoff': cutoff,
        'coverage': coverage,
        'bad_purchase_share': bad_share,
        'median_effective_tenure_days': med_tenure,
        'thresholds': {
            'min_rollup_coverage': float(min_rollup_coverage),
            'max_bad_purchase_share': float(max_bad_purchase_share),
            'min_tenure_days': int(min_tenure_days),
            'max_tenure_days': int(max_tenure_days),
        },
        'status': 'PASS' if not failures else 'FAIL',
        'failures': failures,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / f"ci_assets_sanity_{cutoff.replace('-','')}.json"
    out.write_text(pd.Series(report).to_json(indent=2), encoding='utf-8')
    if failures:
        print('CI assets sanity FAIL:', '; '.join(failures))
        sys.exit(1)
    print('CI assets sanity PASS')


if __name__ == '__main__':
    main()

