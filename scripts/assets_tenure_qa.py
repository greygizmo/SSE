import pandas as pd
import numpy as np
from pathlib import Path
from gosales.utils.db import get_curated_connection
from gosales.utils.paths import OUTPUTS_DIR


def compute_tenure_qa(cutoff: str = "2024-12-31") -> Path:
    cutoff_ts = pd.to_datetime(cutoff)
    eng = get_curated_connection()
    fa = pd.read_sql('SELECT customer_id, item_rollup, purchase_date, expiration_date, qty FROM fact_assets', eng)
    fa['purchase_date'] = pd.to_datetime(fa['purchase_date'], errors='coerce')
    fa['expiration_date'] = pd.to_datetime(fa['expiration_date'], errors='coerce')
    min_valid = pd.Timestamp('1996-01-01')
    invalid = fa['purchase_date'].isna() | (fa['purchase_date'] < min_valid)
    valid = ~invalid
    # tenure (valid only)
    fa_valid = fa.loc[valid].copy()
    fa_valid['tenure_days'] = (cutoff_ts - fa_valid['purchase_date']).dt.days
    # median by rollup and global
    med_by_rollup = fa_valid.groupby('item_rollup')['tenure_days'].median().rename('median_tenure_days').reset_index()
    global_med = float(fa_valid['tenure_days'].median()) if len(fa_valid) else 3650.0
    # Build effective purchase for invalid
    map_med = med_by_rollup.set_index('item_rollup')['median_tenure_days'].to_dict()
    def eff(row):
        if not invalid.loc[row.name]:
            return row['purchase_date']
        med = map_med.get(row['item_rollup'], global_med)
        return cutoff_ts - pd.Timedelta(days=int(med))
    fa['purchase_effective'] = fa.apply(eff, axis=1)
    fa['bad_purchase_flag'] = invalid.astype('int8')
    # Summaries
    summary = (
        fa.assign(
            purchase_year=fa['purchase_date'].dt.year,
            effective_year=fa['purchase_effective'].dt.year,
        )
        .groupby('item_rollup')
        .agg(
            rows=('customer_id','count'),
            bad_pct=('bad_purchase_flag', lambda s: float(s.mean())),
            median_tenure_days=('purchase_effective', lambda s: float((cutoff_ts - s).dt.days.median())),
            min_eff_year=('effective_year','min'),
            max_eff_year=('effective_year','max'),
        )
        .reset_index()
        .sort_values('rows', ascending=False)
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / f"assets_tenure_qa_{cutoff.replace('-','')}.csv"
    summary.to_csv(out, index=False)

    # Histogram of tenure days (effective), 50 bins by default
    try:
        eff_days = (cutoff_ts - fa['purchase_effective']).dt.days
        hist, edges = np.histogram(eff_days.dropna().astype(int), bins=50)
        hist_df = pd.DataFrame({
            'bin_left': edges[:-1].astype(float),
            'bin_right': edges[1:].astype(float),
            'count': hist.astype(int),
        })
        hist_path = OUTPUTS_DIR / f"assets_tenure_hist_{cutoff.replace('-','')}.csv"
        hist_df.to_csv(hist_path, index=False)
    except Exception:
        hist_path = None

    # Per-rollup bad-date reliance over time (year)
    try:
        yr = fa['purchase_date'].dt.year.fillna(-1).astype(int)
        rel = (
            pd.DataFrame({
                'item_rollup': fa['item_rollup'].astype(str),
                'year': yr,
                'bad': fa['bad_purchase_flag'].astype(int),
            })
            .groupby(['item_rollup','year'])
            .agg(rows=('bad','size'), bad_share=('bad','mean'))
            .reset_index()
            .sort_values(['item_rollup','year'])
        )
        rel_path = OUTPUTS_DIR / f"assets_bad_date_by_year_{cutoff.replace('-','')}.csv"
        rel.to_csv(rel_path, index=False)
    except Exception:
        rel_path = None

    return out


if __name__ == '__main__':
    p = compute_tenure_qa()
    print('Wrote:', p)
