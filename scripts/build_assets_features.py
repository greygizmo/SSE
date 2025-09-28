"""Rebuild asset-based feature extracts for a given cutoff date.

ETL notebooks depend on CSV extracts of the customer-level features generated
from ``fact_assets``. This command-line utility pulls the raw table, feeds it
through ``gosales.etl.assets.features_at_cutoff``, sanitizes column names, and
writes both the rollup and per-asset outputs back to ``gosales/outputs``.

The script is intentionally tiny so analysts can regenerate features on demand
without needing to invoke the entire training pipeline.
"""

import argparse
import pandas as pd
from gosales.utils.db import get_curated_connection
from gosales.etl.assets import features_at_cutoff
from gosales.utils.paths import OUTPUTS_DIR


def main():
    ap = argparse.ArgumentParser(description="Build asset-based features at a cutoff date")
    ap.add_argument("--cutoff", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    eng = get_curated_connection()
    fa = pd.read_sql('SELECT * FROM fact_assets', eng)
    roll, per = features_at_cutoff(fa, args.cutoff)

    # Sanitize column names for rollup pivot
    def safe(col: str) -> str:
        return 'assets_rollup_' + str(col).strip().lower().replace(' ', '_').replace('/', '_')

    if not roll.empty:
        roll.columns = [c if c == 'customer_id' else safe(c) for c in roll.columns]

    out_dir = OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    roll.to_csv(out_dir / f'assets_rollup_features_{args.cutoff}.csv', index=False)
    per.to_csv(out_dir / f'assets_features_{args.cutoff}.csv', index=False)
    print('Wrote:', out_dir / f'assets_rollup_features_{args.cutoff}.csv')
    print('Wrote:', out_dir / f'assets_features_{args.cutoff}.csv')


if __name__ == '__main__':
    main()

