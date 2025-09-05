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

