"""Snapshot the top rows of key asset-related SQL views for debugging.

This diagnostic script queries the Moneyball asset and rollup views directly in
Azure SQL, writes the first thousand rows for each to CSV, and prints column
metadata so analysts can sanity-check upstream schema changes.
"""

import pandas as pd
from pathlib import Path

from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import ROOT_DIR

logger = get_logger(__name__)

VIEWS = {
    "customer_asset_rollups": "[dbo].[customer_asset_rollups]",
    "moneyball_assets": "[dbo].[Moneyball Assets]",
    "items_category_limited": "[dbo].[items_category_limited]",
}


def peek_top(view_sql_name: str, top_n: int = 1000) -> pd.DataFrame:
    eng = get_db_connection()
    query = f"SELECT TOP ({top_n}) * FROM {view_sql_name}"
    logger.info(f"Querying: {query}")
    df = pd.read_sql(query, eng)
    return df


def main():
    out_dir = ROOT_DIR.parent / "gosales" / "outputs" / "view_peeks"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for key, sql_name in VIEWS.items():
        try:
            df = peek_top(sql_name, 1000)
            results[key] = df
            out_path = out_dir / f"{key}_top1000.csv"
            df.to_csv(out_path, index=False)
            print(f"{key}: rows={len(df)}, cols={len(df.columns)} -> {out_path}")
            print("columns:")
            for c in df.columns:
                print(f" - {c} ({df[c].dtype})")
            print()
        except Exception as e:
            print(f"ERROR reading {key}: {e}")

    # Also dump a small head to console for quick glance
    for key, df in results.items():
        print(f"\n=== {key} HEAD(5) ===")
        print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

