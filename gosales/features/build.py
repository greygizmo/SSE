from __future__ import annotations

import json
from pathlib import Path
import click
import pandas as pd
import polars as pl

from gosales.utils.config import load_config
from gosales.utils.db import get_db_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger
from gosales.features.engine import create_feature_matrix
from gosales.features.utils import compute_sha256


logger = get_logger(__name__)


@click.command()
@click.option("--division", required=True)
@click.option("--cutoff", required=True, help="YYYY-MM-DD (or list: 2024-03-31,2024-06-30)")
@click.option("--windows", default="3,6,12,24")
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
@click.option("--with-eb/--no-eb", default=True)
@click.option("--with-affinity/--no-affinity", default=True)
@click.option("--with-als/--no-als", default=False)
def main(division: str, cutoff: str, windows: str, config: str, with_eb: bool, with_affinity: bool, with_als: bool) -> None:
    cfg = load_config(config)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_db_connection()

    # For now, engine already computes a comprehensive set; toggles can be wired later
    cutoffs = [c.strip() for c in cutoff.split(",") if c.strip()]
    for cut in cutoffs:
        fm = create_feature_matrix(engine, division, cut, cfg.run.prediction_window_months)
        if fm.is_empty():
            logger.warning(f"Empty feature matrix for cutoff {cut}")
            continue
        # Artifacts naming
        base = f"{division.lower()}_{cut}"
        # Deterministic sort
        fm = fm.sort(["customer_id"])
        # Stats JSON (coverage)
        fm_pd = fm.to_pandas()
        stats = {
            "columns": {
                col: {
                    "dtype": str(fm_pd[col].dtype),
                    "non_null": int(fm_pd[col].notna().sum()),
                    "coverage": float(round(fm_pd[col].notna().mean(), 6)),
                }
                for col in fm_pd.columns
            }
        }
        feat_path = OUTPUTS_DIR / f"features_{base}.parquet"
        fm.write_parquet(feat_path)
        pd.DataFrame(
            [{"name": col, "dtype": str(fm_pd[col].dtype), "coverage": float(round(fm_pd[col].notna().mean(), 6))} for col in fm_pd.columns]
        ).to_csv(OUTPUTS_DIR / f"feature_catalog_{base}.csv", index=False)
        stats["checksum"] = compute_sha256(feat_path)
        with open(OUTPUTS_DIR / f"feature_stats_{base}.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Wrote features and stats for {division} @ {cut}")


if __name__ == "__main__":
    main()


