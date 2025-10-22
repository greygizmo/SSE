from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import pandas as pd

from gosales.utils.config import load_config
from gosales.utils.db import get_curated_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger
from gosales.features.als_audit import main as audit_main  # for CLI reuse if desired
from gosales.features.als_audit import _compute_weights  # reuse weight logic


logger = get_logger(__name__)


@click.command()
@click.option("--cutoff", required=True, type=str)
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
@click.option("--use-goal/--no-use-goal", default=True, help="Scope divisions using product_goal when available (default true)")
@click.option("--limit", default=None, type=int, help="Optional max divisions to audit")
@click.option("--output-dir", default=None, type=str, help="Directory for audit JSONs (defaults under outputs/validation/als_policy)")
def main(cutoff: str, config: str, use_goal: bool, limit: Optional[int], output_dir: Optional[str]) -> None:
    """Batch ALS audit: iterate divisions and write one JSON per division for quick review."""
    cfg = load_config(config)
    engine = get_curated_connection()

    # Discover divisions from fact_transactions
    try:
        col = 'product_goal' if use_goal else 'product_division'
        df = pd.read_sql_query(f"SELECT DISTINCT {col} AS division FROM fact_transactions", engine)
        divs = [str(v).strip() for v in df['division'].dropna().tolist() if str(v).strip()]
    except Exception:
        logger.warning("Failed to load division list; falling back to empty set.")
        divs = []
    if limit is not None:
        divs = divs[: int(limit)]

    outdir = Path(output_dir) if output_dir else (OUTPUTS_DIR / "validation" / "als_policy")
    outdir.mkdir(parents=True, exist_ok=True)

    # Write a global file as well
    try:
        from gosales.features.als_audit import main as _audit_cli
        _audit_cli.main(args=["--cutoff", cutoff, "--config", config, "--output", str(outdir / f"als_policy_global_{cutoff}.json")], standalone_mode=False)
    except Exception as e:
        logger.warning("Global audit failed: %s", e)

    for div in divs:
        try:
            from gosales.features.als_audit import main as _audit_cli
            _audit_cli.main(args=["--cutoff", cutoff, "--config", config, "--division", div, "--output", str(outdir / f"als_policy_{div}_{cutoff}.json")], standalone_mode=False)
            logger.info("Wrote ALS audit for %s", div)
        except Exception as e:
            logger.warning("ALS audit failed for %s: %s", div, e)


if __name__ == "__main__":
    main()

