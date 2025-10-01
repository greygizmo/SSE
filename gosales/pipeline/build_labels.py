"""CLI wrapper for building supervised learning labels across cutoffs.

This script feeds configuration into :mod:`gosales.labels.targets`, writes the
resulting parquet/CSV artifacts, and emits prevalence summaries so modelers know
what class balance to expect before training.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from gosales.utils.config import load_config
from gosales.utils.db import get_db_connection, get_curated_connection, validate_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger
from gosales.labels.targets import LabelParams, build_labels_for_division, prevalence_report


logger = get_logger(__name__)


@click.command()
@click.option("--division", required=True, help="Target division name")
@click.option("--cutoff", required=True, help="Cutoff date YYYY-MM-DD (or comma-separated list)")
@click.option("--window-months", default=6, type=int)
@click.option("--mode", default="expansion", type=click.Choice(["expansion", "all"]))
@click.option("--gp-min-threshold", default=0.0, type=float)
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
def main(division: str, cutoff: str, window_months: int, mode: str, gp_min_threshold: float, config: str) -> None:
    cfg = load_config(config)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    # Prefer curated connection (e.g., SQLite) so labels align with built fact tables
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()
    if not validate_connection(engine):
        raise RuntimeError("Database connection for label building is unhealthy")

    cutoffs = [c.strip() for c in cutoff.split(",") if c.strip()]
    for cut in cutoffs:
        # Apply per-division window override if configured
        w_override = int(getattr(cfg.labels, 'per_division_window_months', {}).get(division.lower(), window_months))
        params = LabelParams(
            division=division,
            cutoff=cut,
            window_months=w_override,
            mode=mode,  # type: ignore[arg-type]
            gp_min_threshold=gp_min_threshold,
            min_positive_target=getattr(cfg.labels, 'sparse_min_positive_target', None),
            max_window_months=int(getattr(cfg.labels, 'sparse_max_window_months', 12)),
        )

        logger.info(f"Building labels: division={division}, cutoff={cut}, window={window_months}, mode={mode}, thresh={gp_min_threshold}")
        labels = build_labels_for_division(engine, params)
        if labels.is_empty():
            logger.warning("Empty labels frame; skipping write for this cutoff.")
            continue

        # Guardrails
        labels_pd = labels.to_pandas()
        uniq = labels_pd[['customer_id', 'division']].drop_duplicates()
        if len(uniq) != len(labels_pd):
            logger.warning("Duplicate (customer, division) rows detected; deduplication may be needed.")
        prev = prevalence_report(labels)
        try:
            prev_rate = float(prev['prevalence'].iloc[0]) if not prev.empty else 0.0
            if prev_rate < 0.005 or prev_rate > 0.5:
                logger.warning(f"Unusual prevalence {prev_rate:.4f}; check windows/thresholds.")
        except Exception:
            pass

        # Artifacts
        base = f"{division.lower()}_{cut}"
        labels_path = OUTPUTS_DIR / f"labels_{base}.parquet"
        labels.write_parquet(labels_path)
        prev.to_csv(OUTPUTS_DIR / f"label_prevalence_{base}.csv", index=False)

        report = {
            "division": division,
            "cutoff": cut,
            "window_months": int(window_months),
            "mode": mode,
            "gp_min_threshold": float(gp_min_threshold),
            "counts": {
                "rows": int(len(labels_pd)),
                "positives": int(labels_pd["label"].sum()),
                "censored": int(labels_pd["censored_flag"].sum()) if "censored_flag" in labels_pd.columns else 0,
            },
        }
        with open(OUTPUTS_DIR / f"cutoff_report_{base}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Labels written to {labels_path}")


if __name__ == "__main__":
    main()


