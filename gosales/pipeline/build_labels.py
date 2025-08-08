from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from gosales.utils.config import load_config
from gosales.utils.db import get_db_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger
from gosales.labels.targets import LabelParams, build_labels_for_division, prevalence_report


logger = get_logger(__name__)


@click.command()
@click.option("--division", required=True, help="Target division name")
@click.option("--cutoff", required=True, help="Cutoff date YYYY-MM-DD")
@click.option("--window-months", default=6, type=int)
@click.option("--mode", default="expansion", type=click.Choice(["expansion", "all"]))
@click.option("--gp-min-threshold", default=0.0, type=float)
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
def main(division: str, cutoff: str, window_months: int, mode: str, gp_min_threshold: float, config: str) -> None:
    cfg = load_config(config)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_db_connection()

    params = LabelParams(
        division=division,
        cutoff=cutoff,
        window_months=window_months,
        mode=mode,  # type: ignore[arg-type]
        gp_min_threshold=gp_min_threshold,
    )

    logger.info(f"Building labels: division={division}, cutoff={cutoff}, window={window_months}, mode={mode}, thresh={gp_min_threshold}")
    labels = build_labels_for_division(engine, params)
    if labels.is_empty():
        logger.warning("Empty labels frame; aborting write.")
        return

    # Artifacts
    labels_pd = labels.to_pandas()
    base = f"{division.lower()}_{cutoff}"
    labels_path = OUTPUTS_DIR / f"labels_{base}.parquet"
    labels.write_parquet(labels_path)

    prev = prevalence_report(labels)
    prev.to_csv(OUTPUTS_DIR / f"label_prevalence_{base}.csv", index=False)

    report = {
        "division": division,
        "cutoff": cutoff,
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


