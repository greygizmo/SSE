"""Batch runner for the leakage gauntlet across all supported divisions."""

from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
from typing import Iterable

import click

from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger
from gosales.etl.sku_map import division_set, get_supported_models


logger = get_logger(__name__)


def _target_divisions() -> list[str]:
    # Mirror score_all preferred targets
    try:
        divisions = list(division_set())
    except Exception:
        divisions = []
    try:
        models = list(get_supported_models())
    except Exception:
        models = []
    preferred_divisions = {"Training", "Services", "Simulation", "Scanning", "CAMWorks"}
    preferred_models = set(models) | {"Printers", "SWX_Seats", "PDM_Seats", "SW_Electrical", "SW_Inspection", "Success_Plan"}
    targets = sorted(preferred_divisions | preferred_models)
    return targets


def _run_gauntlet(div: str, cutoff: str, window_months: int, *, run_shift_grid: bool = False) -> dict:
    args = [
        sys.executable, "-m", "gosales.pipeline.run_leakage_gauntlet",
        "--division", div,
        "--cutoff", cutoff,
        "--window-months", str(int(window_months)),
        "--no-static-only",
        "--run-shift14-training",
        "--run-repro-check",
        "--run-topk-ablation",
    ]
    if run_shift_grid:
        args.append("--run-shift-grid")
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Gauntlet subprocess failed for {div}: {e}")
    # Read back the per-division report
    report = OUTPUTS_DIR / "leakage" / div / cutoff / f"leakage_report_{div}_{cutoff}.json"
    try:
        return json.loads(report.read_text(encoding='utf-8')) if report.exists() else {"division": div, "cutoff": cutoff, "overall": "MISSING"}
    except Exception:
        return {"division": div, "cutoff": cutoff, "overall": "ERROR"}


@click.command()
@click.option("--cutoff", required=True, help="Cutoff date YYYY-MM-DD, e.g. 2024-06-30")
@click.option("--window-months", default=6, type=int)
@click.option("--divisions", default=None, help="Optional comma-separated list of divisions to include (default: auto)")
@click.option("--run-shift-grid/--no-run-shift-grid", default=False, help="Also run shift-grid summary (non-gating)")
def main(cutoff: str, window_months: int, divisions: str | None, run_shift_grid: bool) -> None:
    if divisions:
        targets = [d.strip() for d in str(divisions).split(',') if d.strip()]
    else:
        targets = _target_divisions()
    logger.info("Running Leakage Gauntlet for %d targets at %s", len(targets), cutoff)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUTS_DIR / "leakage"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    summary = {"cutoff": cutoff, "window_months": int(window_months), "divisions": []}
    overall = "PASS"
    for d in targets:
        rep = _run_gauntlet(d, cutoff, window_months, run_shift_grid=run_shift_grid)
        results.append(rep)
        summary["divisions"].append({"division": d, "overall": rep.get("overall", "UNKNOWN")})
        if rep.get("overall") == "FAIL":
            overall = "FAIL"
    summary["overall"] = overall
    # Write cross-division summary
    p = out_dir / f"leakage_summary_{cutoff}.json"
    p.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info("Wrote cross-division leakage summary to %s", p)


if __name__ == "__main__":
    main()

