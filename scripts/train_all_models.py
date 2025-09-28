"""Fire-and-forget wrapper that trains every division model.

Analysts often need to retrain the complete catalog of division models for a new
cutoff. Rather than invoking ``gosales.models.train`` manually for each
division, this CLI enumerates the supported targets (plus Solidworks as the
default baseline) and shells out to the training module with consistent
arguments. Failures are logged but do not stop the loop so a single division
does not block the rest.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import click

import sys
try:
    import gosales  # noqa: F401
except Exception:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gosales.etl.sku_map import get_supported_models
from gosales.utils.logger import get_logger


logger = get_logger(__name__)


def _targets() -> list[str]:
    base = list(get_supported_models())
    # Include Solidworks as a core division model
    if 'Solidworks' not in base:
        base.append('Solidworks')
    return base


@click.command()
@click.option('--cutoff', required=True, help='Training cutoff YYYY-MM-DD (e.g., 2024-06-30)')
@click.option('--window-months', default=6, type=int)
@click.option('--models', default='logreg,lgbm', help='Comma-separated model list')
@click.option('--calibration', default='platt,isotonic')
@click.option('--group-cv/--no-group-cv', default=True)
@click.option('--divisions', default=None, help='Optional comma list to subset targets')
def main(cutoff: str, window_months: int, models: str, calibration: str, group_cv: bool, divisions: str | None) -> None:
    targets = _targets()
    if divisions:
        allow = {d.strip().lower() for d in divisions.split(',') if d.strip()}
        targets = [t for t in targets if t.lower() in allow]
    for t in targets:
        logger.info("Training %s @ %s (window=%d)", t, cutoff, window_months)
        cmd = [sys.executable, '-m', 'gosales.models.train', '--division', t, '--cutoffs', cutoff, '--window-months', str(window_months), '--models', models, '--calibration', calibration]
        if group_cv:
            cmd.append('--group-cv')
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.warning("Training failed for %s: %s", t, e)


if __name__ == '__main__':
    main()

