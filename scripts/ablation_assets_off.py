"""Run an assets-disabled model training ablation and summarize metric deltas.

This CLI temporarily disables the feature flags that pull Moneyball asset features
for a single division, retrains the standard gosales model configuration, and then
compares the resulting evaluation metrics against the latest production baseline.
It exists to give analysts a reproducible way to quantify how much the asset
enrichment contributes before rolling out changes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import click

import sys

# Ensure repository root is importable when running as a script
try:
    import gosales  # type: ignore
except Exception:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gosales.utils.config import load_config
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger


logger = get_logger(__name__)


def _metrics_path(division: str) -> Path:
    return OUTPUTS_DIR / f"metrics_{division.lower()}.json"


def _load_metrics(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


@click.command()
@click.option("--division", required=True, help="Target division (e.g., Solidworks)")
@click.option("--cutoff", default=None, help="Cutoff YYYY-MM-DD (default: config.run.cutoff_date)")
@click.option("--window-months", default=None, type=int, help="Prediction window months (default: config.run.prediction_window_months)")
@click.option("--models", default="logreg,lgbm", help="Models to train (comma-separated)")
@click.option("--config", default=str((Path(__file__).parents[1] / 'gosales' / 'config.yaml').resolve()))
def main(division: str, cutoff: Optional[str], window_months: Optional[int], models: str, config: str) -> None:
    """Run an assets-OFF ablation training and compare metrics to baseline.

    Steps:
    - Backup existing metrics_<division>.json if present (baseline assumed assets ON)
    - Train with GOSALES_FEATURES_USE_ASSETS=false at the specified cutoff/window
    - Load new metrics as 'assets_off', restore baseline metrics file
    - Write ablation_assets_off_<division>_<cutoff>.json with baseline, assets_off, deltas
    """
    cfg = load_config(config)
    cutoff = cutoff or str(cfg.run.cutoff_date)
    window = int(window_months or cfg.run.prediction_window_months)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    met_path = _metrics_path(division)
    baseline = _load_metrics(met_path)
    backup_path = None
    if met_path.exists():
        backup_path = met_path.with_suffix(met_path.suffix + ".baseline")
        shutil.copy2(met_path, backup_path)
        logger.info("Backed up baseline metrics to %s", backup_path)

    # Train with assets disabled
    env = os.environ.copy()
    env["GOSALES_FEATURES_USE_ASSETS"] = "0"
    cmd = [
        os.sys.executable,
        "-m",
        "gosales.models.train",
        "--division",
        division,
        "--cutoffs",
        cutoff,
        "--window-months",
        str(window),
        "--models",
        models,
    ]
    try:
        logger.info("Training assets-OFF model for %s @ %s (window=%d)", division, cutoff, window)
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.error("Assets-OFF training failed: %s", e)
        raise SystemExit(1)

    assets_off = _load_metrics(met_path)

    # Restore baseline metrics if we backed it up
    if backup_path and backup_path.exists():
        try:
            shutil.copy2(backup_path, met_path)
        except Exception:
            pass

    # Compare and write report
    def _fin(m: dict) -> dict:
        return m.get("final", {}) if isinstance(m, dict) else {}

    fin_base = _fin(baseline)
    fin_off = _fin(assets_off)
    out = {
        "division": division,
        "cutoff": cutoff,
        "window_months": window,
        "baseline": fin_base,
        "assets_off": fin_off,
        "delta": {
            "auc": (float(fin_off.get("auc", 0.0)) - float(fin_base.get("auc", 0.0))) if fin_off and fin_base else None,
            "lift@10": (float(fin_off.get("lift@10", 0.0)) - float(fin_base.get("lift@10", 0.0))) if fin_off and fin_base else None,
            "brier": (float(fin_off.get("brier", 0.0)) - float(fin_base.get("brier", 0.0))) if fin_off and fin_base else None,
        },
    }
    out_path = OUTPUTS_DIR / f"ablation_assets_off_{division.lower()}_{cutoff.replace('-','')}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
