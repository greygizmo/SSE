from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import click

import sys

# Ensure repository root on path for direct execution
try:
    import gosales  # type: ignore
except Exception:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gosales.utils.paths import MODELS_DIR
from gosales.utils.logger import get_logger


logger = get_logger(__name__)


def _find_models(models_dir: Path) -> list[Path]:
    return sorted([p for p in models_dir.glob("*_model") if p.is_dir()])


def _meta(p: Path) -> dict:
    try:
        return json.loads((p / "metadata.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


@click.command()
@click.option("--divisions", default=None, help="Comma-separated list to restrict divisions (e.g., Solidworks,Printers)")
def main(divisions: Optional[str]) -> None:
    """Build feature matrices for each trained model's cutoff to align feature lists.

    For every `<division>_model/metadata.json` found, calls:
      python -m gosales.features.build --division <division> --cutoff <cutoff>
    """
    allow: Optional[set[str]] = None
    if divisions:
        allow = {d.strip().lower() for d in divisions.split(",") if d.strip()}

    roots = _find_models(MODELS_DIR)
    if allow:
        roots = [p for p in roots if p.name.replace("_model", "").lower() in allow]
    if not roots:
        logger.warning("No model directories found under %s", MODELS_DIR)
        return
    for p in roots:
        meta = _meta(p)
        division = meta.get("division") or p.name.replace("_model", "")
        cutoff = meta.get("cutoff_date")
        if not cutoff:
            logger.warning("Skipping %s: missing cutoff in metadata", p.name)
            continue
        # Build features via module
        # Use sys.executable directly to avoid shell quoting issues
        import sys as _sys
        py = _sys.executable
        cli = [py, "-m", "gosales.features.build", "--division", str(division), "--cutoff", str(cutoff)]
        try:
            logger.info("Building features for %s @ %s", division, cutoff)
            subprocess.run(cli, check=True)
        except Exception as e:
            logger.warning("Feature build failed for %s @ %s: %s", division, cutoff, e)


if __name__ == "__main__":
    main()
