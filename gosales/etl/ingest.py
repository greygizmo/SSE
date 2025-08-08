from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from gosales.utils.logger import get_logger
from gosales.utils.config import load_config


logger = get_logger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def robust_read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Could not read CSV {path}: {last_err}")


@dataclass
class ManifestEntry:
    name: str
    sha256: str
    size_bytes: int
    timestamp: str


def copy_in_raw(files: List[Path], config_path: str | Path | None = None) -> Tuple[Path, List[ManifestEntry]]:
    cfg = load_config(config_path)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    dest_dir = Path(cfg.paths.raw) / ts
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[ManifestEntry] = []
    for f in files:
        f = Path(f)
        dest = dest_dir / f.name
        shutil.copy2(f, dest)
        entry = ManifestEntry(
            name=f.name,
            sha256=_sha256_file(dest),
            size_bytes=dest.stat().st_size,
            timestamp=ts,
        )
        manifest.append(entry)

    with open(dest_dir / "MANIFEST.json", "w", encoding="utf-8") as out:
        json.dump([e.__dict__ for e in manifest], out, indent=2)

    logger.info(f"Copied {len(manifest)} files into raw/{ts} with manifest")
    return dest_dir, manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Copy-in raw inputs and write MANIFEST.json")
    parser.add_argument("--config", type=str, default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
    parser.add_argument("files", nargs="+", help="Paths to input files (CSV)")
    args = parser.parse_args()

    files = [Path(p) for p in args.files]
    copy_in_raw(files, args.config)


