from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def new_run_id() -> str:
    """Return a short, URL-safe run identifier.

    Uses the first 8 hex characters of a UUID4 for brevity while keeping
    collision risk negligible for our usage.
    """
    return uuid.uuid4().hex[:8]


def get_git_sha(short: bool = True) -> str:
    """Return the current Git SHA if available, else 'unknown'."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(args, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def emit_manifest(output_dir: Path, run_id: str, manifest: Dict[str, Any]) -> Path:
    """Write the run manifest JSON next to outputs and return the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"run_context_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def default_manifest(pipeline_version: str | None = None) -> Dict[str, Any]:
    """Create a baseline manifest dict with standard fields and containers."""
    rid = new_run_id()
    manifest: Dict[str, Any] = {
        "run_id": rid,
        "git_sha": get_git_sha(short=True),
        "utc_timestamp": utc_now_iso(),
        "pipeline_version": pipeline_version or os.getenv("GOSALES_PIPELINE_VERSION", "0.1.0"),
        # High-level fields; scoring may refine/expand these
        "cutoff": None,
        "window_months": None,
        "divisions_scored": [],
        "alerts": [],
    }
    return manifest


