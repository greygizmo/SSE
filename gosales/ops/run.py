from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator

from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.config import load_config


def _utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


@contextmanager
def run_context(phase: str) -> Iterator[Dict[str, str]]:
    run_id = _utc_now_id()
    run_dir = OUTPUTS_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_path = run_dir / "logs.jsonl"
    manifest_path = run_dir / "manifest.json"
    registry_path = OUTPUTS_DIR / "runs" / "runs.jsonl"

    def log(event: Dict[str, object]) -> None:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), "phase": phase, "run_id": run_id}
        payload.update(event)
        with open(logs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def write_manifest(files: Dict[str, str]) -> None:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"files": files}, f, indent=2)

    def append_registry(entry: Dict[str, object]) -> None:
        entry_with_ids = {"run_id": run_id, **entry}
        with open(registry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry_with_ids) + "\n")

    t0 = time.time()
    try:
        start_ts = datetime.now(timezone.utc).isoformat()
        log({"level": "INFO", "event": "start"})
        # Append start entry to registry
        try:
            append_registry({"started_at": start_ts, "status": "running", "phase": phase, "artifacts_path": str(run_dir)})
        except Exception:
            pass
        # Write resolved config snapshot
        try:
            cfg = load_config()
            with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
                import yaml
                yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
        except Exception:
            pass

        yield {"run_id": run_id, "run_dir": str(run_dir), "log": log, "write_manifest": write_manifest, "append_registry": append_registry}
        dt = int((time.time() - t0) * 1000)
        log({"level": "INFO", "event": "finish", "duration_ms": dt})
        append_registry({"started_at": start_ts, "finished_at": datetime.now(timezone.utc).isoformat(), "status": "finished", "phase": phase, "artifacts_path": str(run_dir)})
    except Exception as e:
        dt = int((time.time() - t0) * 1000)
        log({"level": "ERROR", "event": "exception", "err": str(e), "duration_ms": dt})
        try:
            append_registry({"started_at": start_ts, "finished_at": datetime.now(timezone.utc).isoformat(), "status": "error", "phase": phase, "artifacts_path": str(run_dir), "error": str(e)})
        except Exception:
            pass
        raise


