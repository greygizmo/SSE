from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator

from gosales.utils.paths import OUTPUTS_DIR


def _utc_now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


@contextmanager
def run_context(phase: str) -> Iterator[Dict[str, str]]:
    run_id = _utc_now_id()
    run_dir = OUTPUTS_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_path = run_dir / "logs.jsonl"

    def log(event: Dict[str, object]) -> None:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), "phase": phase, "run_id": run_id}
        payload.update(event)
        with open(logs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    t0 = time.time()
    try:
        log({"level": "INFO", "event": "start"})
        yield {"run_id": run_id, "run_dir": str(run_dir), "log": log}
        dt = int((time.time() - t0) * 1000)
        log({"level": "INFO", "event": "finish", "duration_ms": dt})
    except Exception as e:
        dt = int((time.time() - t0) * 1000)
        log({"level": "ERROR", "event": "exception", "err": str(e), "duration_ms": dt})
        raise


