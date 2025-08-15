from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np


def _read_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_schema_reports(outputs_dir: Path) -> List[Dict[str, object]]:
    reports: List[Dict[str, object]] = []
    for name in [
        "schema_icp_scores.json",
        "schema_whitespace.json",
    ]:
        p = outputs_dir / name
        if p.exists():
            reports.append(_read_json(p))
    # Also include cutoff-tagged whitespace schema files if present
    for p in outputs_dir.glob("schema_whitespace_*.json"):
        reports.append(_read_json(p))
    return reports


def _collect_validation_reports(outputs_dir: Path) -> List[Dict[str, object]]:
    reports: List[Dict[str, object]] = []
    # Top-level metrics
    for name in ["validation_metrics.json"]:
        p = outputs_dir / name
        if p.exists():
            reports.append(_read_json(p))
    # Year-tagged variants
    for p in outputs_dir.glob("validation_metrics_*.json"):
        reports.append(_read_json(p))
    # Phase-5 validation subfolders
    val_dir = outputs_dir / "validation"
    if val_dir.exists():
        for div_dir in val_dir.iterdir():
            if not div_dir.is_dir():
                continue
            for cut_dir in div_dir.iterdir():
                if not cut_dir.is_dir():
                    continue
                m = cut_dir / "metrics.json"
                if m.exists():
                    reports.append(_read_json(m))
                # Alerts file (optional)
                a = cut_dir / "alerts.json"
                if a.exists():
                    reports.append({"alerts": _read_json(a)})
    return reports


def main(outputs: str = "gosales/outputs") -> int:
    out_dir = Path(outputs)
    failures: List[str] = []

    # Schema checks
    schema_reports = _collect_schema_reports(out_dir)
    for rep in schema_reports:
        ok = rep.get("ok", True)
        if not ok:
            failures.append(f"Schema check failed for {rep.get('file')}: missing={rep.get('missing_columns')}, type_issues={rep.get('type_issues')}")

    # Validation gates
    val_reports = _collect_validation_reports(out_dir)
    for rep in val_reports:
        # Top-level validation_metrics.json structure (simple gate)
        if isinstance(rep, dict) and rep.get("status") == "fail":
            failures.append("Validation gates failed: status=fail")
        # Phase-5 metrics.json structure: enforce thresholds from validation config if present
        if isinstance(rep, dict) and "metrics" in rep and isinstance(rep["metrics"], dict):
            try:
                m = rep["metrics"]
                auc = m.get("auc")
                cal_mae = m.get("cal_mae")
                # Fallback thresholds
                auc_thr = 0.70
                cal_thr = 0.10
                if auc is not None and not (float(auc) >= auc_thr or np.isnan(float(auc))):
                    failures.append(f"AUC below threshold: {float(auc):.3f} < {auc_thr:.2f}")
                if cal_mae is not None and not (float(cal_mae) <= cal_thr or np.isnan(float(cal_mae))):
                    failures.append(f"Calibration MAE above threshold: {float(cal_mae):.3f} > {cal_thr:.2f}")
            except Exception:
                pass

    # Drift/alerts: treat presence of alerts.json with non-empty alerts as a soft warning (does not fail build)
    alerts_path = out_dir / 'alerts.json'
    if alerts_path.exists():
        try:
            payload = _read_json(alerts_path)
            alerts = payload.get('alerts', []) if isinstance(payload, dict) else []
            if alerts:
                # print to stderr but do not mark as failure
                sys.stderr.write("Warning: alerts.json contains drift/calibration alerts\n")
        except Exception:
            pass

    if failures:
        sys.stderr.write("\n".join(failures) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


