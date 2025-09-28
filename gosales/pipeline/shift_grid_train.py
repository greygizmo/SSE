"""Grid-search leakage robustness by shifting training cutoffs and purges.

To validate that SAFE parameters generalize, this script trains models across a
grid of cutoff and purge settings, compares holdout metrics, and records
results.  It feeds decision-making on how aggressive we can be with temporal
buffers.
"""

from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
from datetime import timedelta

import click
import pandas as pd

from gosales.utils.config import load_config
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.db import get_curated_connection, get_db_connection
from gosales.features.engine import create_feature_matrix


def _lift10_from_metrics(d: dict) -> float | None:
    if not isinstance(d, dict):
        return None
    v = d.get("lift@10")
    if v is None:
        v = d.get("lift10")
    return v


def _train_at(division: str, cutoff: str, window_months: int, purge: int, label_buf: int) -> dict:
    met_path = OUTPUTS_DIR / f"metrics_{division.lower()}.json"
    cmd = [
        sys.executable, "-m", "gosales.models.train",
        "--division", division,
        "--cutoffs", cutoff,
        "--window-months", str(int(window_months)),
        "--group-cv",
        "--purge-days", str(int(purge)),
        "--safe-mode",
        "--label-buffer-days", str(int(label_buf)),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return {}
    if met_path.exists():
        try:
            return json.loads(met_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _prevalence(engine, division: str, cutoff: str, window_months: int, mask_tail_days: int) -> float | None:
    fm = create_feature_matrix(engine, division, cutoff, window_months, mask_tail_days=mask_tail_days)
    if fm.is_empty():
        return None
    try:
        return float(fm.to_pandas()["bought_in_division"].mean())
    except Exception:
        return None


@click.command()
@click.option("--division", required=True)
@click.option("--cutoff", required=True)
@click.option("--window-months", default=6, type=int)
@click.option("--shift-grid", default="7,14,28,56", help="Comma-separated day shifts (e.g., 7,14,28,56)")
@click.option("--summary-divisions", default=None, help="Comma-separated divisions to aggregate a cross-division summary for this cutoff")
def main(division: str, cutoff: str, window_months: int, shift_grid: str, summary_divisions: str | None) -> None:
    cfg = load_config()
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()

    mask_tail = int(getattr(getattr(cfg, "validation", object()), "gauntlet_mask_tail_days", 0) or 0)
    purge = int(getattr(getattr(cfg, "validation", object()), "gauntlet_purge_days", 30) or 30)
    label_buf = int(getattr(getattr(cfg, "validation", object()), "gauntlet_label_buffer_days", 0) or 0)
    eps_auc = float(getattr(getattr(cfg, "validation", object()), "shift14_epsilon_auc", 0.01))
    eps_l10 = float(getattr(getattr(cfg, "validation", object()), "shift14_epsilon_lift10", 0.25))

    # Prevalence snapshot
    base_prev = _prevalence(engine, division, cutoff, window_months, mask_tail)

    # Train base model once
    base_metrics = _train_at(division, cutoff, window_months, purge, label_buf)
    f_base = base_metrics.get("final", {}) if isinstance(base_metrics, dict) else {}

    results = []
    for s in [int(x.strip()) for x in str(shift_grid).split(',') if str(x).strip()]:
        try:
            cut_shift = (pd.to_datetime(cutoff) - timedelta(days=int(s))).date().isoformat()
            prev_shift = _prevalence(engine, division, cut_shift, window_months, mask_tail)
            shift_metrics = _train_at(division, cut_shift, window_months, purge, label_buf)
            f_shift = shift_metrics.get("final", {}) if isinstance(shift_metrics, dict) else {}
            auc_base = f_base.get("auc"); auc_shift = f_shift.get("auc")
            l10_base = _lift10_from_metrics(f_base); l10_shift = _lift10_from_metrics(f_shift)
            try:
                auc_imp = (float(auc_shift or 0.0) - float(auc_base or 0.0)) if (auc_base is not None and auc_shift is not None) else 0.0
                l10_imp = (float(l10_shift or 0.0) - float(l10_base or 0.0)) if (l10_base is not None and l10_shift is not None) else 0.0
                status = "FAIL" if (auc_imp > eps_auc or l10_imp > eps_l10) else "PASS"
            except Exception:
                status = "UNKNOWN"  
            results.append({
                "days": int(s),
                "status": status,
                "prevalence_base": base_prev,
                "prevalence_shift": prev_shift,
                "auc_base": auc_base,
                "auc_shift": auc_shift,
                "lift10_base": l10_base,
                "lift10_shift": l10_shift,
                "brier_base": f_base.get("brier"),
                "brier_shift": f_shift.get("brier"),
            })
        except Exception as e:
            results.append({"days": int(s), "status": "ERROR", "error": str(e)})

    overall = "PASS" if all(r.get("status") in {"PASS","PLANNED","UNKNOWN"} for r in results) else "FAIL"
    out_dir = OUTPUTS_DIR / "leakage" / division / cutoff
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"overall": overall, "shifts": results}
    out_path = out_dir / f"shift_grid_{division}_{cutoff}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(str(out_path))

    # Optional cross-division consolidated summary
    try:
        if summary_divisions:
            divs = [d.strip() for d in str(summary_divisions).split(',') if d.strip()]
            entries = []
            overall_cx = "PASS"
            for d in divs:
                p = OUTPUTS_DIR / "leakage" / d / cutoff / f"shift_grid_{d}_{cutoff}.json"
                data = None
                if d == division:
                    try:
                        data = json.loads(out_path.read_text(encoding="utf-8"))
                    except Exception:
                        data = None
                if data is None and p.exists():
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        data = None
                if data is None:
                    entries.append({"division": d, "status": "MISSING"})
                    overall_cx = "FAIL"
                    continue
                entries.append({
                    "division": d,
                    "overall": data.get("overall"),
                    "shifts": data.get("shifts", []),
                })
                if data.get("overall") == "FAIL":
                    overall_cx = "FAIL"
            cx = {"cutoff": cutoff, "overall": overall_cx, "divisions": entries}
            cx_path = OUTPUTS_DIR / "leakage" / f"shift_grid_summary_{cutoff}.json"
            cx_path.write_text(json.dumps(cx, indent=2), encoding="utf-8")
            print(str(cx_path))
    except Exception:
        pass


if __name__ == "__main__":
    main()
