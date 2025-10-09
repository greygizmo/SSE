"""Legacy holdout gate helpers (DB mutation removed).

This module now only exposes the lightweight `validate_holdout` score gate used by
older orchestration flows. The former `validate_against_holdout` routine that
wrote future data into the curated warehouse has been fully retired in favour of
`gosales.validation.forward`, which builds holdout artifacts safely without any
database writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR


logger = get_logger(__name__)


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    prec, rec, _ = skm.precision_recall_curve(y_true, y_score)
    return float(skm.auc(rec, prec))


def _lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k_percent: int) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(n * (k_percent / 100.0)))
    idx = np.argsort(-y_score, kind="stable")[:k]
    top_rate = float(np.mean(y_true[idx]))
    base_rate = float(np.mean(y_true))
    if np.isclose(base_rate, 0.0):
        return float("nan")
    return top_rate / base_rate


def validate_holdout(
    icp_scores_csv: str | Path,
    *,
    year_tag: str | None = None,
    gates: Dict[str, float] | None = None,
) -> Path:
    """Evaluate score outputs against simple gating thresholds.

    This function predates the forward-validation pipeline and is kept for
    backwards compatibility with older tooling that expects a JSON summary file.
    It is intentionally lightweight and does not attempt to materialise holdout
    feature matrices.
    """

    df = pd.read_csv(icp_scores_csv)
    df = df.dropna(subset=["icp_score"])

    gates = gates or {"auc": 0.70, "lift_at_10": 2.0, "cal_mae": 0.10}
    results: List[Dict[str, float]] = []
    status_ok = True

    for div, g in df.groupby("division_name"):
        y = (
            pd.to_numeric(g.get("bought_in_division", 0), errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        scores = pd.to_numeric(g["icp_score"], errors="coerce").fillna(0.0).to_numpy()
        if len(y) == 0:
            continue

        try:
            auc = float(skm.roc_auc_score(y, scores))
        except Exception:
            auc = float("nan")

        try:
            ps = pd.Series(scores)
            uniq = ps.nunique(dropna=False)
            if uniq >= 10:
                bins = pd.qcut(ps, q=10, labels=False, duplicates="drop")
            else:
                bins = pd.cut(
                    ps,
                    bins=max(1, min(10, uniq)),
                    labels=False,
                    include_lowest=True,
                    duplicates="drop",
                )
            cal = pd.DataFrame({"y": y, "p": scores, "bin": bins})
            grp = (
                cal.groupby("bin", observed=False)
                .agg(mean_p=("p", "mean"), frac_pos=("y", "mean"), count=("y", "size"))
                .dropna()
            )
            cal_mae = float(
                (grp["mean_p"].sub(grp["frac_pos"]).abs() * grp["count"]).sum()
                / max(1, grp["count"].sum())
            )
        except Exception:
            cal_mae = float("nan")

        brier = float(np.mean((scores - y) ** 2))
        lift10 = _lift_at_k(y, scores, 10)

        res = {
            "division_name": div,
            "auc": auc,
            "pr_auc": _pr_auc(y, scores),
            "brier": brier,
            "cal_mae": cal_mae,
            "lift_at_10": lift10,
        }
        results.append(res)

        if not np.isnan(auc) and auc < gates["auc"]:
            status_ok = False
        if not np.isnan(lift10) and lift10 < gates["lift_at_10"]:
            status_ok = False
        if not np.isnan(cal_mae) and cal_mae > gates["cal_mae"]:
            status_ok = False

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / (
        f"validation_metrics_{year_tag}.json" if year_tag else "validation_metrics.json"
    )
    out.write_text(
        json.dumps(
            {
                "divisions": results,
                "gates": gates,
                "status": "ok" if status_ok else "fail",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote validation metrics to %s", out)
    return out


def validate_against_holdout(*_: object, **__: object) -> dict[str, str]:
    """Deprecated compatibility shim.

    The unsafe implementation that mutated curated tables has been removed. Users
    should invoke ``gosales.validation.forward`` for holdout evaluation.
    """

    logger.warning(
        "validate_against_holdout() is deprecated and no longer performs any work. "
        "Use gosales.validation.forward for holdout validation."
    )
    return {
        "status": "deprecated",
        "message": "Use gosales.validation.forward; no database writes performed.",
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(OUTPUTS_DIR / "icp_scores.csv"))
    ap.add_argument("--year", default=None)
    args = ap.parse_args()
    validate_holdout(args.scores, year_tag=args.year)
