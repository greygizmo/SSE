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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.config import load_config
from gosales.validation.holdout_data import load_holdout_buyers


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

    cfg = load_config()
    run_cfg = getattr(cfg, "run", object())
    default_cutoff = getattr(run_cfg, "cutoff_date", None)
    try:
        default_window = int(getattr(run_cfg, "prediction_window_months", 6) or 6)
    except Exception:
        default_window = 6

    gates = gates or {"auc": 0.70, "lift_at_10": 2.0, "cal_mae": 0.10}
    results: List[Dict[str, float]] = []
    status_ok = True

    for div, g in df.groupby("division_name"):
        scores = pd.to_numeric(g["icp_score"], errors="coerce").fillna(0.0).to_numpy()

        if "customer_id" in g.columns:
            customer_series = pd.to_numeric(g["customer_id"], errors="coerce").astype(
                "Int64"
            )
        else:
            customer_series = pd.Series(dtype="Int64")

        y_series = (
            pd.to_numeric(g.get("bought_in_division", 0), errors="coerce")
            .fillna(0)
            .astype(int)
        )

        if "cutoff_date" in g.columns:
            cutoff_val = pd.to_datetime(g["cutoff_date"].iloc[0], errors="coerce")
        else:
            cutoff_val = None
        if (cutoff_val is None or pd.isna(cutoff_val)) and default_cutoff is not None:
            cutoff_val = pd.to_datetime(default_cutoff, errors="coerce")

        if "prediction_window_months" in g.columns:
            try:
                window_months = int(float(g["prediction_window_months"].iloc[0]))
            except Exception:
                window_months = default_window
        else:
            window_months = default_window

        applied_holdout = False
        if (
            cutoff_val is not None
            and not pd.isna(cutoff_val)
            and not customer_series.empty
        ):
            try:
                holdout = load_holdout_buyers(
                    cfg, str(div), cutoff_val, int(window_months)
                )
                if holdout.buyers is not None and not holdout.buyers.empty:
                    holdout_ids = set(int(x) for x in holdout.buyers.dropna().tolist())
                    mask = customer_series.isin(holdout_ids)
                    y_series = mask.astype(int)
                    applied_holdout = True
                    if holdout.source:
                        logger.info(
                            "Holdout gate using %d buyers from %s for division %s",
                            int(mask.sum()),
                            holdout.source,
                            div,
                        )
            except Exception as err:
                logger.warning("Holdout enrichment failed for %s: %s", div, err)

        y = y_series.to_numpy()
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
            "holdout_applied": bool(applied_holdout),
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
    """Backward-compatible holdout validation entry point.

    Older orchestration flows still import :func:`validate_against_holdout`. The
    original implementation assumed ``fact_transactions`` existed in the curated
    warehouse and crashed otherwise. This safe shim discovers the most recent
    ``icp_scores`` export and runs :func:`validate_holdout` against it, falling
    back gracefully when prerequisite artifacts are missing.
    """

    def _coerce_path(value: object) -> Optional[Path]:
        if isinstance(value, Path):
            return value
        if isinstance(value, str) and value.strip():
            return Path(value.strip())
        return None

    # Backwards compatibility: accept positional or keyword score path hints
    provided_scores = _coerce_path(
        __.get("icp_scores_csv") or __.get("scores") or __.get("scores_csv")
    )
    if provided_scores is None and _:
        provided_scores = _coerce_path(_[0])

    strict = bool(__.get("strict", False))
    year_tag = __.get("year_tag")

    candidates: List[Path] = []
    if provided_scores is not None:
        candidates.append(provided_scores)

    outputs_dir = OUTPUTS_DIR
    primary = outputs_dir / "icp_scores.csv"
    if primary not in candidates:
        candidates.append(primary)

    if outputs_dir.exists():
        try:
            fallbacks = sorted(
                (p for p in outputs_dir.glob("icp_scores_*.csv") if p.is_file()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for fb in fallbacks:
                if fb not in candidates:
                    candidates.append(fb)
        except Exception as exc:
            logger.debug("Failed to enumerate fallback icp_scores files: %s", exc)

    scores_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.is_file():
            scores_path = candidate
            break
    if scores_path is None:
        message = (
            "No icp_scores CSV found; holdout validation skipped. "
            f"Searched: {[str(p) for p in candidates or [primary]]}"
        )
        logger.warning(message)
        return {
            "status": "skipped",
            "message": message,
        }

    logger.info(
        "Running validate_against_holdout using scores file %s. "
        "For full validation flows prefer gosales.validation.forward.",
        scores_path,
    )

    try:
        metrics_path = validate_holdout(str(scores_path), year_tag=year_tag)
    except Exception as exc:
        logger.exception("Holdout validation failed for %s: %s", scores_path, exc)
        if strict:
            raise
        return {
            "status": "error",
            "message": f"Holdout validation failed: {exc}",
            "scores_path": str(scores_path),
        }

    message = f"Holdout validation succeeded; metrics written to {metrics_path}."
    logger.info(message)
    return {
        "status": "ok",
        "message": message,
        "scores_path": str(scores_path),
        "metrics_path": str(metrics_path),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(OUTPUTS_DIR / "icp_scores.csv"))
    ap.add_argument("--year", default=None)
    args = ap.parse_args()
    validate_holdout(args.scores, year_tag=args.year)
