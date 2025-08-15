from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR


logger = get_logger(__name__)


def _capture_and_lift(df: pd.DataFrame, k_frac: float, label_col: str = "bought_in_division", score_col: str = "icp_score") -> Tuple[float, float]:
    if df.empty:
        return 0.0, 0.0
    d = df[[label_col, score_col]].copy()
    d[label_col] = pd.to_numeric(d[label_col], errors='coerce').fillna(0).astype(int)
    d[score_col] = pd.to_numeric(d[score_col], errors='coerce').fillna(0.0)
    n = len(d)
    k = max(1, int(n * k_frac))
    top = d.nlargest(k, score_col)
    positives = int(d[label_col].sum())
    capture = float(top[label_col].sum()) / max(1, positives)
    base_rate = (d[label_col].mean()) if positives > 0 else 1e-9
    top_rate = float(top[label_col].mean())
    lift = top_rate / max(1e-9, base_rate)
    return capture, lift


def gains_and_capture(icp_scores_csv: Path | str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load only needed columns for performance
    usecols = ["division_name", "icp_score", "bought_in_division"]
    try:
        df = pd.read_csv(icp_scores_csv, usecols=lambda c: c in usecols)
    except Exception:
        df = pd.read_csv(icp_scores_csv)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    req = {"division_name", "icp_score"}
    if not req.issubset(set(df.columns)):
        logger.warning("icp_scores missing required columns for validation; skipping")
        return pd.DataFrame(), pd.DataFrame()

    # Gains by decile per division
    gains_rows = []
    cap_rows = []
    for div, g in df.groupby('division_name'):
        g = g.copy()
        # Use qcut with labels only if it creates exactly 10 bins; otherwise assign decile by rank
        scores = pd.to_numeric(g['icp_score'], errors='coerce').fillna(0.0)
        try:
            g['decile'] = pd.qcut(scores, q=10, labels=list(range(10, 0, -1)), duplicates='drop')
            # If labels mismatch due to dropped duplicates, fall back to rank-based deciles
            if g['decile'].nunique(dropna=True) < 10:
                raise ValueError("insufficient unique bins")
        except Exception:
            ranks = scores.rank(method='average', pct=True)
            # Highest scores should get decile 10
            g['decile'] = (np.ceil(ranks * 10)).clip(1, 10).astype(int)
        gains = g.groupby('decile', observed=False)['bought_in_division'].mean().reset_index()
        gains['division_name'] = div
        gains_rows.append(gains)
        # capture@k for k in {5,10,20}%
        for k in (0.05, 0.10, 0.20):
            cap, lift = _capture_and_lift(g, k)
            cap_rows.append({"division_name": div, "k_percent": int(k*100), "capture": cap, "lift": lift})

    gains_df = pd.concat(gains_rows, ignore_index=True) if gains_rows else pd.DataFrame()
    cap_df = pd.DataFrame(cap_rows)
    return gains_df, cap_df


def emit_validation_artifacts(icp_scores_csv: Path | str, cutoff_tag: str | None = None) -> None:
    gains_df, cap_df = gains_and_capture(icp_scores_csv)
    if gains_df.empty and cap_df.empty:
        return
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    gpath = OUTPUTS_DIR / (f"gains_{cutoff_tag}.csv" if cutoff_tag else "gains.csv")
    cpath = OUTPUTS_DIR / (f"capture_at_k_{cutoff_tag}.csv" if cutoff_tag else "capture_at_k.csv")
    if not gains_df.empty:
        gains_df.to_csv(gpath, index=False)
    if not cap_df.empty:
        cap_df.to_csv(cpath, index=False)
    logger.info(f"Wrote validation artifacts: {gpath.name}, {cpath.name}")


