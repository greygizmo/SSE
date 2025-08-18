from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


logger = get_logger(__name__)


# Features used by challenger meta-learner. Tests may monkeypatch this list.
CHALLENGER_FEAT_COLS = ["p_icp_pct", "lift_norm", "als_norm", "EV_norm"]


def _percentile_normalize(s: pd.Series) -> pd.Series:
    """Map values in s to [0,1] by rank-percentile with stable handling of ties."""
    if s is None or len(s) == 0:
        return pd.Series([], dtype=float)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    # Use average rank method to be stable across runs
    ranks = s.rank(method="average", pct=True)
    return ranks.astype(float)


def _compute_affinity_lift(df: pd.DataFrame, col: str = "mb_lift_max") -> pd.Series:
    vals = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return _percentile_normalize(vals)


# Store centroid path for reuse across runs
ALS_CENTROID_PATH = OUTPUTS_DIR / "als_owner_centroid.npy"


def _apply_eligibility(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray | None]:
    """Filter out ineligible rows while capturing ALS centroid of owners.

    Returns the filtered dataframe and the centroid vector. If no owner embeddings
    are present, attempts to load a previously computed centroid from disk.
    """
    als_cols = [c for c in df.columns if c.startswith("als_f")]
    centroid: np.ndarray | None = None
    if als_cols and "owned_division_pre_cutoff" in df.columns:
        owners = df[df["owned_division_pre_cutoff"].astype(bool)]
        if not owners.empty:
            centroid = owners[als_cols].astype(float).mean(axis=0).to_numpy(dtype=float)
            try:
                ALS_CENTROID_PATH.parent.mkdir(parents=True, exist_ok=True)
                np.save(ALS_CENTROID_PATH, centroid)
            except Exception:
                pass
        elif ALS_CENTROID_PATH.exists():
            try:
                centroid = np.load(ALS_CENTROID_PATH)
            except Exception:
                centroid = None
    elif ALS_CENTROID_PATH.exists():
        try:
            centroid = np.load(ALS_CENTROID_PATH)
        except Exception:
            centroid = None

    if "owned_division_pre_cutoff" in df.columns:
        df = df[~df["owned_division_pre_cutoff"].astype(bool)].copy()

    return df, centroid


def _compute_als_norm(df: pd.DataFrame, cfg=None, owner_centroid: np.ndarray | None = None) -> pd.Series:
    """Compute ALS similarity normalized to [0,1].

    - If ``owner_centroid`` is provided, use it for similarity.
    - Else, prefer centroid of rows where ``owned_division_pre_cutoff`` is True.
      Fall back to global centroid if no owned rows.
    """
    als_cols = [c for c in df.columns if c.startswith("als_f")]
    if not als_cols:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    mat = df[als_cols].astype(float)
    if mat.empty:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    if owner_centroid is not None:
        centroid_vec = np.asarray(owner_centroid, dtype=float)
    else:
        # Prefer owned-pre-cutoff centroid when available
        if 'owned_division_pre_cutoff' in df.columns:
            try:
                base = mat[df['owned_division_pre_cutoff'].astype(bool)]
                if not base.empty:
                    centroid_vec = base.mean(axis=0).to_numpy(dtype=float)
                else:
                    centroid_vec = mat.mean(axis=0).to_numpy(dtype=float)
            except Exception:
                centroid_vec = mat.mean(axis=0).to_numpy(dtype=float)
        else:
            centroid_vec = mat.mean(axis=0).to_numpy(dtype=float)

    m = mat.to_numpy(dtype=float)
    # Normalize embeddings and centroid to unit length prior to similarity calc
    m_norm = normalize(m, axis=1)
    centroid_norm = normalize(centroid_vec.reshape(1, -1), axis=1)
    sims = cosine_similarity(m_norm, centroid_norm).ravel()
    return _percentile_normalize(pd.Series(sims, index=df.index))


def _compute_expected_value(df: pd.DataFrame, cfg=None) -> pd.Series:
    """Compute EV proxy with capping per config and normalize to [0,1].

    Prefers rfm__all__gp_sum__12m if present; otherwise returns zeros.
    Applies cap at cfg.whitespace.ev_cap_percentile when available.
    """
    ev_cols = [c for c in df.columns if str(c).lower() in {"rfm__all__gp_sum__12m", "gp_sum_last_12m"}]
    if not ev_cols:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    raw = pd.to_numeric(df[ev_cols[0]], errors='coerce').fillna(0.0)
    cap = None
    try:
        if cfg is not None and getattr(getattr(cfg, 'whitespace', object()), 'ev_cap_percentile', None) is not None:
            p = float(cfg.whitespace.ev_cap_percentile)
            if 0.0 < p <= 1.0:
                cap = float(raw.quantile(p))
    except Exception:
        cap = None
    if cap is not None:
        raw = raw.clip(upper=cap)
    return _percentile_normalize(raw)


def _score_p_icp(df: pd.DataFrame, model, feat_cols: Iterable[str] | None = None) -> pd.Series:
    """Score calibrated ICP probabilities using ``model``.

    When ``feat_cols`` is ``None`` the function falls back to using all numeric
    columns present in ``df`` but explicitly drops common label/score columns
    (e.g. ``label``, ``score``). Any remaining extra numeric columns are
    ignored based on the model's ``n_features_in_`` attribute.
    """

    if feat_cols:
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        X = df.reindex(columns=feat_cols)
    else:
        num = df.select_dtypes(include=[np.number]).copy()
        known = {
            "label",
            "labels",
            "score",
            "scores",
            "icp_score",
            "p_icp",
            "p_icp_pct",
            "p_hat",
            "score_challenger",
        }
        drop_cols = [c for c in num.columns if c.lower() in known]
        if drop_cols:
            num = num.drop(columns=drop_cols)
        if hasattr(model, "n_features_in_") and num.shape[1] > int(model.n_features_in_):
            X = num.iloc[:, : int(model.n_features_in_)]
        else:
            X = num
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index)

def _apply_eligibility(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]:
    """Apply whitespace eligibility rules and track exclusion counts.

    Each rule is evaluated on the current mask of remaining rows so that the
    per-rule exclusion counts are disjoint. A boolean ``_eligible`` column is
    added to the returned frame indicating per-row eligibility.
    """
    mask = pd.Series(True, index=df.index)
    elig = getattr(getattr(cfg, "whitespace", object()), "eligibility", None)
    counts = {
        "start_rows": int(len(df)),
        "owned_excluded": 0,
        "recent_contact_excluded": 0,
        "open_deal_excluded": 0,
        "region_mismatch_excluded": 0,
    }
    if elig:
        if getattr(elig, "exclude_if_owned_ever", False) and "owned_division_pre_cutoff" in df.columns:
            owned_mask = df["owned_division_pre_cutoff"].astype(bool)
            cond = owned_mask & mask
            counts["owned_excluded"] = int(cond.sum())
            mask &= ~cond
        if getattr(elig, "exclude_if_recent_contact_days", 0) and "days_since_last_contact" in df.columns:
            rc = pd.to_numeric(df["days_since_last_contact"], errors="coerce").fillna(1e9) <= int(
                elig.exclude_if_recent_contact_days
            )
            cond = rc & mask
            counts["recent_contact_excluded"] = int(cond.sum())
            mask &= ~cond
        if getattr(elig, "exclude_if_open_deal", False) and "has_open_deal" in df.columns:
            od = df["has_open_deal"].astype(bool)
            cond = od & mask
            counts["open_deal_excluded"] = int(cond.sum())
            mask &= ~cond
        if getattr(elig, "require_region_match", False) and "region_match" in df.columns:
            mismatch = (~df["region_match"].astype(bool)) & mask
            counts["region_mismatch_excluded"] = int(mismatch.sum())
            mask &= ~mismatch
    df = df.copy()
    df["_eligible"] = mask
    counts["kept_rows"] = int(mask.sum())
    total_excluded = (
        counts["owned_excluded"]
        + counts["recent_contact_excluded"]
        + counts["open_deal_excluded"]
        + counts["region_mismatch_excluded"]
    )
    dropped = counts["start_rows"] - counts["kept_rows"]
    if total_excluded != dropped:
        logger.warning(
            "Eligibility counts mismatch: exclusions=%s dropped=%s", total_excluded, dropped
        )
    else:
        logger.info("Eligibility applied: %s kept, %s dropped", counts["kept_rows"], dropped)
    return df[mask].copy(), counts


def _scale_weights_by_coverage(base_weights: Iterable[float], als_norm: pd.Series, lift_norm: pd.Series, threshold: float = 0.30) -> Tuple[List[float], Dict[str, float]]:
    w = list(base_weights)
    if len(w) != 4:
        raise ValueError("Expected 4 weights: [p_icp_pct, lift, als, ev]")
    adjustments: Dict[str, float] = {}
    def coverage(s: pd.Series) -> float:
        return float((pd.to_numeric(s, errors='coerce').fillna(0.0) > 0).mean())
    cov_lift = coverage(lift_norm)
    cov_als = coverage(als_norm)
    # Downweight components with low coverage; keep p_icp and ev fixed
    # Scale factor = min(1, cov/threshold) so when cov<th, shrink proportionally
    def factor(cov: float) -> float:
        if not math.isfinite(cov) or cov <= 0:
            return 0.0
        return min(1.0, cov / max(1e-9, threshold))
    f_lift = factor(cov_lift)
    f_als = factor(cov_als)
    adjustments["aff_weight_factor"] = f_lift
    adjustments["als_weight_factor"] = f_als
    w_scaled = [w[0], w[1] * f_lift, w[2] * f_als, w[3]]
    s = sum(w_scaled)
    if s > 0:
        w_div = [wi / s for wi in w_scaled]
    else:
        w_div = [0.0] * len(w_scaled)
    if sum(w_div) == 0:
        base_sum = sum(w)
        if w[0] > 0 or w[3] > 0:
            w_div = [wi / base_sum for wi in w]
            logger.warning(
                "Weight scaling resulted in zero weights; falling back to base weights"
            )
        else:
            w_div = [1.0 / len(w)] * len(w)
            logger.warning(
                "Weight scaling resulted in zero weights; falling back to uniform weights"
            )
    return w_div, adjustments


def _explain(row: pd.Series) -> str:
    # Short reason, emphasize strongest 1-2 drivers, keep compliant
    parts: List[str] = []
    p = float(row.get('p_icp', 0.0))
    if p >= 0.80:
        parts.append(f"High p={p:.2f}")
    elif p >= 0.65:
        parts.append(f"Good p={p:.2f}")
    # Consider affinity and EV
    lift = float(row.get('lift_norm', 0.0))
    als = float(row.get('als_norm', 0.0))
    ev = float(row.get('EV_norm', 0.0))
    drivers: List[str] = []
    if lift >= 0.7:
        drivers.append("affinity")
    if als >= 0.7:
        drivers.append("ALS")
    if ev >= 0.7:
        drivers.append("EV")
    if drivers:
        parts.append("+ ".join([d for d in drivers[:2]]))
    if not parts:
        parts.append("Ranked opportunity")
    txt = "; ".join(parts)
    # guard length and tokens (basic)
    forbidden = {"race", "gender", "religion", "ssn", "social security", "age", "ethnicity", "disability", "veteran", "pregnan"}
    low = txt.lower()
    if any(t in low for t in forbidden):
        txt = "High likelihood"
    # limit length
    if len(txt) > 140:
        txt = txt[:137] + "..."
    return txt


@dataclass
class RankInputs:
    scores: pd.DataFrame  # columns: division_name, customer_id, icp_score, (optional) bought_in_division, EV proxy columns


def rank_whitespace(inputs: RankInputs, *, weights: Iterable[float] = (0.60, 0.20, 0.10, 0.10)) -> pd.DataFrame:
    df = inputs.scores.copy()
    if df.empty:
        return df
    # Apply eligibility rules and capture ALS centroid
    df, als_centroid = _apply_eligibility(df)
    if df.empty:
        return df
    # Per-division normalization of p_icp to percentile
    df['p_icp'] = pd.to_numeric(df['icp_score'], errors='coerce').fillna(0.0)
    df['p_icp_pct'] = df.groupby('division_name')['p_icp'].transform(_percentile_normalize)
    # Affinity lift and ALS similarity
    df['lift_norm'] = _compute_affinity_lift(df)
    df['als_norm'] = _compute_als_norm(df, owner_centroid=als_centroid)
    # EV proxy with cap and normalization
    try:
        from gosales.utils.config import load_config
        cfg = load_config()
    except Exception:
        cfg = None
    df['EV_norm'] = _compute_expected_value(df, cfg)

    # Ensure numeric types before blending
    for _col in ['p_icp_pct', 'lift_norm', 'als_norm', 'EV_norm']:
        df[_col] = pd.to_numeric(df.get(_col, 0.0), errors='coerce').fillna(0.0)

    # Scale weights by signal coverage
    w_adj, _ = _scale_weights_by_coverage(list(weights), df['als_norm'], df['lift_norm'])

    # Final blended score (champion)
    champion_score = (
        w_adj[0] * df['p_icp_pct'] +
        w_adj[1] * df['lift_norm'] +
        w_adj[2] * df['als_norm'] +
        w_adj[3] * df['EV_norm']
    ).astype(float)
    df['score'] = champion_score

    # Optional challenger: simple logistic meta-learner over normalized components
    try:
        _cfg = cfg
        challenger_on = bool(getattr(getattr(_cfg, 'whitespace', object()), 'challenger_enabled', False))
        challenger_model = str(getattr(getattr(_cfg, 'whitespace', object()), 'challenger_model', 'lr'))
    except Exception:
        challenger_on = False
        challenger_model = 'lr'
    if challenger_on and challenger_model == 'lr':
        try:
            from sklearn.linear_model import LogisticRegression

            feat_cols = list(CHALLENGER_FEAT_COLS)
            missing = [c for c in feat_cols if c not in df]
            for c in missing:
                df[c] = 0.0
            Xmeta = df[feat_cols].to_numpy(dtype=float)
            # Pseudo-label: use p_icp as soft target for ranking consistency; this is a heuristic challenger
            ysoft = df['p_icp'].to_numpy(dtype=float)
            # Fit Platt-like logistic on the normalized components to approximate p_icp ordering
            # Guard: small C for stability; deterministic
            clf = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=200)
            # Binarize soft target around its median to allow logistic to learn a separating surface
            import numpy as _np
            ybin = (ysoft >= _np.nanmedian(ysoft)).astype(int)
            if _np.unique(ybin).size >= 2:
                clf.fit(Xmeta, ybin)
                df['score_challenger'] = clf.decision_function(Xmeta).astype(float)
            else:
                df['score_challenger'] = champion_score
        except Exception:
            df['score_challenger'] = champion_score
    else:
        df['score_challenger'] = champion_score

    # Stable tie-breakers on champion score
    df = df.sort_values(by=['division_name', 'score', 'p_icp', 'customer_id'], ascending=[True, False, False, True], kind='mergesort')

    # Cooldown: de-emphasize accounts surfaced recently without action
    try:
        cooldown_days = int(getattr(getattr(cfg, 'whitespace', object()), 'cooldown_days', 0))
        cooldown_factor = float(getattr(getattr(cfg, 'whitespace', object()), 'cooldown_factor', 1.0))
    except Exception:
        cooldown_days = 0
        cooldown_factor = 1.0

    if (
        cooldown_days > 0
        and cooldown_factor < 1.0
        and 'days_since_last_surfaced' in df.columns
    ):
        days = pd.to_numeric(df['days_since_last_surfaced'], errors='coerce').fillna(cooldown_days + 1)
        mask = days < cooldown_days
        if mask.any():
            df.loc[mask, 'score'] = df.loc[mask, 'score'] * cooldown_factor
            # Refresh ordering after score adjustment
            df = df.sort_values(by=['division_name', 'score', 'p_icp', 'customer_id'], ascending=[True, False, False, True], kind='mergesort')

    # Explanations
    df['nba_reason'] = df.apply(_explain, axis=1)
    # Output columns
    out_cols = ['customer_id', 'division_name', 'score', 'score_challenger', 'p_icp', 'p_icp_pct', 'lift_norm', 'als_norm', 'EV_norm', 'nba_reason']
    present = [c for c in out_cols if c in df.columns]
    return df[present].reset_index(drop=True)


def save_ranked_whitespace(df: pd.DataFrame, *, cutoff_tag: str | None = None) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"whitespace_{cutoff_tag}.csv" if cutoff_tag else "whitespace.csv"
    path = OUTPUTS_DIR / name
    df.to_csv(path, index=False)
    return path
