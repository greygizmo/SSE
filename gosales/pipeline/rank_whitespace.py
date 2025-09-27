from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.grades import (
    assign_letter_grades_from_percentiles,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
try:
    from gosales.utils.normalize import normalize_division as _norm_division
except Exception:
    _norm_division = lambda s: str(s).strip()


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
    """Compute normalized affinity lift (market-basket) from best available column.

    Prefers ``mb_lift_max``; if absent, falls back to any column starting with
    ``mb_lift_max`` (e.g., ``mb_lift_max_lag60d``). Returns percentile-normalized values.
    """
    src = None
    if col in df.columns:
        src = col
    else:
        cands = [c for c in df.columns if str(c).startswith(col)]
        if cands:
            src = cands[0]
    if src is None:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    vals = pd.to_numeric(df[src], errors="coerce").fillna(0.0)
    return _percentile_normalize(vals)


# Store centroid path for reuse across runs
ALS_CENTROID_PATH = OUTPUTS_DIR / "als_owner_centroid.npy"
ASSETS_ALS_CENTROID_PATH = OUTPUTS_DIR / "assets_als_owner_centroid.npy"

def _assets_als_centroid_path_for_div(div: str) -> Path:
    key = str(div or "").strip().lower().replace(" ", "_").replace("/", "_")
    return OUTPUTS_DIR / f"assets_als_owner_centroid_{key}.npy"


def _apply_eligibility_and_centroid(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray | None]:
    """Filter out simple ineligible rows (owned pre-cutoff) while capturing ALS owner centroid.

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

    # Compute and persist assets-ALS centroid similarly (if columns present)
    try:
        aals_cols = [c for c in df.columns if c.startswith("als_assets_f")]
        if aals_cols and "owned_division_pre_cutoff" in df.columns:
            a_owners = df[df["owned_division_pre_cutoff"].astype(bool)]
            if not a_owners.empty:
                a_centroid = a_owners[aals_cols].astype(float).mean(axis=0).to_numpy(dtype=float)
                try:
                    ASSETS_ALS_CENTROID_PATH.parent.mkdir(parents=True, exist_ok=True)
                    np.save(ASSETS_ALS_CENTROID_PATH, a_centroid)
                except Exception:
                    pass
    except Exception:
        pass

    if "owned_division_pre_cutoff" in df.columns:
        df = df[~df["owned_division_pre_cutoff"].astype(bool)].copy()

    return df, centroid


def _compute_als_norm(df: pd.DataFrame, cfg=None, owner_centroid: np.ndarray | None = None) -> pd.Series:
    """Compute ALS similarity normalized to [0,1].

    - If owner_centroid is provided, use it; else derive from owned-pre-cutoff when available.
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
        if 'owned_division_pre_cutoff' in df.columns:
            try:
                base = mat[df['owned_division_pre_cutoff'].astype(bool)]
                centroid_vec = (base.mean(axis=0) if not base.empty else mat.mean(axis=0)).to_numpy(dtype=float)
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
        owned_drop_total = 0
        # Transaction-based ownership exclusion (last-12m proxy)
        if getattr(elig, "exclude_if_owned_ever", False) and "owned_division_pre_cutoff" in df.columns:
            owned_mask_tx = df["owned_division_pre_cutoff"].astype(bool)
            cond_tx = owned_mask_tx & mask
            owned_drop_total += int(cond_tx.sum())
            mask &= ~cond_tx
        # Optional: exclude if active assets indicate ownership in the same division
        try:
            if getattr(elig, "exclude_if_active_assets", False) and "division_name" in df.columns:
                # For each division, look for a matching per-division assets flag column
                divs = sorted(set(df['division_name'].astype(str).dropna().unique()))
                dropped_here = 0
                for d in divs:
                    col = f"owns_assets_div_{_norm_division(d).lower()}"
                    if col in df.columns:
                        m = (df['division_name'].astype(str) == d) & df[col].astype(bool) & mask
                        dropped_here += int(m.sum())
                        mask &= ~m
                owned_drop_total += dropped_here
        except Exception:
            pass
        # Optional reinclusion policy: allow former owners back if assets have been expired long enough
        try:
            days_thr = int(getattr(elig, "reinclude_if_assets_expired_days", 0) or 0)
        except Exception:
            days_thr = 0
        if days_thr > 0 and "division_name" in df.columns:
            try:
                divs = sorted(set(df['division_name'].astype(str).dropna().unique()))
                readd = pd.Series(False, index=df.index)
                for d in divs:
                    col = f"former_owner_div_{_norm_division(d).lower()}"
                    if col in df.columns:
                        # Prefer per-division days-since if available
                        dcol = f"assets_days_since_last_expiration_div_{_norm_division(d).lower()}"
                        if dcol in df.columns:
                            days = pd.to_numeric(df[dcol], errors='coerce').fillna(0)
                        else:
                            days = pd.to_numeric(df.get('assets_days_since_last_expiration', 0), errors='coerce').fillna(0)
                        cand = ((df['division_name'].astype(str) == d) & df[col].astype(bool) & (days >= days_thr))
                        readd |= cand
                to_reinclude = readd & (~mask)
                if to_reinclude.any():
                    # Re-include these rows
                    mask |= to_reinclude
                    # Adjust owned_excluded count to reflect reinclusion
                    owned_drop_total -= int(to_reinclude.sum())
            except Exception:
                pass
        counts["owned_excluded"] = int(max(0, owned_drop_total))
        if getattr(elig, "exclude_if_recent_contact_days", 0) and "days_since_last_contact" in df.columns:
            rc = pd.to_numeric(df["days_since_last_contact"], errors="coerce").fillna(1e9) <= int(
                getattr(elig, "exclude_if_recent_contact_days", 0)
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


def _compute_item2vec_norm(df: pd.DataFrame, owner_centroid: np.ndarray | None = None) -> pd.Series:
    """Compute item2vec similarity to owner centroid if i2v features present.

    Returns normalized [0,1] similarities; zeros if no features.
    """
    i2v_cols = [c for c in df.columns if c.startswith("i2v_f")]
    if not i2v_cols:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    mat = df[i2v_cols].astype(float)
    if mat.empty:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    if owner_centroid is not None and owner_centroid.size == mat.shape[1]:
        centroid_vec = owner_centroid.astype(float)
    else:
        centroid_vec = mat.mean(axis=0).to_numpy(dtype=float)
    # Cosine similarity to centroid
    try:
        sim = cosine_similarity(mat, centroid_vec.reshape(1, -1)).ravel()
    except Exception:
        sim = np.zeros(len(df), dtype=float)
    # Percentile normalize
    return _percentile_normalize(pd.Series(sim, index=df.index))

def _compute_assets_als_norm(df: pd.DataFrame, owner_centroid: np.ndarray | None = None) -> pd.Series:
    """Compute assets-ALS similarity with per-division centroids when possible.

    - If a cached centroid exists for the division (assets_als_owner_centroid_<division>.npy), use it.
    - Else, compute from rows with owns_assets_div_<division> == True when available; else from owned_division_pre_cutoff; else global mean.
    - Falls back to global cached centroid (ASSETS_ALS_CENTROID_PATH) when division-specific unavailable.
    """
    cols = [c for c in df.columns if c.startswith("als_assets_f")]
    if not cols:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    if df.empty:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    # If no division column, do a global computation
    if 'division_name' not in df.columns:
        mat = df[cols].astype(float)
        if mat.empty:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
        try:
            if owner_centroid is not None:
                centroid_vec = np.asarray(owner_centroid, dtype=float)
            elif ASSETS_ALS_CENTROID_PATH.exists():
                centroid_vec = np.load(ASSETS_ALS_CENTROID_PATH)
            else:
                centroid_vec = mat.mean(axis=0).to_numpy(dtype=float)
            sims = cosine_similarity(mat, centroid_vec.reshape(1, -1)).ravel()
            return _percentile_normalize(pd.Series(sims, index=df.index))
        except Exception:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    out = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    for div, g in df.groupby('division_name', dropna=False):
        idx = g.index
        sub = g[cols].astype(float)
        if sub.empty:
            continue
        # Try division-specific cached centroid
        path = _assets_als_centroid_path_for_div(div)
        centroid_vec = None
        if owner_centroid is not None:
            centroid_vec = np.asarray(owner_centroid, dtype=float)
        else:
            if path.exists():
                try:
                    centroid_vec = np.load(path)
                except Exception:
                    centroid_vec = None
            if centroid_vec is None:
                # Compute from owns_assets_div_<div> when available
                owns_col = f"owns_assets_div_{str(div).strip().lower()}"
                try:
                    if owns_col in g.columns and g[owns_col].astype(bool).any():
                        centroid_vec = g.loc[g[owns_col].astype(bool), cols].astype(float).mean(axis=0).to_numpy(dtype=float)
                        try:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(path, centroid_vec)
                        except Exception:
                            pass
                except Exception:
                    centroid_vec = None
            if centroid_vec is None:
                # Fallback to transaction-based owned centroid or global mean
                try:
                    if 'owned_division_pre_cutoff' in g.columns and g['owned_division_pre_cutoff'].astype(bool).any():
                        centroid_vec = g.loc[g['owned_division_pre_cutoff'].astype(bool), cols].astype(float).mean(axis=0).to_numpy(dtype=float)
                    else:
                        centroid_vec = sub.mean(axis=0).to_numpy(dtype=float)
                except Exception:
                    centroid_vec = sub.mean(axis=0).to_numpy(dtype=float)
        try:
            sims = cosine_similarity(sub, centroid_vec.reshape(1, -1)).ravel()
            out.loc[idx] = _percentile_normalize(pd.Series(sims, index=idx))
        except Exception:
            out.loc[idx] = 0.0
    return out


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
    # Standardize key dtypes
    try:
        if 'customer_id' in df.columns:
            df['customer_id'] = df['customer_id'].astype(str)
    except Exception:
        pass
    if df.empty:
        return df
    # Apply simple ownership eligibility and capture ALS centroid
    df, als_centroid = _apply_eligibility_and_centroid(df)
    if df.empty:
        return df
    # Per-division normalization of p_icp to percentile
    df['p_icp'] = pd.to_numeric(df['icp_score'], errors='coerce').fillna(0.0)
    df['p_icp_pct'] = df.groupby('division_name')['p_icp'].transform(_percentile_normalize)
    # Affinity lift and ALS similarity
    df['lift_norm'] = _compute_affinity_lift(df)
    df['als_norm'] = _compute_als_norm(df, owner_centroid=als_centroid)
    # ALS coverage enforcement and optional assets-ALS / item2vec backfill
    try:
        from gosales.utils.config import load_config
        _cfg = load_config()
        als_thr = float(getattr(getattr(_cfg, 'whitespace', object()), 'als_coverage_threshold', 0.30))
        use_i2v = bool(getattr(getattr(_cfg, 'features', object()), 'use_item2vec', False))
    except Exception:
        als_thr = 0.30
        use_i2v = False
    try:
        cov_als = (pd.to_numeric(df['als_norm'], errors='coerce').fillna(0.0) > 0).mean()
    except Exception:
        cov_als = 0.0
    # Prefer assets-ALS fallback when available, else item2vec
    assets_als_present = any(c.startswith('als_assets_f') for c in df.columns)
    if cov_als < als_thr:
        mask_zero_als = pd.to_numeric(df['als_norm'], errors='coerce').fillna(0.0) <= 0
        if assets_als_present:
            try:
                aals = _compute_assets_als_norm(df, owner_centroid=None)
                if mask_zero_als.any():
                    df.loc[mask_zero_als, 'als_norm'] = aals[mask_zero_als]
            except Exception:
                pass
        # If still zero, try i2v
        i2v_present = any(c.startswith('i2v_f') for c in df.columns)
        if (use_i2v or i2v_present):
            i2v_norm = _compute_item2vec_norm(df, owner_centroid=None)
            mask_zero_als = pd.to_numeric(df['als_norm'], errors='coerce').fillna(0.0) <= 0
            if mask_zero_als.any():
                df.loc[mask_zero_als, 'als_norm'] = i2v_norm[mask_zero_als]
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

    # Scale weights by signal coverage (global or by segment)
    try:
        als_cov_thr = float(getattr(getattr(cfg, 'whitespace', object()), 'als_coverage_threshold', 0.30)) if cfg else 0.30
    except Exception:
        als_cov_thr = 0.30

    # Optional segment-wise weighting
    seg_cols_cfg: List[str] = []
    seg_min_rows = 0
    try:
        seg_cols_cfg = list(getattr(getattr(cfg, 'whitespace', object()), 'segment_columns', []))
        seg_min_rows = int(getattr(getattr(cfg, 'whitespace', object()), 'segment_min_rows', 250))
    except Exception:
        seg_cols_cfg = []
        seg_min_rows = 250

    # Derive size_bin if requested and feasible
    if any(c.lower() == 'size_bin' for c in seg_cols_cfg) and 'size_bin' not in df.columns:
        base_size_col = None
        for cand in ['total_gp_all_time', 'rfm__all__gp_sum__12m', 'transactions_last_12m', 'rfm__all__tx_n__12m']:
            if cand in df.columns:
                base_size_col = cand
                break
        if base_size_col is not None:
            try:
                q = pd.qcut(pd.to_numeric(df[base_size_col], errors='coerce').fillna(0.0), 3, labels=['small','mid','large'])
                df['size_bin'] = q.astype(str)
            except Exception:
                pass

    seg_cols = [c for c in seg_cols_cfg if c in df.columns]
    if seg_cols:
        df['score'] = 0.0
        for keys, g in df.groupby(seg_cols, dropna=False):
            idx = g.index
            if len(g) < seg_min_rows:
                # Fall back to global weights
                w_adj, _ = _scale_weights_by_coverage(list(weights), df.loc[idx, 'als_norm'], df.loc[idx, 'lift_norm'], threshold=als_cov_thr)
            else:
                w_adj, _ = _scale_weights_by_coverage(list(weights), g['als_norm'], g['lift_norm'], threshold=als_cov_thr)
            sc = (
                w_adj[0] * g['p_icp_pct'] +
                w_adj[1] * g['lift_norm'] +
                w_adj[2] * g['als_norm'] +
                w_adj[3] * g['EV_norm']
            ).astype(float)
            df.loc[idx, 'score'] = sc
    else:
        w_adj, _ = _scale_weights_by_coverage(list(weights), df['als_norm'], df['lift_norm'], threshold=als_cov_thr)
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
    # Executive-friendly percentiles and letter grades
    try:
        # p_icp percentile is already in p_icp_pct; derive letter grade
        df['p_icp_grade'] = assign_letter_grades_from_percentiles(pd.to_numeric(df.get('p_icp_pct', 0.0), errors='coerce').fillna(0.0))
    except Exception:
        df['p_icp_grade'] = 'F'
    try:
        # Champion score percentile and grade per division
        if 'score' in df.columns and 'division_name' in df.columns:
            pct = df.groupby('division_name')['score'].rank(method='average', pct=True).astype(float)
            df['score_pct'] = pct
            df['score_grade'] = assign_letter_grades_from_percentiles(pct)
        else:
            df['score_pct'] = 0.0
            df['score_grade'] = 'F'
    except Exception:
        df['score_pct'] = 0.0
        df['score_grade'] = 'F'
    # Output columns
    out_cols = [
        'customer_id', 'customer_name', 'division_name',
        'score', 'score_pct', 'score_grade', 'score_challenger',
        'p_icp', 'p_icp_pct', 'p_icp_grade', 'lift_norm', 'als_norm', 'EV_norm',
        'nba_reason'
    ]
    present = [c for c in out_cols if c in df.columns]
    return df[present].reset_index(drop=True)


def save_ranked_whitespace(df: pd.DataFrame, *, cutoff_tag: str | None = None) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"whitespace_{cutoff_tag}.csv" if cutoff_tag else "whitespace.csv"
    path = OUTPUTS_DIR / name
    df.to_csv(path, index=False)
    return path
