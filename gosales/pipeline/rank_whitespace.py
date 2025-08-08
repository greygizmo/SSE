from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import click
import numpy as np
import pandas as pd
import json

from gosales.utils.config import load_config
from gosales.utils.paths import OUTPUTS_DIR, MODELS_DIR
from gosales.utils.logger import get_logger


logger = get_logger(__name__)


def _percentile_normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    ranks = series.rank(method="average", pct=True)
    return ranks.clip(0.0, 1.0)


def _load_model_for_division(division: str):
    # Lazy import to avoid heavy deps on CLI load
    import joblib
    model_path = MODELS_DIR / f"{division.lower()}_model" / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def _score_p_icp(df: pd.DataFrame, division: str) -> Tuple[pd.Series, pd.Series]:
    model = _load_model_for_division(division)
    feature_list_path = MODELS_DIR / f"{division.lower()}_model" / "feature_list.json"
    feat_cols: List[str] | None = None
    if feature_list_path.exists():
        try:
            feat_cols = json.loads(feature_list_path.read_text(encoding="utf-8"))
        except Exception:
            feat_cols = None
    X = df[feat_cols] if feat_cols else df.select_dtypes(include=[np.number])
    p = model.predict_proba(X)[:, 1]
    p = pd.Series(p, index=df.index, name="p_icp")
    p_pct = _percentile_normalize(p).rename("p_icp_pct")
    return p, p_pct


def _compute_expected_value(df: pd.DataFrame, cfg) -> pd.Series:
    # Simple proxy: use recent all-scope GP (e.g., 12m) if available; else 0
    col = None
    for c in ["rfm__all__gp_sum__12m", "rfm__all__gp_sum__24m", "total_gp_all_time"]:
        if c in df.columns:
            col = c
            break
    ev = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col else pd.Series(0.0, index=df.index)
    cap = ev.quantile(cfg.whitespace.ev_cap_percentile)
    ev_capped = ev.clip(upper=cap)
    # Normalize
    ev_norm = _percentile_normalize(ev_capped).rename("EV_norm")
    return ev_norm


def _compute_ev_norm_by_segment(df: pd.DataFrame, cfg) -> pd.Series:
    # Prefer segment medians if segment columns present; fallback to global proxy
    seg_cols = []
    for c in ["industry", "industry_sub", "region", "territory"]:
        if c in df.columns:
            seg_cols.append(c)
    # Raw EV signal as before
    col = None
    for c in ["rfm__all__gp_sum__12m", "rfm__all__gp_sum__24m", "total_gp_all_time"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        return _compute_expected_value(df, cfg)
    base = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    # Segment medians blended with global median (simple average for now)
    if seg_cols:
        med = []
        for c in seg_cols:
            try:
                seg_med = df.groupby(c)[col].transform(lambda s: pd.to_numeric(s, errors='coerce').median())
                med.append(seg_med)
            except Exception:
                continue
        if med:
            seg_est = sum(med) / len(med)
            raw_ev = 0.5 * base + 0.5 * seg_est.fillna(base)
        else:
            raw_ev = base
    else:
        raw_ev = base
    cap = raw_ev.quantile(cfg.whitespace.ev_cap_percentile)
    ev_capped = raw_ev.clip(upper=cap)
    return _percentile_normalize(ev_capped).rename("EV_norm")


def _compute_affinity_lift(df: pd.DataFrame) -> pd.Series:
    # Prefer engineered affinity feature; else placeholders if present
    for c in [
        "affinity__div__lift_topk__12m",
        "mb_lift_max",
        "mb_lift_mean",
    ]:
        if c in df.columns:
            lift = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            return _percentile_normalize(lift).rename("lift_norm")
    return pd.Series(0.0, index=df.index, name="lift_norm")


def _compute_als_norm(df: pd.DataFrame, cfg) -> pd.Series:
    # If explicit similarity present, use it
    for c in ["als_sim_division", "als_affinity"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            return _percentile_normalize(s).rename("als_norm")
    # Else, if embedding columns present (als_f0..als_fN), compute dot with division centroid from pre-cutoff owners
    als_cols = [c for c in df.columns if c.startswith("als_f")]
    if als_cols:
        try:
            owned_mask = df.get('owned_division_pre_cutoff', pd.Series(False, index=df.index)).astype(bool)
            base = df.loc[owned_mask, als_cols]
            if base.empty:
                return pd.Series(0.0, index=df.index, name="als_norm")
            centroid = base.mean(axis=0).values
            # dot product
            mat = df[als_cols].fillna(0.0).values
            sim = mat.dot(centroid)
            s = pd.Series(sim, index=df.index)
            return _percentile_normalize(s).rename("als_norm")
        except Exception:
            return pd.Series(0.0, index=df.index, name="als_norm")
    return pd.Series(0.0, index=df.index, name="als_norm")


def _apply_eligibility(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]:
    # Assume df has columns indicating current ownership, region, open deals, recent contacts as available
    mask = pd.Series(True, index=df.index)
    elig = cfg.whitespace.eligibility
    counts = {
        'start_rows': int(len(df)),
        'owned_excluded': 0,
        'recent_contact_excluded': 0,
        'open_deal_excluded': 0,
        'region_mismatch_excluded': 0,
    }
    if elig.exclude_if_owned_ever and 'owned_division_pre_cutoff' in df.columns:
        owned_mask = df['owned_division_pre_cutoff'].astype(bool)
        counts['owned_excluded'] = int(owned_mask.sum())
        mask &= ~owned_mask
    if elig.exclude_if_recent_contact_days and 'days_since_last_contact' in df.columns:
        cond = (pd.to_numeric(df['days_since_last_contact'], errors='coerce').fillna(1e9) <= int(elig.exclude_if_recent_contact_days))
        counts['recent_contact_excluded'] = int(cond.sum())
        mask &= ~cond
    if elig.exclude_if_open_deal and 'has_open_deal' in df.columns:
        od = df['has_open_deal'].astype(bool)
        counts['open_deal_excluded'] = int(od.sum())
        mask &= ~od
    if elig.require_region_match and 'region_match' in df.columns:
        rm = df['region_match'].astype(bool)
        counts['region_mismatch_excluded'] = int((~rm).sum())
        mask &= rm
    out = df.loc[mask].copy()
    counts['kept_rows'] = int(len(out))
    out['_eligibility_kept'] = counts['kept_rows']
    return out, counts


def _explain(row: pd.Series) -> str:
    parts = []
    # Main driver: calibrated probability
    if pd.notna(row.get('p_icp')):
        parts.append(f"High p={row['p_icp']:.2f}")
    # Affinity
    if row.get('lift_norm', 0.0) > 0.75:
        parts.append("strong affinity")
    # ALS
    if row.get('als_norm', 0.0) > 0.75:
        parts.append("ALS similarity")
    # EV proxy (approximate dollarization via percentile bucket)
    if pd.notna(row.get('EV_norm')) and row['EV_norm'] > 0.7:
        parts.append("high EV")
    if not parts:
        parts = ["solid ICP"]
    txt = "; ".join(parts)
    return txt[:140]


@click.command()
@click.option("--cutoff", required=True)
@click.option("--window-months", default=6, type=int)
@click.option("--division", default=None, help="If set, only rank this division; otherwise rank all with models")
@click.option("--weights", default=None, help="Comma weights for [p_icp_pct, lift_norm, als_norm, EV_norm]")
@click.option("--normalize", default=None, help="percentile|pooled")
@click.option("--capacity-mode", default=None)
@click.option("--accounts-per-rep", default=None, type=int)
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
def main(cutoff: str, window_months: int, division: str | None, weights: str | None, normalize: str | None, capacity_mode: str | None, accounts_per_rep: int | None, config: str) -> None:
    cfg = load_config(config, cli_overrides={"run": {"prediction_window_months": window_months}})
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover divisions with models if not specified
    divisions = [division] if division else [p.name.replace("_model", "") for p in MODELS_DIR.glob("*_model") if (p / "model.pkl").exists()]
    if not divisions:
        logger.warning("No models found for ranking")
        return

    # Parse weights override
    w = cfg.whitespace.weights
    if weights:
        try:
            w = [float(x) for x in weights.split(",")]
            if len(w) != 4:
                raise ValueError
        except Exception:
            logger.warning("Invalid --weights; using config defaults")
            w = cfg.whitespace.weights

    rows = []
    metrics_div: list[dict] = []
    for div in divisions:
        # For now, reuse the latest features parquet for the cutoff
        feat_path = OUTPUTS_DIR / f"features_{div.lower()}_{cutoff}.parquet"
        if not feat_path.exists():
            logger.warning(f"Missing features for division {div} at cutoff {cutoff}: {feat_path}")
            continue
        df = pd.read_parquet(feat_path)

        # Derive owned pre-cutoff if not present (any div-scope tx in windows implies owned)
        if 'owned_division_pre_cutoff' not in df.columns:
            win_cols = [f'rfm__div__tx_n__{w}m' for w in cfg.features.windows_months]
            present_cols = [c for c in win_cols if c in df.columns]
            if present_cols:
                df['owned_division_pre_cutoff'] = (df[present_cols].sum(axis=1) > 0).astype(bool)
            elif 'rfm__div__recency_days__life' in df.columns:
                df['owned_division_pre_cutoff'] = pd.to_numeric(df['rfm__div__recency_days__life'], errors='coerce').fillna(0) > 0
            else:
                df['owned_division_pre_cutoff'] = False

        # Eligibility
        df, elig_counts = _apply_eligibility(df, cfg)
        if df.empty:
            continue

        # Signals
        p_icp, p_icp_pct = _score_p_icp(df.drop(columns=['customer_id', 'bought_in_division'], errors='ignore'), div)
        lift_norm = _compute_affinity_lift(df)
        als_norm = _compute_als_norm(df, cfg)
        ev_norm = _compute_ev_norm_by_segment(df, cfg)

        # Weight degradation if coverage is sparse
        w_div = list(w)
        adjustments = {}
        try:
            als_cov = float((als_norm > 0).mean())
            aff_cov = float((lift_norm > 0).mean())
            thr = float(cfg.whitespace.als_coverage_threshold)
            if als_cov < thr and w_div[2] > 0:
                factor = als_cov / max(1e-9, thr)
                adjustments['als_weight_factor'] = round(factor, 3)
                w_div[2] *= factor
            if aff_cov < thr and w_div[1] > 0:
                factor = aff_cov / max(1e-9, thr)
                adjustments['aff_weight_factor'] = round(factor, 3)
                w_div[1] *= factor
            s = sum(w_div)
            if s > 0:
                w_div = [x / s for x in w_div]
        except Exception:
            pass

        tmp = pd.DataFrame({
            'customer_id': df['customer_id'].values,
            'division': div,
            'p_icp': p_icp.values,
            'p_icp_pct': p_icp_pct.values,
            'lift_norm': lift_norm.values,
            'als_norm': als_norm.values,
            'EV_norm': ev_norm.values,
            'label': df.get('bought_in_division', pd.Series(0, index=df.index)).astype(int).values,
        })
        # Blend
        tmp['score'] = (
            w_div[0] * tmp['p_icp_pct'] +
            w_div[1] * tmp['lift_norm'] +
            w_div[2] * tmp['als_norm'] +
            w_div[3] * tmp['EV_norm']
        )
        # Explanations
        tmp['nba_reason'] = tmp.apply(_explain, axis=1)
        rows.append(tmp)
        metrics_div.append({
            'division': div,
            'eligibility_counts': elig_counts,
            'coverage': {
                'als': float((als_norm > 0).mean()),
                'affinity': float((lift_norm > 0).mean()),
            },
            'weights_final': w_div,
            'adjustments': adjustments,
        })

    if not rows:
        logger.warning("No ranked rows produced")
        return

    out = pd.concat(rows, ignore_index=True)
    # Tie-breakers: higher p_icp, higher EV, then customer_id asc
    out = out.sort_values(['score', 'p_icp', 'EV_norm', 'customer_id'], ascending=[False, False, False, True])
    # Capacity slicing (top_percent mode)
    cap_mode = capacity_mode or cfg.whitespace.capacity_mode
    thresholds_rows = []
    selected = out
    if cap_mode == 'top_percent':
        cap_percent = cfg.modeling.capacity_percent
        k = max(1, int(len(out) * (cap_percent / 100.0)))
        thr = float(np.sort(out['score'].values)[-k])
        selected = out[out['score'] >= thr]
        thresholds_rows.append({'mode': 'top_percent', 'k_percent': cap_percent, 'threshold': thr, 'count': int(len(selected))})
    elif cap_mode == 'per_rep':
        # Select top N per rep if 'rep' column exists, else fallback to top_percent
        n = accounts_per_rep or cfg.whitespace.accounts_per_rep
        if 'rep' in out.columns:
            selected = out.sort_values(['rep','score','p_icp','EV_norm','customer_id'], ascending=[True, False, False, False, True])
            selected = selected.groupby('rep', as_index=False).head(n)
            thresholds_rows.append({'mode': 'per_rep', 'accounts_per_rep': int(n), 'count': int(len(selected))})
        else:
            logger.warning("per_rep capacity requested but 'rep' column not found; falling back to top_percent")
            cap_percent = cfg.modeling.capacity_percent
            k = max(1, int(len(out) * (cap_percent / 100.0)))
            thr = float(np.sort(out['score'].values)[-k])
            selected = out[out['score'] >= thr]
            thresholds_rows.append({'mode': 'top_percent', 'k_percent': cap_percent, 'threshold': thr, 'count': int(len(selected))})

    # Cross-division bias share at capacity
    shares = (
        selected.groupby('division')['customer_id'].size().sort_values(ascending=False)
    )
    total_sel = max(1, int(len(selected)))
    share_map = {div: float(cnt) / total_sel for div, cnt in shares.items()}
    max_share = max(share_map.values()) if share_map else 0.0
    if max_share > cfg.whitespace.bias_division_max_share_topN:
        logger.warning(f"Division share {max_share:.2f} exceeds bias guard threshold at capacity")

    # Deterministic checksum
    checksum = pd.util.hash_pandas_object(out[['customer_id','division','score']]).sum()
    # Capture@K (global) using available labels
    capture_at_k = {}
    try:
        if 'label' in out.columns:
            total_pos = int(out['label'].sum())
            for k in list(set(getattr(cfg.modeling, 'top_k_percents', [10]))):
                kk = max(1, int(len(out) * (k / 100.0)))
                topk = out.nlargest(kk, ['score','p_icp','EV_norm','customer_id'])
                hit = int(topk['label'].sum())
                capture_at_k[str(k)] = float(hit / max(1, total_pos))
    except Exception:
        pass

    metrics = {
        'cutoff': cutoff,
        'weights': w,
        'normalize': normalize or cfg.whitespace.normalize,
        'rows': int(len(out)),
        'checksum': int(checksum),
        'capacity_mode': cap_mode,
        'selected_rows': int(len(selected)),
        'division_shares_topN': share_map,
        'capture_at_k': capture_at_k,
        'by_division': metrics_div,
    }
    out_path = OUTPUTS_DIR / f"whitespace_{cutoff}.csv"
    out.to_csv(out_path, index=False)
    # Explanations export (subset)
    expl_path = OUTPUTS_DIR / f"whitespace_explanations_{cutoff}.csv"
    out[['customer_id','division','score','nba_reason','p_icp','p_icp_pct','lift_norm','als_norm','EV_norm']].to_csv(expl_path, index=False)
    # Thresholds export
    pd.DataFrame(thresholds_rows).to_csv(OUTPUTS_DIR / f"thresholds_whitespace_{cutoff}.csv", index=False)
    # Metrics export
    (OUTPUTS_DIR / f"whitespace_metrics_{cutoff}.json").write_text(pd.Series(metrics).to_json(indent=2), encoding='utf-8')
    logger.info(f"Wrote {out_path} with {len(out)} rows; selected {len(selected)} for capacity mode {cap_mode}")


if __name__ == "__main__":
    main()


