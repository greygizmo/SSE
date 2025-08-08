from __future__ import annotations

from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd

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
    feat_cols = None
    if feature_list_path.exists():
        feat_cols = pd.read_json(feature_list_path).tolist()
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


def _compute_affinity_lift(df: pd.DataFrame) -> pd.Series:
    # Placeholder: if market-basket features are present, e.g., mb_lift_max
    for c in ["mb_lift_max", "mb_lift_mean"]:
        if c in df.columns:
            lift = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            # Normalize to [0,1]
            return _percentile_normalize(lift).rename("lift_norm")
    return pd.Series(0.0, index=df.index, name="lift_norm")


def _compute_als_norm(df: pd.DataFrame, cfg) -> pd.Series:
    # If ALS embeddings were exported to per-customer division similarity, consume; else zeros
    for c in ["als_sim_division", "als_affinity"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            norm = _percentile_normalize(s).rename("als_norm")
            return norm
    return pd.Series(0.0, index=df.index, name="als_norm")


def _apply_eligibility(df: pd.DataFrame, cfg) -> pd.DataFrame:
    # Assume df has columns indicating current ownership, region, open deals, recent contacts as available
    mask = pd.Series(True, index=df.index)
    elig = cfg.whitespace.eligibility
    if elig.exclude_if_owned_ever and 'owned_division_pre_cutoff' in df.columns:
        mask &= ~df['owned_division_pre_cutoff'].astype(bool)
    if elig.exclude_if_recent_contact_days and 'days_since_last_contact' in df.columns:
        mask &= (pd.to_numeric(df['days_since_last_contact'], errors='coerce').fillna(1e9) > int(elig.exclude_if_recent_contact_days))
    if elig.exclude_if_open_deal and 'has_open_deal' in df.columns:
        mask &= ~df['has_open_deal'].astype(bool)
    if elig.require_region_match and 'region_match' in df.columns:
        mask &= df['region_match'].astype(bool)
    out = df.loc[mask].copy()
    out['_eligibility_kept'] = mask.sum()
    return out


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
    for div in divisions:
        # For now, reuse the latest features parquet for the cutoff
        feat_path = OUTPUTS_DIR / f"features_{div.lower()}_{cutoff}.parquet"
        if not feat_path.exists():
            logger.warning(f"Missing features for division {div} at cutoff {cutoff}: {feat_path}")
            continue
        df = pd.read_parquet(feat_path)

        # Eligibility
        df = _apply_eligibility(df, cfg)
        if df.empty:
            continue

        # Signals
        p_icp, p_icp_pct = _score_p_icp(df.drop(columns=['customer_id', 'bought_in_division'], errors='ignore'), div)
        lift_norm = _compute_affinity_lift(df)
        als_norm = _compute_als_norm(df, cfg)
        ev_norm = _compute_expected_value(df, cfg)

        tmp = pd.DataFrame({
            'customer_id': df['customer_id'].values,
            'division': div,
            'p_icp': p_icp.values,
            'p_icp_pct': p_icp_pct.values,
            'lift_norm': lift_norm.values,
            'als_norm': als_norm.values,
            'EV_norm': ev_norm.values,
        })
        # Blend
        tmp['score'] = (
            w[0] * tmp['p_icp_pct'] +
            w[1] * tmp['lift_norm'] +
            w[2] * tmp['als_norm'] +
            w[3] * tmp['EV_norm']
        )
        rows.append(tmp)

    if not rows:
        logger.warning("No ranked rows produced")
        return

    out = pd.concat(rows, ignore_index=True)
    # Tie-breakers: higher p_icp, higher EV, then customer_id asc
    out = out.sort_values(['score', 'p_icp', 'EV_norm', 'customer_id'], ascending=[False, False, False, True])
    # Deterministic checksum
    checksum = pd.util.hash_pandas_object(out[['customer_id','division','score']]).sum()
    metrics = {
        'cutoff': cutoff,
        'weights': w,
        'rows': int(len(out)),
        'checksum': int(checksum),
    }
    out_path = OUTPUTS_DIR / f"whitespace_{cutoff}.csv"
    out.to_csv(out_path, index=False)
    (OUTPUTS_DIR / f"whitespace_metrics_{cutoff}.json").write_text(pd.Series(metrics).to_json(indent=2), encoding='utf-8')
    logger.info(f"Wrote {out_path} with {len(out)} rows")


if __name__ == "__main__":
    main()


