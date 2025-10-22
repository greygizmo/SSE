from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import polars as pl

from gosales.utils.config import load_config
from gosales.utils.db import get_curated_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger


logger = get_logger(__name__)


def _compute_weights(df: pd.DataFrame, *, use_qty: bool, include_revenue: bool, price_factor: float, cap: Optional[float], half_life_days: int, end_ts: pd.Timestamp) -> pd.Series:
    qty = pd.to_numeric(df.get('quantity', 1.0), errors='coerce').fillna(1.0)
    gp = pd.to_numeric(df.get('gross_profit', 0.0), errors='coerce').fillna(0.0).clip(lower=0.0)
    rev = pd.to_numeric(df.get('revenue', 0.0), errors='coerce').fillna(0.0).clip(lower=0.0)
    q_term = np.log1p(1.0 + qty) if use_qty else 0.0
    gp_term = np.log1p(1.0 + gp)
    price_term = price_factor * (np.log1p(1.0 + rev) if include_revenue else 0.0)
    w = (q_term + gp_term + price_term).astype('float64')
    if cap is not None:
        w = np.minimum(w, float(cap))
    # time decay
    lam = np.log(2.0) / float(max(1, int(half_life_days)))
    age_days = (end_ts - pd.to_datetime(df['order_date'], errors='coerce')).dt.days.clip(lower=0)
    decay = np.exp(-lam * age_days.astype('float64'))
    w = (w * decay).astype('float64')
    return w


@click.command()
@click.option("--cutoff", required=True, type=str, help="Cutoff date (YYYY-MM-DD)")
@click.option("--division", required=False, type=str, help="Optional division name for scoping")
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
@click.option("--rows", default=None, type=int, help="Deterministic sample size (defaults to features.als_weight_stats_sample_rows)")
@click.option("--output", default=None, type=str, help="Optional path to write JSON; prints to stdout if omitted")
def main(cutoff: str, division: Optional[str], config: str, rows: Optional[int], output: Optional[str]) -> None:
    """Dump ALS weighting policy and sampled distribution stats for quick audits."""

    cfg = load_config(config)
    engine = get_curated_connection()
    lookback = int(getattr(cfg.features, 'als_lookback_months', 12) or 12)
    start = (pd.to_datetime(cutoff) - pd.DateOffset(months=lookback)).strftime("%Y-%m-%d")
    end = pd.to_datetime(cutoff).strftime("%Y-%m-%d")

    scope_by_goal = bool(getattr(cfg.features, 'als_scope_by_goal', True))
    division_scoped = bool(getattr(cfg.features, 'als_division_scoped', True))
    use_qty = bool(getattr(cfg.features, 'als_weight_by_quantity', True))
    include_revenue = bool(getattr(cfg.features, 'als_weight_include_revenue', True))
    price_factor = float(getattr(cfg.features, 'als_weight_price_factor', 1.0))
    cap = getattr(cfg.features, 'als_weight_cap', None)
    cap = float(cap) if cap is not None else None
    half_life = int(getattr(cfg.features, 'als_time_half_life_days', 180))
    sample_n = int(rows or getattr(cfg.features, 'als_weight_stats_sample_rows', 100000) or 0)

    # Query minimal columns
    q = (
        "SELECT customer_id, order_date, product_sku, product_division, product_goal, quantity, gross_profit, revenue "
        "FROM fact_transactions WHERE order_date <= :end AND order_date >= :start"
    )
    df = pd.read_sql_query(q, engine, params={"start": start, "end": end})
    if df.empty:
        result = {"status": "empty", "cutoff": cutoff, "division": division}
        out = json.dumps(result, indent=2)
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(out, encoding="utf-8")
        else:
            print(out)
        return

    # Optional scoping
    if division and division_scoped:
        scope_col = 'product_goal' if (scope_by_goal and ('product_goal' in df.columns)) else 'product_division'
        aliases = getattr(cfg.features, 'division_aliases', {}) or {}
        allow = [division.strip().lower()] + [v for v in (aliases.get(division.strip().lower(), []) or [])]
        allow = [str(a).strip().lower() for a in allow]
        df = df[df[scope_col].astype('string').str.strip().str.lower().isin(set(allow))].copy()
        if df.empty:
            result = {"status": "empty_after_scope", "cutoff": cutoff, "division": division, "scope_column": scope_col}
            out = json.dumps(result, indent=2)
            if output:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_text(out, encoding="utf-8")
            else:
                print(out)
            return

    # Deterministic sample
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)

    # Weights
    w = _compute_weights(df, use_qty=use_qty, include_revenue=include_revenue, price_factor=price_factor, cap=cap, half_life_days=half_life, end_ts=pd.to_datetime(cutoff))

    result = {
        "cutoff": cutoff,
        "division": division,
        "scope_column": ('product_goal' if (scope_by_goal and ('product_goal' in df.columns)) else 'product_division') if division else None,
        "policy": {
            "weight_by_quantity": use_qty,
            "include_revenue": include_revenue,
            "price_factor": price_factor,
            "weight_cap": cap,
            "time_decay_half_life_days": half_life,
        },
        "stats": {
            "rows": int(len(df)),
            "quantity_nonzero_rate": float(round(pd.to_numeric(df.get('quantity', 1.0), errors='coerce').fillna(1.0).gt(0).mean(), 6)),
            "als_weight_quantiles": {
                "min": float(np.nanmin(w)),
                "p50": float(np.nanpercentile(w, 50)),
                "p90": float(np.nanpercentile(w, 90)),
                "p99": float(np.nanpercentile(w, 99)),
                "max": float(np.nanmax(w)),
            },
        },
    }

    out = json.dumps(result, indent=2)
    output_path = Path(output) if output else (OUTPUTS_DIR / "validation" / "als_policy" / f"als_policy_{division or 'global'}_{cutoff}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out, encoding="utf-8")
    logger.info("ALS policy audit written: %s", output_path)


if __name__ == "__main__":
    main()

