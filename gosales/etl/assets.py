"""Ingest and feature-engineer Moneyball asset data for model consumption.

The module centralizes how we read curated asset tables, normalize customer and
item identifiers, and aggregate tenure and purchase summaries used in customer
propensity models.  Scripts and pipelines import these helpers whenever they
need consistent asset-derived signals.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta

from gosales.utils.db import get_db_connection, get_curated_connection
from gosales.utils.sql import validate_identifier, ensure_allowed_identifier
from gosales.sql.queries import moneyball_assets_select, items_category_limited_select
from gosales.utils.config import load_config
from gosales.utils.logger import get_logger
from gosales.utils.identifiers import normalize_identifier_series
from gosales.utils.normalize import normalize_division

logger = get_logger(__name__)


def _norm(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _load_sources() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Moneyball Assets and Item Category mapping from the configured Azure SQL views.

    Returns
    -------
    moneyball : pandas.DataFrame
        Canonicalized Moneyball asset rows with normalized text columns.
    items : pandas.DataFrame
        Item taxonomy with `itemid` and `Item_Rollup` and normalized join keys.
    """
    cfg = load_config()
    src = getattr(cfg, 'database', None)
    tables = dict(getattr(src, 'source_tables', {}) or {})
    moneyball_view = tables.get('moneyball_assets', '[dbo].[Moneyball Assets]')
    items_view = tables.get('items_category_limited', '[dbo].[items_category_limited]')
    # Validate identifiers to mitigate injection risk in f-strings
    try:
        allow = set(getattr(getattr(cfg, 'database', object()), 'allowed_identifiers', []) or [])
        if allow:
            ensure_allowed_identifier(str(moneyball_view), allow)
            ensure_allowed_identifier(str(items_view), allow)
        else:
            validate_identifier(str(moneyball_view))
            validate_identifier(str(items_view))
    except Exception as e:
        raise ValueError(f"Invalid view identifier in config.database.source_tables: {e}")

    eng = get_db_connection()

    logger.info("Reading Moneyball Assets…")
    def _read_chunks(sql: str, chunksize: int = 200_000) -> pd.DataFrame:
        try:
            it = pd.read_sql_query(sql, eng, chunksize=chunksize)
            frames = [c for c in it]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)
        except Exception:
            return pd.read_sql(sql, eng)

    mb_sql = moneyball_assets_select(moneyball_view)
    mb = _read_chunks(mb_sql)
    # Normalize
    mb['customer_name'] = mb['customer_name'].astype(str).str.strip()
    mb['customer_name_norm'] = _norm(mb['customer_name'])
    mb['product'] = mb['product'].astype(str).str.strip()
    mb['product_norm'] = _norm(mb['product'])
    mb['purchase_date'] = pd.to_datetime(mb['purchase_date'], errors='coerce')
    mb['expiration_date'] = pd.to_datetime(mb['expiration_date'], errors='coerce')
    mb['qty'] = pd.to_numeric(mb['qty'], errors='coerce').fillna(1.0).astype(float)
    mb['internalid'] = mb['internalid'].astype(str)

    logger.info("Reading items_category_limited...")
    items_sql = items_category_limited_select(items_view)
    items = _read_chunks(items_sql)
    items['itemid'] = items['itemid'].astype(str).str.strip()
    items['itemid_norm'] = _norm(items['itemid'])
    items['internalid'] = items['internalid'].astype(str)
    items['Item_Rollup'] = items['Item_Rollup'].astype(str).str.strip()
    items['name'] = items['name'].astype(str).str.strip()

    return mb, items


def _map_customers_to_ids(mb: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical customer_id to Moneyball rows.

    Primary strategy: use NetSuite internalid where available; fall back to legacy
    name-based joins only for the small slice of rows missing internalid.
    """

    mapped = mb.copy()

    # 1) Preferred mapping: align on NetSuite company name (entityid or ns_companyname)
    try:
        ns_cur = get_curated_connection()
        # Try entityid first; fallback to ns_companyname
        try:
            ns_df = pd.read_sql("SELECT internalid, entityid AS company_name FROM dim_ns_customer", ns_cur)
        except Exception:
            ns_df = pd.read_sql("SELECT internalid, ns_companyname AS company_name FROM dim_ns_customer", ns_cur)
        ns_df = ns_df.dropna(subset=['internalid', 'company_name']).copy()
        ns_df['company_name_norm'] = _norm(ns_df['company_name'])
        ns_map = ns_df.drop_duplicates('company_name_norm').set_index('company_name_norm')['internalid']
        mapped['customer_id'] = mapped['customer_name_norm'].map(ns_map)
    except Exception as exc:
        logger.warning(f"dim_ns_customer lookup failed, falling back to legacy mapping: {exc}")
        mapped['customer_id'] = None

    mapped['customer_id'] = normalize_identifier_series(mapped['customer_id'])

    missing_mask = mapped['customer_id'].isna()
    if not missing_mask.any():
        return mapped

    # 2) Fallback map via dim_customer (or raw sales_log) for rows without matches.
    try:
        cur = get_curated_connection()
        dc = pd.read_sql("SELECT customer_id, customer_name, customer_name_norm FROM dim_customer", cur)
    except Exception:
        try:
            src = get_db_connection()
            sl = pd.read_sql(
                "SELECT [Customer] AS customer_name, [CompanyId] AS customer_id FROM dbo.saleslog",
                src,
            )
            sl['customer_name_norm'] = _norm(sl['customer_name'])
            dc = sl.groupby('customer_name_norm')[['customer_name', 'customer_id']].first().reset_index()
        except Exception as e:
            logger.warning(f"Failed to load customer id map from curated and sales_log: {e}")
            dc = pd.DataFrame(columns=['customer_name_norm', 'customer_id'])

    if not dc.empty:
        dc = dc.dropna(subset=['customer_name_norm', 'customer_id']).copy()
        dc['customer_name_norm'] = _norm(dc['customer_name_norm'])
        id_map = dc.drop_duplicates('customer_name_norm').set_index('customer_name_norm')['customer_id']
        fallback_ids = mapped.loc[missing_mask, 'customer_name_norm'].map(id_map)
        mapped.loc[missing_mask, 'customer_id'] = normalize_identifier_series(fallback_ids)

    return mapped


def compute_effective_purchase_dates(fa: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.Series:
    """Compute effective purchase dates for assets consistent with feature logic.

    Rules:
    - If purchase_date is valid (>= 1996-01-01), use it but clamp to cutoff (min(purchase_date, cutoff)).
    - Else, impute using per‑rollup median tenure based on valid dates; fallback to global median (or 10 years).
    - Always return dates <= cutoff.
    """
    cutoff_dt = pd.to_datetime(cutoff)
    f = fa.copy()
    f['purchase_date'] = pd.to_datetime(f.get('purchase_date'), errors='coerce')
    min_valid = pd.Timestamp('1996-01-01')
    valid_mask = f['purchase_date'].notna() & (f['purchase_date'] >= min_valid)
    if valid_mask.any():
        v = f.loc[valid_mask, ['item_rollup', 'purchase_date']].copy()
        v['tenure_days'] = (cutoff_dt - v['purchase_date']).dt.days
        med_by_rollup = v.groupby('item_rollup')['tenure_days'].median().to_dict()
        global_med = float(v['tenure_days'].median()) if len(v) else 3650.0
    else:
        med_by_rollup = {}
        global_med = 3650.0

    def _eff(row):
        p = row.get('purchase_date')
        if pd.notna(p) and p >= min_valid:
            # Clamp to cutoff in case of future-dated purchase entries
            return min(p, cutoff_dt)
        med = float(med_by_rollup.get(row.get('item_rollup'), global_med))
        return cutoff_dt - pd.Timedelta(days=int(med))

    eff = f.apply(_eff, axis=1)
    # Ensure clamped to cutoff
    eff = eff.where(eff <= cutoff_dt, cutoff_dt)
    return pd.to_datetime(eff)


def build_fact_assets(write: bool = True) -> pd.DataFrame:
    """Build the canonical fact_assets table by joining Moneyball → Item rollups, and map customers to IDs.

    Returns the resulting DataFrame and optionally writes to the curated database.
    """
    mb, items = _load_sources()

    # Join on product → itemid (normalized)
    joined = mb.merge(
        items[['itemid', 'itemid_norm', 'internalid', 'Item_Rollup', 'name']].rename(columns={'Item_Rollup': 'item_rollup'}),
        left_on='product_norm', right_on='itemid_norm', how='left', suffixes=('', '_items')
    )

    coverage = float((~joined['item_rollup'].isna()).mean()) if len(joined) else 0.0
    # Use ASCII arrow to avoid console encoding issues on Windows shells
    logger.info(f"Moneyball->Item rollup mapping coverage: {coverage:.2%} ({joined['item_rollup'].notna().sum()} of {len(joined)})")

    # --- Authoritative internalid-first join with fallback on normalized name ---
    try:
        # Config flag for debug columns (safe default if absent)
        try:
            emit_debug = bool(getattr(getattr(load_config(), 'etl', object()), 'assets', object()).emit_debug_columns)  # type: ignore[attr-defined]
        except Exception:
            emit_debug = False

        items_c = items.copy()
        # Simpler, robust map: first non-null row per internalid
        items_map = (
            items_c.dropna(subset=['internalid'])
            .sort_values(['internalid'])
            .drop_duplicates(subset=['internalid'], keep='first')
            [['internalid', 'Item_Rollup', 'itemid', 'name']]
            .rename(columns={'Item_Rollup': 'item_rollup'})
        )

        joined2 = mb.merge(
            items_map[['internalid', 'item_rollup', 'itemid', 'name']].rename(columns={'itemid': 'itemid_items', 'name': 'name_items'}),
            on='internalid', how='left', suffixes=('', '_items')
        )
        # Guard: ensure item_rollup column exists post-merge to avoid KeyError in coverage computation
        if 'item_rollup' not in joined2.columns:
            joined2['item_rollup'] = np.nan
        joined2['join_method'] = np.where(joined2['item_rollup'].notna(), 'internalid', 'unmatched')
        joined2['internalid_items'] = joined2['internalid']

        unresolved_mask = joined2['item_rollup'].isna()
        if unresolved_mask.any():
            fallback_pool = joined2[unresolved_mask].copy()
            fallback = fallback_pool.merge(
                items[['itemid', 'itemid_norm', 'internalid', 'Item_Rollup', 'name']].rename(columns={'Item_Rollup': 'item_rollup', 'internalid': 'internalid_items', 'name': 'name_items'}),
                left_on='product_norm', right_on='itemid_norm', how='left'
            )
            for col in ['item_rollup', 'itemid_items', 'name_items', 'internalid_items']:
                joined2.loc[unresolved_mask, col] = fallback[col].values
            joined2.loc[unresolved_mask & joined2['item_rollup'].notna(), 'join_method'] = 'name_fallback'

        n2 = int(len(joined2))
        mapped2 = int(joined2['item_rollup'].notna().sum())
        by_internalid2 = int((joined2['join_method'] == 'internalid').sum())
        by_name2 = int((joined2['join_method'] == 'name_fallback').sum())
        cov2 = float(mapped2 / n2) if n2 else 0.0
        logger.info(f"[assets] internalid-first coverage: {cov2:.2%} (mapped={mapped2} of {n2}; internalid={by_internalid2}, name_fallback={by_name2})")

        # Optional QA outputs
        try:
            import os, json
            out_dir = 'gosales/outputs'
            os.makedirs(out_dir, exist_ok=True)
            with open(f"{out_dir}/assets_join_metrics.json", 'w', encoding='utf-8') as f:
                json.dump({
                    'rows_total': n2,
                    'rows_mapped': mapped2,
                    'coverage_pct': round(cov2 * 100, 2),
                    'mapped_by_internalid': by_internalid2,
                    'mapped_by_name_fallback': by_name2,
                }, f, indent=2)
            unmatched = joined2[joined2['item_rollup'].isna()].copy()
            if not unmatched.empty:
                cols = [c for c in ['customer_name', 'product', 'internalid', 'product_norm', 'itemid_items', 'name_items'] if c in unmatched.columns]
                unmatched[cols].head(1000).to_csv(f"{out_dir}/assets_unmatched.csv", index=False)
            try:
                conflict_counts = (
                    items.groupby('internalid')['Item_Rollup']
                    .nunique(dropna=True)
                    .reset_index(name='rollup_nunique')
                )
                conflicts = conflict_counts[conflict_counts['rollup_nunique'] > 1]
                if not conflicts.empty:
                    conflicts.to_csv(f"{out_dir}/assets_join_conflicts.csv", index=False)
            except Exception:
                pass
        except Exception:
            pass

        # Replace original join result with authoritative version
        joined = joined2
        # Align canonical column names expected downstream
        try:
            if 'itemid_items' in joined.columns and 'itemid' not in joined.columns:
                joined['itemid'] = joined['itemid_items']
            if 'name_items' in joined.columns and 'name' not in joined.columns:
                joined['name'] = joined['name_items']
        except Exception:
            pass
        # If not emitting debug, drop temp debug fields so schema stays minimal
        if not emit_debug:
            for col in ['join_method', 'internalid_items', 'itemid_items', 'name_items']:
                if col in joined.columns:
                    try:
                        joined.drop(columns=[col], inplace=True)
                    except Exception:
                        pass
    except Exception as _exc:
        logger.warning(f"Authoritative assets join failed, using name-based fallback only: {_exc}")

    # Map to customer_id via dim_customer
    joined = _map_customers_to_ids(joined)

    # Final schema
    fact = joined[[
        'customer_id', 'customer_name', 'customer_name_norm',
        'product', 'product_norm', 'qty',
        'purchase_date', 'expiration_date',
        'itemid', 'item_rollup', 'name',
        'department', 'category', 'sub_category_a', 'sub_category_b', 'audience',
        'internalid'
    ]].copy()

    # Coerce types
    fact['customer_id'] = normalize_identifier_series(fact['customer_id'])
    fact['qty'] = pd.to_numeric(fact['qty'], errors='coerce').fillna(1.0)

    if write:
        cur = get_curated_connection()
        # Wrap destructive write in a transaction to allow rollback on failure; use batched insert
        with cur.begin() as conn:
            fact.to_sql('fact_assets', conn, if_exists='replace', index=False, method='multi', chunksize=50)
        logger.info(f"Wrote fact_assets with {len(fact):,} rows")
    return fact


def features_at_cutoff(fact: pd.DataFrame, cutoff_date: str | datetime) -> Tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute asset features at a cutoff date using fact_assets.

    Returns
    -------
    per_cust_rollup : DataFrame
        customer_id x item_rollup pivot-ready counts for active assets at cutoff.
    per_cust : DataFrame
        aggregate features per customer (totals, expiring_90d, tenure stats).
    """
    cutoff = pd.to_datetime(cutoff_date).normalize()
    # Active if purchase_date <= cutoff and (expiration_date is null or >= cutoff)
    f = fact.copy()
    f['purchase_date'] = pd.to_datetime(f['purchase_date'], errors='coerce')
    f['expiration_date'] = pd.to_datetime(f['expiration_date'], errors='coerce')
    eff = compute_effective_purchase_dates(f, cutoff)
    # bad flag: invalid original purchase date
    min_valid = pd.Timestamp('1996-01-01')
    valid_mask = f['purchase_date'].notna() & (f['purchase_date'] >= min_valid)
    f['purchase_effective'] = eff
    f['bad_purchase_date_flag'] = (~valid_mask).astype('int8')

    active = f[(f['purchase_effective'] <= cutoff) & (f['expiration_date'].isna() | (f['expiration_date'] >= cutoff))].copy()

    # Per-rollup counts (active)
    roll = (
        active.dropna(subset=['item_rollup'])
        .groupby(['customer_id', 'item_rollup'])['qty']
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    # Aggregate per customer
    # Guard near-cutoff expirations to reduce shift sensitivity
    try:
        from gosales.utils.config import load_config as _load_cfg
        _cfg = _load_cfg()
        _guard_days = int(getattr(getattr(_cfg, 'features', object()), 'expiring_guard_days', 0))
    except Exception:
        _guard_days = 0
    _start_guard = cutoff + pd.Timedelta(days=int(_guard_days))
    exp_90 = f[(f['expiration_date'].notna()) & (f['expiration_date'] > _start_guard) & (f['expiration_date'] <= cutoff + pd.Timedelta(days=90))]
    per = active.groupby('customer_id')['qty'].sum().rename('assets_active_total').reset_index()
    # Ever-owned assets: any assets purchased on/before cutoff regardless of current status
    try:
        ever_mask = f['purchase_effective'] <= cutoff
        ever = f.loc[ever_mask].groupby('customer_id')['qty'].sum().rename('assets_ever_total').reset_index()
        per = per.merge(ever, on='customer_id', how='outer')
        per['assets_ever_total'] = pd.to_numeric(per['assets_ever_total'], errors='coerce').fillna(0.0)
    except Exception:
        per['assets_ever_total'] = 0.0
    per = per.merge(exp_90.groupby('customer_id')['qty'].sum().rename('assets_expiring_90d').reset_index(), on='customer_id', how='left')
    per['assets_expiring_90d'] = per['assets_expiring_90d'].fillna(0.0)
    # Additional totals for 30/60d and shares vs active
    for days in (30, 60):
        exp = f[(f['expiration_date'].notna()) & (f['expiration_date'] > _start_guard) & (f['expiration_date'] <= cutoff + pd.Timedelta(days=days))]
        col = f'assets_expiring_{days}d'
        per = per.merge(exp.groupby('customer_id')['qty'].sum().rename(col).reset_index(), on='customer_id', how='left')
        per[col] = per[col].fillna(0.0)
    for days in (30, 60, 90):
        per[f'assets_expiring_{days}d_share'] = per[f'assets_expiring_{days}d'] / per['assets_active_total'].replace(0.0, np.nan)
        per[f'assets_expiring_{days}d_share'] = per[f'assets_expiring_{days}d_share'].fillna(0.0)

    # Tenure: days since earliest effective purchase of any asset
    first_purchase_eff = f.groupby('customer_id')['purchase_effective'].min().rename('first_asset_effective').reset_index()
    first_purchase_eff['assets_tenure_days'] = (cutoff - first_purchase_eff['first_asset_effective']).dt.days
    per = per.merge(first_purchase_eff[['customer_id', 'assets_tenure_days']], on='customer_id', how='left')
    # Quality flags: share of bad purchase dates among active assets
    try:
        bad_share = active.groupby('customer_id')['bad_purchase_date_flag'].mean().rename('assets_bad_purchase_share').reset_index()
        per = per.merge(bad_share, on='customer_id', how='left')
    except Exception:
        per['assets_bad_purchase_share'] = 0.0
    per['assets_tenure_days'] = per['assets_tenure_days'].fillna(0).astype(int)

    # Per-rollup expiring windows (30/60/90 days)
    extra_frames: dict[str, pd.DataFrame] = {}
    for days in (30, 60, 90):
        exp = f[(f['expiration_date'].notna()) & (f['expiration_date'] > _start_guard) & (f['expiration_date'] <= cutoff + pd.Timedelta(days=days))]
        if not exp.empty:
            piv = (
                exp.dropna(subset=['item_rollup'])
                .groupby(['customer_id', 'item_rollup'])['qty']
                .sum()
                .unstack(fill_value=0.0)
                .reset_index()
            )
        else:
            piv = pd.DataFrame({'customer_id': per['customer_id'].astype(str).unique()})
        # Prefix columns later in feature engine; mark with a name
        extra_frames[f'expiring_{days}d'] = piv

    # Per-rollup OnSubs/OffSubs counts as of cutoff (approx proxy using expiration)
    on_mask = (f['expiration_date'].isna()) | (f['expiration_date'] >= cutoff)
    off_mask = f['expiration_date'].notna() & (f['expiration_date'] < cutoff)
    for key, mask in (('on_subs', on_mask), ('off_subs', off_mask)):
        sub = f[mask]
        if not sub.empty:
            piv = (
                sub.dropna(subset=['item_rollup'])
                .groupby(['customer_id', 'item_rollup'])['qty']
                .sum()
                .unstack(fill_value=0.0)
                .reset_index()
            )
        else:
            piv = pd.DataFrame({'customer_id': per['customer_id'].astype(str).unique()})
        extra_frames[key] = piv

    # Totals and shares for subs
    try:
        totals = (
            f.assign(on_sub=on_mask.astype(int), off_sub=off_mask.astype(int))
            .groupby('customer_id')
            .agg(on=('on_sub', 'sum'), off=('off_sub', 'sum'))
            .reset_index()
        )
        totals.rename(columns={'on': 'assets_on_subs_total', 'off': 'assets_off_subs_total'}, inplace=True)
        totals['assets_subs_share_total'] = totals['assets_on_subs_total'] / (totals['assets_on_subs_total'] + totals['assets_off_subs_total']).replace(0, np.nan)
        totals['assets_subs_share_total'] = totals['assets_subs_share_total'].fillna(0.0)
        per = per.merge(totals, on='customer_id', how='left')
        for c in ['assets_on_subs_total', 'assets_off_subs_total', 'assets_subs_share_total']:
            per[c] = per[c].fillna(0.0)
    except Exception:
        pass

    # Last expiration date (global, any rollup) for reinclusion policies and days since last expiration
    try:
        last_exp = (
            f[(f['expiration_date'].notna()) & (f['expiration_date'] < cutoff)]
            .groupby('customer_id')['expiration_date']
            .max()
            .rename('assets_last_expiration_date')
            .reset_index()
        )
        per = per.merge(last_exp, on='customer_id', how='left')
        if 'assets_last_expiration_date' in per.columns:
            per['assets_days_since_last_expiration'] = (cutoff - per['assets_last_expiration_date']).dt.days
            per['assets_days_since_last_expiration'] = per['assets_days_since_last_expiration'].fillna(1e9).astype(int)
        else:
            per['assets_days_since_last_expiration'] = 1e9
    except Exception:
        per['assets_days_since_last_expiration'] = 1e9

    # Per-division days since last expiration: map item_rollup -> division and aggregate
    try:
        import re
        def _norm_key(s: str) -> str:
            s = str(s or '').lower()
            s = re.sub(r"[^0-9a-z]+", "_", s)
            s = re.sub(r"_+", "_", s).strip('_')
            return s
        # Build rollup->division map using heuristics (lightweight copy of engine logic)
        def _heuristic_div(roll_norm: str) -> str | None:
            r = roll_norm
            if any(x in r for x in ("swx", "solidworks", "sw_core", "swx_core")):
                return normalize_division("Solidworks")
            if any(x in r for x in ("epdm", "pdm", "cad_editor")):
                return normalize_division("PDM")
            if "simulation" in r or "sim" in r:
                return normalize_division("Simulation")
            if "training" in r:
                return normalize_division("Training")
            if "services" in r or "service" in r:
                return normalize_division("Services")
            if "success" in r and "plan" in r:
                return normalize_division("Success Plan")
            if r.startswith("scan") or "scann" in r:
                return normalize_division("Scanning")
            if "electrical" in r or "schematic" in r:
                return normalize_division("SW Electrical")
            if "camworks" in r or r.startswith("cam"):
                return normalize_division("CAMWorks")
            # Hardware/Printers and ecosystem cues
            if any(x in r for x in ("fdm", "saf", "sla", "p3", "polyjet", "metals", "formlabs", "printer", "consumable", "spare", "repair", "am_", "3dp", "post_processing", "am_software")):
                return normalize_division("Hardware")
            return None

        df_exp = f[(f['expiration_date'].notna()) & (f['expiration_date'] < cutoff)].copy()
        if not df_exp.empty and 'item_rollup' in df_exp.columns:
            df_exp['roll_norm'] = df_exp['item_rollup'].astype(str).map(_norm_key)
            df_exp['div_norm'] = df_exp['roll_norm'].map(_heuristic_div)
            df_exp = df_exp.dropna(subset=['div_norm'])
            if not df_exp.empty:
                last_by_div = (
                    df_exp.groupby(['customer_id','div_norm'])['expiration_date']
                    .max()
                    .reset_index()
                )
                last_by_div['days_since'] = (cutoff - last_by_div['expiration_date']).dt.days
                # Pivot to wide per-division days since
                piv = last_by_div.pivot(index='customer_id', columns='div_norm', values='days_since').reset_index()
                # Coerce names to 'assets_days_since_last_expiration_div_<division>'
                piv.columns = [
                    'customer_id' if c == 'customer_id' else f"{str(c)}" for c in piv.columns
                ]
                # Rename columns to match merge-prefixing in feature engine (assets_<key>_...)
                # We'll store this frame into extra_frames using key 'days_since_last_expiration_div'
                # The feature engine will prefix with 'assets_days_since_last_expiration_div_'
                # Ensure customer_id is string
                piv['customer_id'] = piv['customer_id'].astype(str)
                extra_frames['days_since_last_expiration_div'] = piv
    except Exception:
        pass

    # Per-rollup subscription share = on_subs / (on_subs + off_subs)
    try:
        on_df = extra_frames.get('on_subs')
        off_df = extra_frames.get('off_subs')
        if on_df is not None and off_df is not None:
            on_df = on_df.copy()
            off_df = off_df.copy()
            # Ensure join keys exist
            if 'customer_id' not in on_df.columns:
                on_df['customer_id'] = per['customer_id'].astype(str)
            if 'customer_id' not in off_df.columns:
                off_df['customer_id'] = per['customer_id'].astype(str)
            # Outer join to align rollup columns
            merged = on_df.merge(off_df, on='customer_id', how='outer', suffixes=('_on', '_off')).fillna(0.0)
            share_cols: dict[str, pd.Series] = {}
            for col in merged.columns:
                if col == 'customer_id' or col.endswith('_off'):
                    continue
                if col.endswith('_on'):
                    base = col[:-3]
                    on_val = pd.to_numeric(merged[col], errors='coerce').fillna(0.0)
                    off_val = pd.to_numeric(merged.get(base + '_off', 0.0), errors='coerce').fillna(0.0)
                    denom = (on_val + off_val).replace(0.0, np.nan)
                    share = (on_val / denom).fillna(0.0)
                    share_cols[base] = share
            if share_cols:
                out = pd.DataFrame({'customer_id': merged['customer_id'].astype(str)})
                for k, v in share_cols.items():
                    # base column heading is the rollup name; engine will prefix
                    out[k] = v
                extra_frames['subs_share'] = out
    except Exception:
        pass

    # Per-rollup composition shares across rollups per customer for on/off
    try:
        def _composition(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame({'customer_id': per['customer_id'].astype(str).unique()})
            comp = df.copy()
            # Sum across rollup columns (exclude key)
            roll_cols = [c for c in comp.columns if c != 'customer_id']
            # numeric
            for c in roll_cols:
                comp[c] = pd.to_numeric(comp[c], errors='coerce').fillna(0.0)
            denom = comp[roll_cols].sum(axis=1).replace(0.0, np.nan)
            for c in roll_cols:
                comp[c] = (comp[c] / denom).fillna(0.0)
            return comp

        if extra_frames.get('on_subs') is not None:
            extra_frames['on_subs_share'] = _composition(extra_frames['on_subs'])
        if extra_frames.get('off_subs') is not None:
            extra_frames['off_subs_share'] = _composition(extra_frames['off_subs'])
    except Exception:
        pass

    return roll, per, extra_frames


def main():
    # Build fact_assets and write to curated DB
    fact = build_fact_assets(write=True)
    # Optionally emit a quick coverage report
    try:
        cov = fact['item_rollup'].notna().mean()
        unmapped = fact[fact['item_rollup'].isna()].groupby('product')['qty'].sum().sort_values(ascending=False).head(50)
        logger.info(f"Item rollup coverage: {cov:.2%}")
        out = unmapped.reset_index().rename(columns={'qty': 'qty_total'})
        out.to_csv('gosales/outputs/unmapped_items.csv', index=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
