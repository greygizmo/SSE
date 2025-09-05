from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta

from gosales.utils.db import get_db_connection, get_curated_connection
from gosales.utils.config import load_config
from gosales.utils.logger import get_logger

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

    eng = get_db_connection()

    logger.info("Reading Moneyball Assets…")
    mb = pd.read_sql(
        f"SELECT [Customer Name] AS customer_name, [Product] AS product, [Purchase Date] AS purchase_date, "
        f"[Expiration Date] AS expiration_date, [QTY] AS qty, [internalid] AS internalid, "
        f"[Department] AS department, [Category] AS category, [Sub Category A] AS sub_category_a, [Sub Category B] AS sub_category_b, "
        f"[Audience] AS audience, [Expired] AS expired, [Sales Rep] AS sales_rep, [CAM Sales Rep] AS cam_sales_rep, [AM Sales Rep] AS am_sales_rep, [Simulation Sales Rep] AS sim_sales_rep "
        f"FROM {moneyball_view}",
        eng,
    )
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
    items = pd.read_sql(
        f"SELECT itemid, internalid, Item_Rollup, name, department_name, Category, Sub_Category_A, Sub_Category_B, Audience "
        f"FROM {items_view}",
        eng,
    )
    items['itemid'] = items['itemid'].astype(str).str.strip()
    items['itemid_norm'] = _norm(items['itemid'])
    items['internalid'] = items['internalid'].astype(str)
    items['Item_Rollup'] = items['Item_Rollup'].astype(str).str.strip()
    items['name'] = items['name'].astype(str).str.strip()

    return mb, items


def _map_customers_to_ids(mb: pd.DataFrame) -> pd.DataFrame:
    """Map Moneyball customer names -> canonical customer_id via dim_customer or sales_log.

    Strategy: exact string match on normalized name; do NOT use numeric prefixes.
    """
    try:
        cur = get_curated_connection()
        dc = pd.read_sql("SELECT customer_id, customer_name, customer_name_norm FROM dim_customer", cur)
    except Exception:
        # Fallback: build map from source sales_log
        try:
            src = get_db_connection()
            sl = pd.read_sql("SELECT [Customer] AS customer_name, [CompanyId] AS customer_id FROM dbo.saleslog", src)
            sl['customer_name_norm'] = _norm(sl['customer_name'])
            dc = sl.groupby('customer_name_norm')[['customer_name', 'customer_id']].first().reset_index()
        except Exception as e:
            logger.warning(f"Failed to load customer id map from curated and sales_log: {e}")
            dc = pd.DataFrame(columns=['customer_name_norm', 'customer_id', 'customer_name'])

    # Deduplicate by norm key
    if not dc.empty:
        dc = dc.dropna(subset=['customer_name_norm']).copy()
        dc['customer_name_norm'] = _norm(dc['customer_name_norm'])
        dc = dc.groupby('customer_name_norm')[['customer_id', 'customer_name']].first().reset_index()
        mapped = mb.merge(dc, on='customer_name_norm', how='left', suffixes=('', '_dim'))
    else:
        mapped = mb.copy()
        mapped['customer_id'] = np.nan
    return mapped


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
    fact['customer_id'] = fact['customer_id'].astype(str)
    fact['qty'] = pd.to_numeric(fact['qty'], errors='coerce').fillna(1.0)

    if write:
        cur = get_curated_connection()
        fact.to_sql('fact_assets', cur, if_exists='replace', index=False)
        logger.info(f"Wrote fact_assets with {len(fact):,} rows")
    return fact


def features_at_cutoff(fact: pd.DataFrame, cutoff_date: str | datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    # Handle bad historical purchase dates (e.g., 1900/1910/1990) by imputing tenure
    min_valid = pd.Timestamp('1996-01-01')
    valid_mask = f['purchase_date'].notna() & (f['purchase_date'] >= min_valid)
    # Compute median tenure by rollup using valid dates
    if valid_mask.any():
        valid = f.loc[valid_mask, ['item_rollup', 'purchase_date']].copy()
        valid['tenure_days'] = (cutoff - valid['purchase_date']).dt.days
        med_by_rollup = valid.groupby('item_rollup')['tenure_days'].median().to_dict()
        global_med = float(valid['tenure_days'].median()) if len(valid) else 3650.0
    else:
        med_by_rollup = {}
        global_med = 3650.0

    # Build effective purchase date using imputed tenure for invalid rows
    def _impute_effective_purchase(row):
        p = row['purchase_date']
        if pd.notna(p) and p >= min_valid:
            return p
        r = row.get('item_rollup')
        med = float(med_by_rollup.get(r, global_med))
        return (cutoff - pd.Timedelta(days=int(med)))

    f['purchase_effective'] = f.apply(_impute_effective_purchase, axis=1)
    f['bad_purchase_date_flag'] = (~valid_mask).astype('int8')

    active = f[(f['purchase_effective'] <= cutoff) & (f['expiration_date'].isna() | (f['expiration_date'] >= cutoff))].copy()

    # Per-rollup counts
    roll = (
        active.dropna(subset=['item_rollup'])
        .groupby(['customer_id', 'item_rollup'])['qty']
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    # Aggregate per customer
    exp_90 = f[(f['expiration_date'].notna()) & (f['expiration_date'] > cutoff) & (f['expiration_date'] <= cutoff + pd.Timedelta(days=90))]
    per = active.groupby('customer_id')['qty'].sum().rename('assets_active_total').reset_index()
    per = per.merge(exp_90.groupby('customer_id')['qty'].sum().rename('assets_expiring_90d').reset_index(), on='customer_id', how='left')
    per['assets_expiring_90d'] = per['assets_expiring_90d'].fillna(0.0)

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

    return roll, per


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
