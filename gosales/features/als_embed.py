from __future__ import annotations

from typing import Tuple, Optional

import pandas as pd
import polars as pl
import numpy as np

from scipy.sparse import coo_matrix, csr_matrix
from threadpoolctl import threadpool_limits


def _build_user_item(df: pd.DataFrame, user_col: str, item_col: str, weight_col: str) -> Tuple[coo_matrix, pd.Index, pd.Index]:
    # Preserve GUIDs by treating user IDs as strings; categorical codes map rows to indices
    users = pd.Categorical(df[user_col].astype('string'))
    items = pd.Categorical(df[item_col].astype('string'))
    mat = coo_matrix(
        (pd.to_numeric(df[weight_col], errors='coerce').astype('float64'), (users.codes, items.codes)),
        shape=(users.categories.size, items.categories.size)
    )
    return mat, pd.Index(users.categories), pd.Index(items.categories)


def _als_with_implicit(mat: coo_matrix, factors: int, reg: float, alpha: float) -> pd.DataFrame | None:
    try:
        import implicit
        # Use fixed random_state for deterministic embeddings across runs
        model = implicit.als.AlternatingLeastSquares(
            factors=factors, regularization=reg, random_state=42
        )
        # Convert to CSR to avoid implicit's COOâ†’CSR conversion warning and cap BLAS threads
        mat_scaled_csr = csr_matrix((mat * alpha).astype('double'))
        with threadpool_limits(1, "blas"):
            model.fit(mat_scaled_csr)
        return pd.DataFrame(model.user_factors)
    except Exception:
        return None


def _svd_fallback(mat: coo_matrix, factors: int) -> pd.DataFrame | None:
    try:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=factors, random_state=42)
        X = svd.fit_transform(csr_matrix(mat))
        return pd.DataFrame(X)
    except Exception:
        return None


def customer_als_embeddings(
    engine,
    cutoff: str,
    factors: int = 16,
    reg: float = 0.1,
    alpha: float = 10.0,
    lookback_months: int | None = 12,
    lag_days: int | None = None,
    weight_by_quantity: Optional[bool] = None,
    division_name: Optional[str] = None,
) -> pl.DataFrame:
    """Compute ALS-style customer embeddings from transaction history.

    Transactions are restricted to those occurring on or before ``cutoff``. If
    ``lookback_months`` is provided, only transactions within that many months
    before the cutoff are used. This allows for a simple time-decay on older
    activity by effectively discarding stale interactions.
    """

    als_columns = [f"als_f{d}" for d in range(factors)]

    def _empty_embeddings() -> pl.DataFrame:
        data: dict[str, pl.Series] = {
            "customer_id": pl.Series(name="customer_id", values=[], dtype=pl.Int64),
        }
        for col in als_columns:
            data[col] = pl.Series(name=col, values=[], dtype=pl.Float64)
        return pl.DataFrame(data)

    # Build interactions from feature period only
    # Prefer line-grain transactions with quantities and GP for positive-feedback weights
    # Load minimal columns needed; attempt to include revenue if present
    try:
        tx = pd.read_sql(
            "SELECT customer_id, order_date, product_sku, product_division, quantity, gross_profit, revenue FROM fact_transactions",
            engine,
        )
    except Exception:
        try:
            tx = pd.read_sql(
                "SELECT customer_id, order_date, product_sku, product_division, quantity, gross_profit FROM fact_transactions",
                engine,
            )
        except Exception:
            tx = pd.read_sql(
                "SELECT customer_id, order_date, product_sku, product_division, quantity FROM fact_transactions",
                engine,
            )
    if tx.empty:
        return _empty_embeddings()
    tx['order_date'] = pd.to_datetime(tx['order_date'], errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff)
    effective_end = cutoff_dt
    if lag_days is not None:
        try:
            lag_days_int = int(lag_days)
            if lag_days_int > 0:
                effective_end = cutoff_dt - pd.Timedelta(days=lag_days_int)
        except Exception:
            effective_end = cutoff_dt
    if lookback_months is not None:
        start_dt = cutoff_dt - pd.DateOffset(months=lookback_months)
        tx = tx[(tx['order_date'] <= effective_end) & (tx['order_date'] >= start_dt)].copy()
    else:
        tx = tx[(tx['order_date'] <= effective_end)].copy()
    if tx.empty:
        return _empty_embeddings()
    full_tx = tx.copy()
    # Determine weighting policy and filters
    try:
        cfg = load_config()
    except Exception:
        cfg = None
    # Determine weighting policy
    if weight_by_quantity is None:
        try:
            weight_by_quantity = bool(getattr(getattr(cfg, 'features', object()), 'als_weight_by_quantity', True)) if cfg else True
        except Exception:
            weight_by_quantity = True
    # Revenue price policy
    try:
        include_revenue = bool(getattr(getattr(cfg, 'features', object()), 'als_weight_include_revenue', True)) if cfg else True
        price_factor = float(getattr(getattr(cfg, 'features', object()), 'als_weight_price_factor', 1.0)) if cfg else 1.0
        weight_cap = getattr(getattr(cfg, 'features', object()), 'als_weight_cap', None) if cfg else None
        weight_cap = float(weight_cap) if weight_cap is not None else None
        division_scoped = bool(getattr(getattr(cfg, 'features', object()), 'als_division_scoped', True)) if cfg else True
        revenue_cap_global = getattr(getattr(cfg, 'features', object()), 'als_revenue_cap_usd', None) if cfg else None
        revenue_cap_by_div = dict(getattr(getattr(cfg, 'features', object()), 'als_revenue_cap_by_division', {})) if cfg else {}
        time_decay_enabled = bool(getattr(getattr(cfg, 'features', object()), 'als_time_decay_enabled', True)) if cfg else True
        half_life_days = int(getattr(getattr(cfg, 'features', object()), 'als_time_half_life_days', 180)) if cfg else 180
    except Exception:
        include_revenue = True
        price_factor = 1.0
        weight_cap = None
        division_scoped = True
        revenue_cap_global = None
        revenue_cap_by_div = {}
        time_decay_enabled = True
        half_life_days = 180

    # Optional division scoping using Goals when available
    if division_scoped and division_name:
        try:
            aliases = getattr(getattr(cfg, 'features', object()), 'division_aliases', {}) if cfg else {}
            allow = [str(division_name).strip().lower()]
            allow += [v for v in (aliases.get(division_name.lower(), []) or [])]
            allow = [str(a).strip().lower() for a in allow]
        except Exception:
            allow = [str(division_name).strip().lower()]
        scope_col = None
        try:
            scope_by_goal = bool(getattr(getattr(cfg, 'features', object()), 'als_scope_by_goal', True)) if cfg else True
        except Exception:
            scope_by_goal = True
        if scope_by_goal and 'product_goal' in tx.columns:
            scope_col = 'product_goal'
        elif 'product_division' in tx.columns:
            scope_col = 'product_division'
        if scope_col is not None:
            norm = tx[scope_col].astype('string').str.strip().str.lower()
            tx = tx[norm.isin(set(allow))].copy()
            if tx.empty:
                return _empty_embeddings()

    # Weights: optionally combine quantity and positive GP to emphasize meaningful purchases
    tx['quantity'] = pd.to_numeric(tx['quantity'], errors='coerce').fillna(1.0)
    if 'gross_profit' in tx.columns:
        gp_pos = pd.to_numeric(tx['gross_profit'], errors='coerce').fillna(0.0)
        gp_pos = gp_pos.clip(lower=0.0)
    else:
        gp_pos = pd.Series([0.0] * len(tx), index=tx.index)
    tx['customer_id'] = tx['customer_id'].astype('string')
    tx['product_sku'] = tx['product_sku'].astype('string')
    # Simple monotone transform to temper extremes
    if weight_by_quantity:
        q_term = np.log1p(1.0 + tx['quantity'])
    else:
        q_term = 0.0
    gp_term = np.log1p(1.0 + gp_pos)
    price_term = 0.0
    if include_revenue and 'revenue' in tx.columns:
        rev = pd.to_numeric(tx['revenue'], errors='coerce').fillna(0.0)
        rev = rev.clip(lower=0.0)
        # Apply revenue caps (global and per-division)
        try:
            if isinstance(revenue_cap_global, (int, float)) and revenue_cap_global is not None:
                rev = np.minimum(rev, float(revenue_cap_global))
            if revenue_cap_by_div and 'product_division' in tx.columns:
                caps = {str(k).strip().lower(): float(v) for k, v in (revenue_cap_by_div or {}).items()}
                div_norm = tx['product_division'].astype('string').str.strip().str.lower()
                cap_series = div_norm.map(caps).astype('float64')
                cap_series = cap_series.where(~cap_series.isna(), other=np.inf)
                rev = np.minimum(rev, cap_series.values)
        except Exception:
            pass
        price_term = price_factor * np.log1p(1.0 + rev)
    tx['als_weight'] = (q_term + gp_term + (price_term if isinstance(price_term, np.ndarray) else 0.0)).astype('float64')
    if weight_cap is not None and weight_cap > 0:
        tx['als_weight'] = np.minimum(tx['als_weight'], float(weight_cap))
    # Time decay multiplier (default on)
    if time_decay_enabled:
        try:
            half_life = max(1, int(half_life_days))
        except Exception:
            half_life = 180
        lam = np.log(2.0) / float(half_life)
        age_days = (effective_end - tx['order_date']).dt.days.clip(lower=0)
        decay = np.exp(-lam * age_days.astype('float64'))
        tx['als_weight'] = (tx['als_weight'] * decay).astype('float64')
    grp = tx.groupby(['customer_id','product_sku'])['als_weight'].sum().rename('weight').reset_index()
    # Division-aware min volume fallback to global interactions
    try:
        min_pairs = int(getattr(getattr(cfg, 'features', object()), 'als_min_scoped_interactions', 1000)) if cfg else 1000
    except Exception:
        min_pairs = 1000
    global _LAST_SCOPED_FALLBACK
    _LAST_SCOPED_FALLBACK = False
    if division_scoped and division_name and grp.shape[0] < max(1, min_pairs):
        _LAST_SCOPED_FALLBACK = True
        _logger.info(
            "ALS division-scoped interactions below threshold (pairs=%s < %s); falling back to global interactions.",
            grp.shape[0], min_pairs,
        )
        tx = full_tx.copy()
        tx['order_date'] = pd.to_datetime(tx['order_date'], errors='coerce')
        tx['quantity'] = pd.to_numeric(tx['quantity'], errors='coerce').fillna(1.0)
        gp_pos2 = pd.to_numeric(tx.get('gross_profit', 0.0), errors='coerce').fillna(0.0).clip(lower=0.0)
        q_term2 = np.log1p(1.0 + tx['quantity']) if weight_by_quantity else 0.0
        gp_term2 = np.log1p(1.0 + gp_pos2)
        price_term2 = 0.0
        if include_revenue and 'revenue' in tx.columns:
            rev2 = pd.to_numeric(tx['revenue'], errors='coerce').fillna(0.0).clip(lower=0.0)
            price_term2 = price_factor * np.log1p(1.0 + rev2)
        w2 = (q_term2 + gp_term2 + (price_term2 if isinstance(price_term2, np.ndarray) else 0.0)).astype('float64')
        if weight_cap is not None and weight_cap > 0:
            w2 = np.minimum(w2, float(weight_cap))
        if time_decay_enabled:
            lam2 = np.log(2.0) / float(max(1, int(half_life_days)))
            age_days2 = (effective_end - tx['order_date']).dt.days.clip(lower=0)
            w2 = (w2 * np.exp(-lam2 * age_days2.astype('float64'))).astype('float64')
        tmp = pd.DataFrame({'customer_id': tx['customer_id'].astype('string'), 'product_sku': tx['product_sku'].astype('string'), 'w': w2})
        grp = tmp.groupby(['customer_id','product_sku'])['w'].sum().rename('weight').reset_index()
    if grp.empty:
        return _empty_embeddings()
    mat, user_index, _ = _build_user_item(grp, 'customer_id', 'product_sku', 'weight')

    # Try implicit ALS first; if unavailable, fallback to TruncatedSVD
    U = _als_with_implicit(mat, factors=factors, reg=reg, alpha=alpha)
    if U is None:
        U = _svd_fallback(mat, factors=factors)
    if U is None or U.empty:
        return _empty_embeddings()
    U.columns = als_columns[: U.shape[1]]
    # user_index categories align to original string IDs
    U['customer_id'] = user_index.astype('string').values
    # Prefer numeric customer_id when safely convertible (keeps tests and downstream contracts happy)
    try:
        tmp = pd.to_numeric(U['customer_id'], errors='coerce')
        if not tmp.isna().any():
            U['customer_id'] = tmp.astype('Int64')
    except Exception:
        pass
    return pl.from_pandas(U)


from gosales.utils.config import load_config
from gosales.utils.logger import get_logger

_logger = get_logger(__name__)

_LAST_SCOPED_FALLBACK: bool = False

def get_last_scoped_fallback() -> bool:
    return _LAST_SCOPED_FALLBACK


