from __future__ import annotations

from typing import Tuple
import pandas as pd
import polars as pl

from scipy.sparse import coo_matrix, csr_matrix


def _build_user_item(df: pd.DataFrame, user_col: str, item_col: str, weight_col: str) -> Tuple[coo_matrix, pd.Index, pd.Index]:
    users = pd.Categorical(pd.to_numeric(df[user_col], errors='coerce').astype('Int64').astype('int64'))
    items = pd.Categorical(df[item_col].astype('string'))
    mat = coo_matrix(
        (pd.to_numeric(df[weight_col], errors='coerce').astype('float64'), (users.codes, items.codes)),
        shape=(users.categories.size, items.categories.size)
    )
    return mat, pd.Index(users.categories), pd.Index(items.categories)


def _als_with_implicit(mat: coo_matrix, factors: int, reg: float, alpha: float) -> pd.DataFrame | None:
    try:
        import implicit
        model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=reg)
        model.fit((mat * alpha).astype('double'))
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
) -> pl.DataFrame:
    """Compute ALS-style customer embeddings from transaction history.

    Transactions are restricted to those occurring on or before ``cutoff``. If
    ``lookback_months`` is provided, only transactions within that many months
    before the cutoff are used. This allows for a simple time-decay on older
    activity by effectively discarding stale interactions.
    """

    # Build interactions from feature period only
    tx = pd.read_sql(
        "SELECT customer_id, order_date, product_sku, quantity FROM fact_transactions",
        engine,
    )
    if tx.empty:
        return pl.DataFrame()
    tx['order_date'] = pd.to_datetime(tx['order_date'], errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff)
    if lookback_months is not None:
        start_dt = cutoff_dt - pd.DateOffset(months=lookback_months)
        tx = tx[(tx['order_date'] <= cutoff_dt) & (tx['order_date'] >= start_dt)].copy()
    else:
        tx = tx[(tx['order_date'] <= cutoff_dt)].copy()
    if tx.empty:
        return pl.DataFrame()
    # Weights: total quantity (fallback to 1 if missing)
    tx['quantity'] = pd.to_numeric(tx['quantity'], errors='coerce').fillna(1.0)
    grp = tx.groupby(['customer_id','product_sku'])['quantity'].sum().rename('weight').reset_index()
    if grp.empty:
        return pl.DataFrame()
    mat, user_index, _ = _build_user_item(grp, 'customer_id', 'product_sku', 'weight')

    # Try implicit ALS first; if unavailable, fallback to TruncatedSVD
    U = _als_with_implicit(mat, factors=factors, reg=reg, alpha=alpha)
    if U is None:
        U = _svd_fallback(mat, factors=factors)
    if U is None or U.empty:
        return pl.DataFrame()
    U.columns = [f'als_f{d}' for d in range(U.shape[1])]
    U['customer_id'] = user_index.astype('int64').values
    return pl.from_pandas(U)


