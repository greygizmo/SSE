from __future__ import annotations

from typing import Tuple
import pandas as pd
import polars as pl

from scipy.sparse import coo_matrix


def _build_user_item(df: pd.DataFrame, user_col: str, item_col: str, weight_col: str) -> Tuple[coo_matrix, pd.Index, pd.Index]:
    users = pd.Categorical(df[user_col].astype('int64'))
    items = pd.Categorical(df[item_col].astype('category'))
    mat = coo_matrix((df[weight_col].astype('float64'), (users.codes, items.codes)), shape=(users.categories.size, items.categories.size))
    return mat, pd.Index(users.categories), pd.Index(items.categories)


def customer_als_embeddings(engine, cutoff: str, factors: int = 16, reg: float = 0.1, alpha: float = 10.0) -> pl.DataFrame:
    try:
        import implicit
    except Exception as e:
        # Return empty frame if implicit not available
        return pl.DataFrame()

    # Build interactions from feature period only
    tx = pd.read_sql("SELECT customer_id, order_date, product_sku, quantity FROM fact_transactions", engine)
    tx['order_date'] = pd.to_datetime(tx['order_date'], errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff)
    tx = tx[(tx['order_date'] <= cutoff_dt)].copy()
    # Weights: total quantity (fallback to 1 if missing)
    tx['quantity'] = pd.to_numeric(tx['quantity'], errors='coerce').fillna(1.0)
    grp = tx.groupby(['customer_id','product_sku'])['quantity'].sum().rename('weight').reset_index()
    mat, user_index, item_index = _build_user_item(grp, 'customer_id', 'product_sku', 'weight')

    # Train ALS implicit
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=reg)
    # Apply alpha scaling for confidence as in Hu et al.
    model.fit((mat * alpha).astype('double'))

    # Extract user factors
    user_factors = model.user_factors  # shape (n_users, factors)
    emb = pd.DataFrame(user_factors, columns=[f'als_f{d}' for d in range(factors)])
    emb['customer_id'] = user_index.astype('int64').values
    return pl.from_pandas(emb)


