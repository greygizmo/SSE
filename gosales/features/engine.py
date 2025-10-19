"""Core feature-engineering routines shared across training pipelines.

The engine builds time-aware customer matrices, joining transactional, asset,
and embedding signals into a single frame suitable for binary classification.
It is the authoritative implementation of feature logic consumed by CLI
wrappers and notebooks alike.
"""

import polars as pl
import pandas as pd
import numpy as np
from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils import config as cfg
from gosales.utils.paths import OUTPUTS_DIR
from gosales.features.als_embed import customer_als_embeddings
from gosales.etl.sku_map import division_set, get_model_targets, get_sku_mapping
from gosales.utils.normalize import normalize_division
from gosales.etl.assets import build_fact_assets  # for on-demand ensure
from gosales.sql.queries import select_all

logger = get_logger(__name__)


def _safe_read_sql_query(engine, query: str, params: dict | None = None) -> pd.DataFrame:
    """Read SQL with tolerant error handling."""

    try:
        return pd.read_sql_query(query, engine, params=params)
    except Exception as exc:  # pragma: no cover - diagnostic path
        try:
            head = query.strip().splitlines()[0]
        except Exception:
            head = query
        logger.debug(
            "SQL read failed",
            extra={"query_head": head, "error": str(exc)},
        )
        return pd.DataFrame()


def _coerce_flag_series(series: pd.Series, truthy: set[str] | None = None) -> pd.Series:
    """Convert assorted textual/numeric flag encodings into Int8 indicators."""

    if series is None:
        return pd.Series(dtype="Int8")
    truthy_values = {"1", "true", "t", "yes", "y", "on"}
    if truthy:
        truthy_values |= {str(val).strip().lower() for val in truthy}
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return (numeric.fillna(0).astype(float) > 0).astype("Int8")
    series_str = series.astype(str).str.strip().str.lower()
    return series_str.isin(truthy_values).astype("Int8")


def _load_line_item_metadata(engine, cutoff_date: str | None = None) -> pd.DataFrame:
    """Return lightweight line-item metadata for branch/rep and flag derivations."""

    params = None
    where_clause = ""
    if cutoff_date:
        params = {"cutoff": cutoff_date}
        where_clause = "WHERE Rec_Date <= :cutoff"
    query = f"""
        SELECT
            CompanyId AS customer_id,
            Rec_Date AS order_date,
            Branch AS branch,
            Rep AS rep,
            New AS new_flag,
            New_Business AS new_business_flag,
            Referral_NS_Field AS referral_ns_field
        FROM fact_sales_line
        {where_clause}
    """
    df = _safe_read_sql_query(engine, query, params)
    if df.empty:
        return df
    df["customer_id"] = df["customer_id"].astype(str)
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df


def _prepare_branch_source(line_meta: pd.DataFrame) -> pd.DataFrame:
    """Extract branch/rep columns from line metadata, returning empty on missing data."""

    if line_meta.empty:
        return pd.DataFrame()
    cols = [c for c in ("customer_id", "order_date", "branch", "rep") if c in line_meta.columns]
    if not cols:
        return pd.DataFrame()
    sl = line_meta[cols].copy()
    sl = sl.dropna(subset=["customer_id"])
    if sl.empty:
        return sl
    branch_available = "branch" in sl.columns and sl["branch"].notna().any()
    rep_available = "rep" in sl.columns and sl["rep"].notna().any()
    if not branch_available and not rep_available:
        return pd.DataFrame()
    return sl


def _load_branch_rep_curated(engine, cutoff_date: str | None = None) -> pd.DataFrame:
    """Fallback branch/rep loader using curated raw extracts."""

    params = {"cutoff": cutoff_date} if cutoff_date else None
    where_clause = " WHERE order_date <= :cutoff" if cutoff_date else ""
    df = _safe_read_sql_query(
        engine,
        f"SELECT customer_id, order_date, branch, rep FROM fact_sales_log_raw{where_clause}",
        params,
    )
    if df.empty:
        return pd.DataFrame()
    df["customer_id"] = df["customer_id"].astype(str)
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df


def _compute_flags_from_line_meta(line_meta: pd.DataFrame, cutoff_date: str | None = None) -> pd.DataFrame:
    """Build ever_acr / ever_new_customer flags from line-item metadata."""

    if line_meta.empty:
        return pd.DataFrame()
    cols = [c for c in ("customer_id", "order_date", "new_flag", "new_business_flag", "referral_ns_field") if c in line_meta.columns]
    if not cols:
        return pd.DataFrame()

    flags = line_meta[cols].copy()
    flags["customer_id"] = flags["customer_id"].astype(str)
    if cutoff_date and "order_date" in flags.columns:
        cutoff_ts = pd.to_datetime(cutoff_date, errors="coerce")
        if cutoff_ts is not pd.NaT:
            flags = flags[flags["order_date"] <= cutoff_ts]
    if flags.empty:
        return pd.DataFrame()

    new_candidates: list[pd.Series] = []
    if "new_flag" in flags.columns:
        new_candidates.append(_coerce_flag_series(flags["new_flag"], {"new"}))
    if "new_business_flag" in flags.columns:
        new_candidates.append(_coerce_flag_series(flags["new_business_flag"], {"new", "new customer", "new business"}))
    if new_candidates:
        new_stack = pd.concat(new_candidates, axis=1).fillna(0)
        new_combined = (new_stack.astype(float) > 0).any(axis=1).astype("Int8")
    else:
        new_combined = pd.Series(0, index=flags.index, dtype="Int8")

    acr_series = pd.Series(0, index=flags.index, dtype="Int8")
    if "referral_ns_field" in flags.columns:
        referral = flags["referral_ns_field"].astype(str).str.strip()
        mask = referral.str.contains(r"\bACR\b", case=False, regex=True, na=False) | referral.str.contains("account coverage review", case=False, na=False)
        acr_series = mask.astype("Int8")

    derived = pd.DataFrame(
        {
            "customer_id": flags["customer_id"],
            "ever_new_customer": new_combined,
            "ever_acr": acr_series,
        }
    )
    return derived.groupby("customer_id", as_index=False).max()


def _compute_line_window_features(
    line_df: pd.DataFrame,
    *,
    cutoff_dt: pd.Timestamp,
    windows: list[int],
    target_division: str,
    use_usd: bool,
    enable_canonical: bool,
    enable_margin: bool,
    enable_currency: bool,
    enable_diversity: bool = False,
    enable_returns: bool = False,
    enable_order_comp: bool = False,
    enable_trend: bool = False,
) -> list[pd.DataFrame]:
    """Return per-window customer aggregates derived from fact_sales_line."""

    if line_df is None or line_df.empty:
        return []
    if not any([enable_canonical, enable_margin, enable_currency, enable_diversity, enable_returns, enable_order_comp, enable_trend]):
        return []

    df = line_df.copy()
    df["order_date"] = pd.to_datetime(df.get("order_date"), errors="coerce")
    df = df.dropna(subset=["order_date"])
    if df.empty:
        return []

    df["customer_id"] = df.get("customer_id").astype(str)
    df["division_canonical"] = (
        df.get("division_canonical")
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df["item_rollup"] = (
        df.get("item_rollup", pd.Series([None] * len(df)))
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df["sales_order"] = df.get("Sales_Order", pd.Series([None] * len(df))).astype(str).str.strip()

    revenue_col = None
    if use_usd and "Revenue_usd" in df.columns:
        revenue_col = "Revenue_usd"
    if revenue_col is None and "Revenue" in df.columns:
        revenue_col = "Revenue"
    df["revenue_value"] = pd.to_numeric(df.get(revenue_col), errors="coerce").fillna(0.0) if revenue_col else 0.0

    gp_col = None
    if use_usd and "GP_usd" in df.columns:
        gp_col = "GP_usd"
    if gp_col is None and "GP" in df.columns:
        gp_col = "GP"
    df["gp_value"] = pd.to_numeric(df.get(gp_col), errors="coerce").fillna(0.0) if gp_col else 0.0

    term_gp_col = None
    if use_usd and "Term_GP_usd" in df.columns:
        term_gp_col = "Term_GP_usd"
    if term_gp_col is None and "Term_GP" in df.columns:
        term_gp_col = "Term_GP"
    df["term_gp_value"] = pd.to_numeric(df.get(term_gp_col), errors="coerce").fillna(0.0) if term_gp_col else 0.0

    df["currency_code"] = df.get("SalesOrder_Currency", pd.Series(["UNKNOWN"] * len(df))).astype(str).str.strip().str.upper()
    df["is_cad"] = (df["currency_code"] == "CAD").astype(float)
    df["has_fx"] = pd.to_numeric(df.get("USD_CAD_Conversion_rate"), errors="coerce").notna().astype(float)

    if "is_return_line" in df.columns:
        df["is_return_line"] = pd.to_numeric(df["is_return_line"], errors="coerce").fillna(0.0)
    else:
        df["is_return_line"] = (df["revenue_value"] < 0).astype(float)

    target_canonical = normalize_division(target_division).upper() if target_division else ""

    line_frames: list[pd.DataFrame] = []

    for window in windows:
        if not isinstance(window, (int, float)):
            continue
        try:
            window = int(window)
        except Exception:
            continue
        start = cutoff_dt - pd.DateOffset(months=window)
        subset = df[(df["order_date"] > start) & (df["order_date"] <= cutoff_dt)]
        if subset.empty:
            continue

        grouped = subset.groupby("customer_id", dropna=False)
        total_revenue = grouped["revenue_value"].sum()
        total_gp = grouped["gp_value"].sum()
        total_term_gp = grouped["term_gp_value"].sum()
        line_counts = grouped.size()

        aggregates = {
            "customer_id": total_revenue.index.astype(str),
        }

        if enable_margin:
            col_total_rev = f"margin__total_revenue_usd__{window}m"
            col_total_gp = f"margin__total_gp_usd__{window}m"
            col_total_term = f"margin__total_term_gp_usd__{window}m"
            aggregates[col_total_rev] = total_revenue.values
            aggregates[col_total_gp] = total_gp.values
            aggregates[col_total_term] = total_term_gp.values
            denom = total_revenue.replace(0.0, np.nan)
            gp_rate = (total_gp / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            term_rate = (total_term_gp / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            aggregates[f"margin__gp_rate__{window}m"] = gp_rate.values
            aggregates[f"margin__term_gp_rate__{window}m"] = term_rate.values

        if enable_currency:
            cad_revenue = subset[subset["is_cad"] == 1].groupby("customer_id")["revenue_value"].sum()
            fx_share = grouped["has_fx"].mean()
            denom = total_revenue.replace(0.0, np.nan)
            cad_share = (cad_revenue / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            aggregates[f"currency__cad_share__{window}m"] = cad_share.reindex(total_revenue.index, fill_value=0.0).values
            aggregates[f"currency__cad_revenue_usd__{window}m"] = cad_revenue.reindex(total_revenue.index, fill_value=0.0).values
            aggregates[f"currency__fx_applied_share__{window}m"] = fx_share.reindex(total_revenue.index, fill_value=0.0).values

        if enable_canonical:
            canonical = subset.groupby(["customer_id", "division_canonical"], dropna=False)["revenue_value"].sum().reset_index()
            counts = canonical.groupby("customer_id")["division_canonical"].nunique()
            unknown_rev = canonical[canonical["division_canonical"] == "UNKNOWN"].set_index("customer_id")["revenue_value"]
            top_rev = canonical.groupby("customer_id")["revenue_value"].max()
            if target_canonical:
                target_rev = canonical[canonical["division_canonical"] == target_canonical].set_index("customer_id")["revenue_value"]
            else:
                target_rev = pd.Series(dtype=float)
            denom = total_revenue.replace(0.0, np.nan)
            aggregates[f"xdiv__canon_unique__{window}m"] = counts.reindex(total_revenue.index, fill_value=0).astype(float).values
            aggregates[f"xdiv__canon_unknown_share__{window}m"] = (unknown_rev / denom).replace([np.inf, -np.inf], np.nan).reindex(total_revenue.index, fill_value=0.0).fillna(0.0).values
            aggregates[f"xdiv__canon_top1_share__{window}m"] = (top_rev / denom).replace([np.inf, -np.inf], np.nan).reindex(total_revenue.index, fill_value=0.0).fillna(0.0).values
            aggregates[f"xdiv__canon_target_share__{window}m"] = (target_rev / denom).replace([np.inf, -np.inf], np.nan).reindex(total_revenue.index, fill_value=0.0).fillna(0.0).values

        if enable_diversity:
            rollup_group = subset[["customer_id", "item_rollup", "revenue_value"]].copy()
            rollup_group["item_rollup"] = rollup_group["item_rollup"].fillna("unknown").astype(str).str.strip().str.lower()
            agg_roll = rollup_group.groupby(["customer_id", "item_rollup"], dropna=False)["revenue_value"].sum().reset_index()
            roll_counts = agg_roll.groupby("customer_id")["item_rollup"].nunique()
            denom_rev = total_revenue.replace(0.0, np.nan)
            shares = agg_roll.copy()
            shares["share"] = shares["revenue_value"] / shares["customer_id"].map(denom_rev)
            shares["share"] = shares["share"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            hhi = shares.groupby("customer_id")["share"].apply(lambda s: float(np.square(s.values.astype(float)).sum()) if len(s) else 0.0)
            top_rollup_share = shares.groupby("customer_id")["share"].max()
            aggregates[f"diversity__rollup_nunique__{window}m"] = roll_counts.reindex(total_revenue.index, fill_value=0).astype(float).values
            aggregates[f"diversity__rollup_hhi__{window}m"] = hhi.reindex(total_revenue.index, fill_value=0.0).fillna(0.0).values
            aggregates[f"diversity__rollup_top1_share__{window}m"] = top_rollup_share.reindex(total_revenue.index, fill_value=0.0).fillna(0.0).values

        if enable_returns:
            return_mask = subset["is_return_line"] > 0.5
            if return_mask.any():
                return_counts = subset[return_mask].groupby("customer_id")["is_return_line"].count()
                return_revenue = subset[return_mask].groupby("customer_id")["revenue_value"].sum()
                return_gp = subset[return_mask].groupby("customer_id")["gp_value"].sum()
            else:
                return_counts = pd.Series(dtype=float)
                return_revenue = pd.Series(dtype=float)
                return_gp = pd.Series(dtype=float)
            denom_lines = line_counts.replace(0, np.nan)
            denom_rev = total_revenue.replace(0.0, np.nan)
            denom_gp = total_gp.replace(0.0, np.nan)
            line_share = (return_counts / denom_lines).replace([np.inf, -np.inf], np.nan)
            revenue_share = (return_revenue / denom_rev).replace([np.inf, -np.inf], np.nan)
            gp_share = (return_gp / denom_gp).replace([np.inf, -np.inf], np.nan)
            aggregates[f"returns__line_share__{window}m"] = line_share.reindex(total_revenue.index, fill_value=0.0).fillna(0.0).values
            aggregates[f"returns__line_count__{window}m"] = return_counts.reindex(total_revenue.index, fill_value=0.0).values
            aggregates[f"returns__revenue_share__{window}m"] = revenue_share.reindex(total_revenue.index, fill_value=0.0).fillna(0.0).values
            aggregates[f"returns__gp_share__{window}m"] = gp_share.reindex(total_revenue.index, fill_value=0.0).fillna(0.0).values

        if enable_order_comp:
            avg_rev_per_line = (total_revenue / line_counts.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            aggregates[f"order__avg_revenue_per_line_usd__{window}m"] = avg_rev_per_line.reindex(total_revenue.index, fill_value=0.0).values

            orders = subset[["customer_id", "sales_order", "revenue_value"]].copy()
            orders["sales_order"] = orders["sales_order"].fillna("unknown").astype(str)
            order_line_counts = orders.groupby(["customer_id", "sales_order"]).size().rename("order_line_count").reset_index()
            order_revenue = orders.groupby(["customer_id", "sales_order"])["revenue_value"].sum().rename("order_revenue_usd").reset_index()
            order_stats = order_line_counts.merge(order_revenue, on=["customer_id", "sales_order"], how="outer")
            if not order_stats.empty:
                per_customer_order = order_stats.groupby("customer_id").agg(
                    order_lines_per_order_mean=("order_line_count", "mean"),
                    order_lines_per_order_max=("order_line_count", "max"),
                    order_revenue_per_order_mean=("order_revenue_usd", "mean"),
                    order_revenue_per_order_median=("order_revenue_usd", "median"),
                )
                per_customer_order = per_customer_order.rename(
                    columns={
                        "order_lines_per_order_mean": f"order__lines_per_order_mean__{window}m",
                        "order_lines_per_order_max": f"order__lines_per_order_max__{window}m",
                        "order_revenue_per_order_mean": f"order__avg_revenue_per_order_usd__{window}m",
                        "order_revenue_per_order_median": f"order__median_revenue_per_order_usd__{window}m",
                    }
                )
                for col in per_customer_order.columns:
                    aggregates[col] = (
                        per_customer_order[col]
                        .reindex(total_revenue.index, fill_value=0.0)
                        .replace([np.inf, -np.inf], 0.0)
                        .fillna(0.0)
                        .values
                    )

        frame = pd.DataFrame(aggregates)
        numeric_cols = [c for c in frame.columns if c != "customer_id"]
        if numeric_cols:
            frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            line_frames.append(frame)

    return line_frames
def create_feature_matrix(engine, division_name: str, cutoff_date: str = None, prediction_window_months: int = 6, mask_tail_days: int | None = None, label_buffer_days: int | None = None):
    """
    Creates a rich feature matrix for a specific division for ML training with proper time-based splitting.

    This function reads from the clean `fact_transactions` and `dim_customer` tables
    and engineers a wide range of behavioral features, including recency, monetary value,
    customer growth, and ecosystem engagement.

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
        division_name (str): The name of the division to create the feature matrix for (e.g., 'Solidworks').
        cutoff_date (str, optional): Date string (YYYY-MM-DD) to use as feature cutoff. If None, uses all historical data.
        prediction_window_months (int): Number of months after cutoff_date to define the prediction target.

    Returns:
        polars.DataFrame: The feature matrix with a binary target column `bought_in_division`.
    """
    logger.info(f"Creating feature matrix for division: {division_name}...")
    try:
        ds = normalize_division(division_name)
        logger.info("Division string (repr/len): %r / %d", ds, len(ds))
    except Exception:
        pass
    # Canonical division string used for comparisons
    norm_division_name = normalize_division(division_name)
    # Detect if caller passed a target model name (e.g., 'Printers') rather than a raw division
    target_skus = tuple(get_model_targets(norm_division_name))
    use_custom_targets = len(target_skus) > 0
    # Support division aliases (e.g., 'cad' -> ['solidworks','pdm', ...]) from config
    alias_divisions: list[str] = []
    try:
        _cfg = cfg.load_config()
        raw_alias = getattr(getattr(_cfg, 'features', object()), 'division_aliases', {})  # type: ignore[attr-defined]
        if isinstance(raw_alias, dict):
            alias_divisions = [normalize_division(d) for d in (raw_alias.get(norm_division_name, []) or []) if str(d).strip()]
    except Exception:
        alias_divisions = []
    use_alias = (not use_custom_targets) and bool(alias_divisions)
    # Define label filter once for reuse in buyers and feature recency columns
    division_col = (
        pl.col("product_division")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
    )
    if use_custom_targets:
        label_filter = pl.col("product_sku").is_in(list(target_skus))
    elif use_alias:
        label_filter = division_col.is_in(alias_divisions)
    else:
        # Allow goal-based targeting when product_goal exists in transactions
        goal_col = (
            pl.col("product_goal")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.to_lowercase()
        )
        # Use OR filter: prefer goal match if present; otherwise division match
        label_filter = (goal_col == norm_division_name) | (division_col == norm_division_name)
    if cutoff_date:
        logger.info(f"Using cutoff date: {cutoff_date} (features from data <= cutoff)")
        logger.info(f"Target: purchases in {prediction_window_months} months after cutoff")

    # --- 1. Load Base Data ---
    try:
        eligible_ids: set[str] | None = None
        # Use case-insensitive division filter to avoid missing rows due to casing
        division_filter = ", ".join(f"'{normalize_division(d)}'" for d in division_set())
        # Attempt to include product_goal where available; fall back gracefully if missing
        base_cols_with_qty = "customer_id, order_date, product_division, product_goal, product_sku, gross_profit, quantity"
        base_cols_no_qty = "customer_id, order_date, product_division, product_goal, product_sku, gross_profit"

        def _read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
            chunks = pd.read_sql_query(sql, engine, params=params, chunksize=100_000)
            frames = [chunk for chunk in chunks]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)

        if cutoff_date:
            # Try reading with quantity; fallback to without if column missing
            feature_sql_qty = (
                f"SELECT {base_cols_with_qty} FROM fact_transactions "
                f"WHERE order_date <= :cutoff AND LOWER(TRIM(product_division)) IN ({division_filter})"
            )
            feature_sql_noqty = (
                f"SELECT {base_cols_no_qty} FROM fact_transactions "
                f"WHERE order_date <= :cutoff AND LOWER(TRIM(product_division)) IN ({division_filter})"
            )
            try:
                feature_data = _read_sql(feature_sql_qty, {"cutoff": cutoff_date})
            except Exception:
                try:
                    feature_data = _read_sql(feature_sql_noqty, {"cutoff": cutoff_date})
                    if not feature_data.empty and 'quantity' not in feature_data.columns:
                        feature_data['quantity'] = 1.0
                except Exception:
                    # Fallback without product_goal entirely
                    feature_sql_qty2 = feature_sql_qty.replace(", product_goal", "")
                    feature_sql_noqty2 = feature_sql_noqty.replace(", product_goal", "")
                    try:
                        feature_data = _read_sql(feature_sql_qty2, {"cutoff": cutoff_date})
                    except Exception:
                        feature_data = _read_sql(feature_sql_noqty2, {"cutoff": cutoff_date})
                        if not feature_data.empty and 'quantity' not in feature_data.columns:
                            feature_data['quantity'] = 1.0
            feature_data["order_date"] = pd.to_datetime(feature_data["order_date"])
            has_quantity = 'quantity' in feature_data.columns

            from dateutil.relativedelta import relativedelta

            cutoff_dt = pd.to_datetime(cutoff_date)
            prediction_end = (cutoff_dt + relativedelta(months=prediction_window_months)).strftime("%Y-%m-%d")
            pred_sql = (
                "SELECT customer_id, order_date, product_division, product_goal, product_sku FROM fact_transactions "
                "WHERE order_date > :cutoff AND order_date <= :pred_end "
                f"AND LOWER(TRIM(product_division)) IN ({division_filter})"
            )
            # Horizon buffer for labels (optional)
            cutoff_label = cutoff_dt
            try:
                if label_buffer_days is not None and int(label_buffer_days) > 0:
                    from datetime import timedelta as _td
                    cutoff_label = cutoff_dt + _td(days=int(label_buffer_days))
            except Exception:
                cutoff_label = cutoff_dt
            prediction_data = _read_sql(
                pred_sql, {"cutoff": cutoff_label.strftime("%Y-%m-%d"), "pred_end": prediction_end}
            )
            # Optional early prospect exclusion to cut compute
            try:
                cfgmod = cfg.load_config()
                exclude_prospects_early = bool(getattr(getattr(cfgmod, 'population', object()), 'exclude_prospects_in_features', False))
                include_prospects_train = bool(getattr(getattr(cfgmod, 'population', object()), 'include_prospects', False))
                warm_window = int(getattr(getattr(cfgmod, 'population', object()), 'warm_window_months', 12) or 12)
                build_segments = [str(s).strip().lower() for s in (getattr(getattr(cfgmod, 'population', object()), 'build_segments', ["warm", "cold"]))]
                build_segments = [s for s in build_segments if s in ("warm","cold")]
                if not build_segments:
                    build_segments = ["warm", "cold"]
            except Exception:
                exclude_prospects_early = False
                include_prospects_train = False
                warm_window = 12
                build_segments = ["warm", "cold"]
            if exclude_prospects_early and not include_prospects_train:
                try:
                    # Warm ids: transactions in last N months (all divisions), pre-cutoff only
                    start_warm = cutoff_dt - pd.DateOffset(months=warm_window)
                    mask_warm = (feature_data['order_date'] > start_warm) & (feature_data['order_date'] <= cutoff_dt)
                    warm_ids = set(feature_data.loc[mask_warm, 'customer_id'].astype(str).unique().tolist())
                except Exception:
                    warm_ids = set()
                # Cold ids via curated assets at cutoff (if available)
                cold_ids = set()
                try:
                    fact_assets_pd = pd.read_sql(select_all('fact_assets'), engine)
                except Exception:
                    fact_assets_pd = pd.DataFrame()
                if not fact_assets_pd.empty:
                    try:
                        from gosales.etl.assets import features_at_cutoff as _assets_at_cut
                        roll, per_cust, _extras = _assets_at_cut(fact_assets_pd, cutoff_date)
                        # Cold definition: ever-owned assets purchased on/before cutoff regardless of subscription status
                        a_ever = pd.to_numeric(per_cust.get('assets_ever_total', 0), errors='coerce').fillna(0.0)
                        mask_cold = (a_ever > 0)
                        cold_ids = set(per_cust.loc[mask_cold, 'customer_id'].astype(str).unique().tolist())
                    except Exception:
                        cold_ids = set()
                # Eligible roster
                try:
                    ids_tx = set(feature_data['customer_id'].astype(str).unique().tolist())
                except Exception:
                    ids_tx = set()
                try:
                    ids_pred = set(prediction_data['customer_id'].astype(str).unique().tolist())
                except Exception:
                    ids_pred = set()
                try:
                    all_ids = set(ids_tx | ids_pred | cold_ids)
                except Exception:
                    all_ids = ids_tx | ids_pred
                # Cold excludes warm to maintain disjoint warm/cold segmentation
                cold_ids = set(cold_ids) - set(warm_ids)
                # Segment selection
                eligible = set()
                if "warm" in build_segments:
                    eligible |= set(warm_ids)
                if "cold" in build_segments:
                    eligible |= set(cold_ids)
                pre_n = len(all_ids)
                if eligible:
                    # Filter feature & prediction frames in place
                    feature_data = feature_data[feature_data['customer_id'].astype(str).isin(eligible)].reset_index(drop=True)
                    prediction_data = prediction_data[prediction_data['customer_id'].astype(str).isin(eligible)].reset_index(drop=True)
                    eligible_ids = eligible
                post_ids = set(feature_data['customer_id'].astype(str).unique().tolist()) | set(prediction_data['customer_id'].astype(str).unique().tolist())
                post_n = len(post_ids)
                warm_n = len(warm_ids)
                cold_n = len(cold_ids)
                prospects_excluded = max(0, int(pre_n - (warm_n + cold_n)))
                logger.info(
                    "Prospect exclusion (features) enabled: warm_window=%dm, roster %d -> %d (warm=%d, cold=%d, prospects_excluded=%d)",
                    warm_window, pre_n, post_n, warm_n, cold_n, prospects_excluded,
                )
                try:
                    # Emit minimal gating summary artifact
                    out_path = OUTPUTS_DIR / f"population_gating_features_{norm_division_name}_{cutoff_date}.csv"
                    pd.DataFrame([
                        {
                            'division': norm_division_name,
                            'cutoff': cutoff_date,
                            'warm_window_months': int(warm_window),
                            'pre_roster': int(pre_n),
                            'post_roster': int(post_n),
                            'warm': int(warm_n),
                            'cold': int(cold_n),
                            'prospects_excluded': int(prospects_excluded),
                        }
                    ]).to_csv(out_path, index=False)
                    # Emit roster diagnostics (source counts & set relationships)
                    diag_path = OUTPUTS_DIR / f"roster_diagnostics_{norm_division_name}_{cutoff_date}.csv"
                    asset_ids = set(per_cust.loc[mask_cold, 'customer_id'].astype(str).unique().tolist()) if 'per_cust' in locals() else set()
                    warm_only = len(set(warm_ids) - set(cold_ids))
                    cold_only = len(set(cold_ids) - set(warm_ids))
                    both = len(set(warm_ids).intersection(set(cold_ids)))
                    future_only = len(set(ids_pred) - (set(ids_tx) | set(asset_ids)))
                    lapsed_non_asset = len((set(ids_tx) - set(warm_ids)) - set(asset_ids))
                    pd.DataFrame([
                        {
                            'pre_tx_ids': int(len(ids_tx)),
                            'pred_tx_ids': int(len(ids_pred)),
                            'asset_owner_ids': int(len(asset_ids)),
                            'pre_roster': int(pre_n),
                            'post_roster': int(post_n),
                            'warm': int(warm_n),
                            'cold': int(cold_n),
                            'warm_only': int(warm_only),
                            'cold_only': int(cold_only),
                            'both_warm_cold': int(both),
                            'eligible': int(len(eligible)),
                            'future_only_pred_not_pre_or_assets': int(future_only),
                            'lapsed_non_asset_pre_not_warm_not_assets': int(lapsed_non_asset),
                        }
                    ]).to_csv(diag_path, index=False)
                except Exception:
                    pass
            prediction_data["order_date"] = pd.to_datetime(prediction_data["order_date"])
            try:
                top_divs = (
                    prediction_data["product_division"]
                    .astype(str)
                    .str.rstrip()
                    .value_counts()
                    .head(20)
                )
                logger.info("Top product_division in window:\n%s", top_divs.to_string())
            except Exception:
                pass

            logger.info(
                f"Feature data: {len(feature_data)} transactions <= {cutoff_date}"
            )
            logger.info(
                f"Prediction data: {len(prediction_data)} transactions in {prediction_window_months}-month window"
            )
        else:
            # No cutoff provided: load all historical data for the relevant divisions
            feature_sql_qty = (
                f"SELECT {base_cols_with_qty} FROM fact_transactions "
                f"WHERE LOWER(TRIM(product_division)) IN ({division_filter})"
            )
            feature_sql_noqty = (
                f"SELECT {base_cols_no_qty} FROM fact_transactions "
                f"WHERE LOWER(TRIM(product_division)) IN ({division_filter})"
            )
            try:
                feature_data = _read_sql(feature_sql_qty)
            except Exception:
                feature_data = _read_sql(feature_sql_noqty)
                if not feature_data.empty and 'quantity' not in feature_data.columns:
                    feature_data['quantity'] = 1.0
            feature_data["order_date"] = pd.to_datetime(feature_data["order_date"])
            prediction_data = feature_data.copy()

        # Ensure quantity present if upstream fallback was used
        if 'quantity' not in feature_data.columns:
            feature_data['quantity'] = 1.0
        transactions = pl.from_pandas(feature_data)
        if "customer_id" in transactions.columns:
            transactions = transactions.with_columns(
                pl.col("customer_id").cast(pl.Utf8)
            )

        customers_pd = pd.read_sql_query(
            "SELECT customer_id FROM dim_customer", engine
        )
        customers_pd["customer_id"] = customers_pd["customer_id"].astype(str)
        try:
            cfgmod = cfg.load_config()
            exclude_prospects_early = bool(getattr(getattr(cfgmod, 'population', object()), 'exclude_prospects_in_features', False))
            include_prospects_train = bool(getattr(getattr(cfgmod, 'population', object()), 'include_prospects', False))
            build_segments = [str(s).strip().lower() for s in (getattr(getattr(cfgmod, 'population', object()), 'build_segments', ["warm","cold"]))]
        except Exception:
            exclude_prospects_early = False
            include_prospects_train = False
            build_segments = ["warm","cold"]
        # Preserve full customer roster for diagnostics and missingness flags. We still gate
        # transactions/prediction data above to reduce compute when exclude_prospects_early is True.
        # This ensures rows without transactions remain present with appropriate *_missing flags,
        # while scoring can later filter segments as needed.
        customers = pl.from_pandas(customers_pd).with_columns(
            pl.col("customer_id").cast(pl.Utf8)
        )
    except Exception as e:
        logger.error(f"Failed to read necessary tables from the database: {e}")
        return pl.DataFrame()

    if transactions.is_empty() or customers.is_empty():
        logger.warning("Transactions or customers data is empty. Cannot build feature matrix.")
        return pl.DataFrame()

    # --- 2. Create the Binary Target Variable ---
    # Target: 1 if the customer bought any product in the target division in the prediction window, 0 otherwise.
    if cutoff_date:
        # Build prediction buyers either by SKU set (custom targets) or by division
        if use_custom_targets:
            mask = prediction_data['product_sku'].astype(str).isin(target_skus)
        else:
            pred_div = (
                prediction_data['product_division']
                .astype(str)
                .str.strip()
                .str.casefold()
            )
            if use_alias:
                mask = pred_div.isin(alias_divisions)
            else:
                # Optional product_goal targeting when provided via curated facts
                use_goal = False
                if 'product_goal' in prediction_data.columns:
                    pred_goal = prediction_data['product_goal'].astype(str).str.strip().str.casefold()
                    try:
                        use_goal = pred_goal.eq(norm_division_name).any()
                    except Exception:
                        use_goal = False
                mask = (prediction_data['product_goal'].astype(str).str.strip().str.casefold() == norm_division_name) if use_goal else (pred_div == norm_division_name)

        # Rollup targeting via line items when division doesn't match aliases/goals/models
        if not use_custom_targets and not use_alias:
            try:
                if 'use_goal' not in locals():
                    use_goal = False
            except Exception:
                use_goal = False
        use_rollup = False
        try:
            if not use_custom_targets and not use_alias and not use_goal:
                from dateutil.relativedelta import relativedelta as _rd
                cutoff_dt2 = pd.to_datetime(cutoff_date)
                pred_end2 = (cutoff_dt2 + _rd(months=prediction_window_months)).strftime("%Y-%m-%d")
                rollup_q = (
                    "SELECT CompanyId AS customer_id, Rec_Date AS order_date, item_rollup "
                    "FROM fact_sales_line WHERE Rec_Date > :cutoff AND Rec_Date <= :pred_end"
                )
                rwin = pd.read_sql_query(rollup_q, engine, params={"cutoff": cutoff_dt2.strftime("%Y-%m-%d"), "pred_end": pred_end2})
                if not rwin.empty and 'item_rollup' in rwin.columns:
                    # Normalize ids to integer-like strings to match dim_customer
                    rwin['customer_id'] = pd.to_numeric(rwin['customer_id'], errors='coerce').astype('Int64')
                    il = rwin['item_rollup'].astype(str).str.strip().str.casefold()
                    sel = il == norm_division_name
                    if sel.any():
                        prediction_buyers_df = rwin.loc[sel, 'customer_id'].astype(str).unique()
                        use_rollup = True
            # If rollup buyers were found, override default mask path
            if use_rollup:
                division_buyers_pd = pd.DataFrame({'customer_id': prediction_buyers_df, 'bought_in_division': 1})
                division_buyers = pl.from_pandas(division_buyers_pd).with_columns(pl.col('customer_id').cast(pl.Utf8)).lazy()
            else:
                prediction_buyers_df = prediction_data.loc[mask, 'customer_id'].astype(str).unique()
                division_buyers_pd = pd.DataFrame({'customer_id': prediction_buyers_df, 'bought_in_division': 1})
                division_buyers = pl.from_pandas(division_buyers_pd).with_columns(pl.col('customer_id').cast(pl.Utf8)).lazy()
        except Exception:
            prediction_buyers_df = prediction_data.loc[mask, 'customer_id'].astype(str).unique()
            division_buyers_pd = pd.DataFrame({'customer_id': prediction_buyers_df, 'bought_in_division': 1})
            division_buyers = pl.from_pandas(division_buyers_pd).with_columns(pl.col('customer_id').cast(pl.Utf8)).lazy()
        logger.info(f"Target: {len(prediction_buyers_df)} customers bought {division_name} in prediction window")
    else:
        # Original behavior: ever bought in historical data
        division_buyers = (
            transactions.lazy()
            .filter(label_filter)
            .select("customer_id")
            .unique()
            .with_columns(pl.lit(1).cast(pl.Int8).alias("bought_in_division"))
        )

    # --- 3. Engineer Behavioral Features ---
    # Recency anchor: use cutoff when provided to avoid temporal leakage; else 'now'.
    try:
        reference_date = pd.to_datetime(cutoff_date).date() if cutoff_date else pd.Timestamp.utcnow().date()
    except Exception:
        reference_date = pd.Timestamp.utcnow().date()

    features = (
        transactions.lazy()
        .group_by("customer_id")
        .agg([
            # Recency Features
            pl.col("order_date").max().alias("last_order_date"),
            pl.col("order_date").filter(label_filter).max().alias(f"last_{division_name}_order_date"),
            
            # Frequency Features
            pl.len().alias("total_transactions_all_time"),
            pl.col("order_date").filter(pl.col("order_date").dt.year().is_in([2023, 2024])).len().alias("transactions_last_2y"),
            
            # Monetary Features
            pl.sum("gross_profit").alias("total_gp_all_time"),
            pl.col("gross_profit").filter(pl.col("order_date").dt.year().is_in([2023, 2024])).sum().alias("total_gp_last_2y"),
            pl.mean("gross_profit").alias("avg_transaction_gp"),
            
            # Cross-division behavioral patterns (non-leaky features)
            pl.col("product_division").filter(pl.col("product_division") == "Services").len().alias("services_transaction_count"),
            pl.col("product_division").filter(pl.col("product_division") == "Simulation").len().alias("simulation_transaction_count"),
            pl.col("product_division").filter(pl.col("product_division") == "Hardware").len().alias("hardware_transaction_count"),
            
            # Services engagement (proxy for technical sophistication)
            pl.col("gross_profit").filter(pl.col("product_division") == "Services").sum().alias("total_services_gp"),
            pl.col("gross_profit").filter(pl.col("product_sku") == "Training").sum().alias("total_training_gp"),
            
            # Growth trajectory features
            pl.col("gross_profit").filter(pl.col("order_date").dt.year() == 2024).sum().alias("gp_2024"),
            pl.col("gross_profit").filter(pl.col("order_date").dt.year() == 2023).sum().alias("gp_2023"),
            
            # General engagement features
            pl.n_unique("product_division").alias("product_diversity_score"),
            pl.n_unique("product_sku").alias("sku_diversity_score"),
        ])
        .collect()
    )

    # Calculate recency features in pandas for easier date arithmetic
    features_pd = features.to_pandas()
    
    # Handle date columns properly
    if 'last_order_date' in features_pd.columns:
        # Convert string dates to datetime and calculate days difference
        last_order_dates = pd.to_datetime(features_pd['last_order_date'], errors='coerce')
        # Calculate days difference safely
        days_diff = []
        for date in last_order_dates:
            if pd.isna(date):
                days_diff.append(999)
            else:
                days_diff.append((reference_date - date.date()).days)
        features_pd['days_since_last_order'] = days_diff
    else:
        features_pd['days_since_last_order'] = 999  # Default for customers with no orders
        
    division_date_col = f'last_{division_name}_order_date'
    if division_date_col in features_pd.columns:
        # Convert string dates to datetime and calculate days difference
        last_division_dates = pd.to_datetime(features_pd[division_date_col], errors='coerce')
        # Calculate days difference safely
        days_diff = []
        for date in last_division_dates:
            if pd.isna(date):
                days_diff.append(999)
            else:
                days_diff.append((reference_date - date.date()).days)
        features_pd[f'days_since_last_{division_name}_order'] = days_diff
    else:
        features_pd[f'days_since_last_{division_name}_order'] = 999  # Default for customers with no orders in division

    # Apply recency floor guard to reduce near-cutoff signals
    try:
        rec_floor = int(getattr(cfg.load_config().features, 'recency_floor_days', 0))
    except Exception:
        rec_floor = 0
    if rec_floor and rec_floor > 0:
        try:
            features_pd['days_since_last_order'] = pd.to_numeric(features_pd['days_since_last_order'], errors='coerce').fillna(999).clip(lower=rec_floor)
        except Exception:
            pass
        try:
            coln = f'days_since_last_{division_name}_order'
            features_pd[coln] = pd.to_numeric(features_pd[coln], errors='coerce').fillna(999).clip(lower=rec_floor)
        except Exception:
            pass
    
    # --- 3a. Windowed RFM and temporal dynamics (pandas) ---
    try:
        cfgmod = cfg.load_config()
        fd = feature_data.copy()
        fd['order_date'] = pd.to_datetime(fd['order_date'])
        cutoff_dt = pd.to_datetime(cutoff_date) if cutoff_date else fd['order_date'].max()
        # Ensure window list is defined even if advanced block is skipped
        window_months = cfgmod.features.windows_months or [3, 6, 12, 24]
        # Align dtype with features_pd merges to avoid expensive coercions
        try:
            fd['customer_id'] = fd['customer_id'].astype(str)
        except Exception:
            pass
        fd = fd.sort_values(['customer_id', 'order_date'])

        # Heuristic: on large SQLite datasets, skip advanced extras to cap memory/time (configurable)
        try:
            _is_sqlite = getattr(engine, 'dialect', None) and engine.dialect.name == 'sqlite'
        except Exception:
            _is_sqlite = False
        try:
            adv_thr = int(getattr(cfgmod.features, 'sqlite_skip_advanced_rows', 10_000_000))
        except Exception:
            adv_thr = 10_000_000
        if adv_thr <= 0:
            effective_adv_thr = float("inf")
        else:
            effective_adv_thr = adv_thr
            # In test/local SQLite with configured external DB, apply a conservative cap unless explicitly overridden
            try:
                configured_engine = str(getattr(cfgmod.database, 'engine', '')).strip().lower()
            except Exception:
                configured_engine = ''
            if _is_sqlite and configured_engine and configured_engine != 'sqlite':
                effective_adv_thr = min(adv_thr, 50_000)
        if _is_sqlite and len(fd) > effective_adv_thr:
            logger.warning(
                "Skipping advanced temporal features: rows=%d exceeds limit=%d (sqlite_skip_advanced_rows=%d)",
                len(fd), effective_adv_thr, adv_thr,
            )
            raise RuntimeError('skip_advanced_large_sqlite')

        window_months = cfgmod.features.windows_months or [3, 6, 12, 24]
        per_customer_frames = []
        per_customer_frames_div = []

        enable_canonical = bool(getattr(cfgmod.features, "enable_canonical_division_features", True))
        enable_margin = bool(getattr(cfgmod.features, "enable_margin_features", True))
        enable_currency = bool(getattr(cfgmod.features, "enable_currency_mix", True))
        enable_rollup_diversity = bool(getattr(cfgmod.features, "enable_rollup_diversity", True))
        enable_returns_features = bool(getattr(cfgmod.features, "enable_returns_features", True))
        enable_order_comp = bool(getattr(cfgmod.features, "enable_order_composition", True))
        enable_trend_features = bool(getattr(cfgmod.features, "enable_trend_features", True))
        enable_tag_als = bool(getattr(cfgmod.features, "enable_tag_als", False))
        use_usd_monetary = bool(getattr(cfgmod.features, "use_usd_monetary", True))

        line_df = pd.DataFrame()
        try:
            need_line = any(
                [
                    enable_canonical,
                    enable_margin,
                    enable_currency,
                    enable_rollup_diversity,
                    enable_returns_features,
                    enable_order_comp,
                    enable_trend_features,
                ]
            )
        except Exception:
            need_line = True

        if need_line:
            cutoff_literal_param = pd.to_datetime(cutoff_dt).strftime("%Y-%m-%d") if cutoff_dt is not None else pd.Timestamp.utcnow().strftime("%Y-%m-%d")
            try:
                line_query = """
                    SELECT
                        CompanyId AS customer_id,
                        Rec_Date AS order_date,
                        division_canonical,
                        item_rollup,
                        Sales_Order,
                        Revenue_usd,
                        Revenue,
                        GP_usd,
                        GP,
                        Term_GP_usd,
                        Term_GP,
                        SalesOrder_Currency,
                        USD_CAD_Conversion_rate,
                        is_return_line
                    FROM fact_sales_line
                    WHERE Rec_Date <= :cutoff
                """
                line_df = pd.read_sql_query(line_query, engine, params={"cutoff": cutoff_literal_param})
            except Exception as exc:
                logger.warning("Line-item feature load failed with extended columns (%s); retrying without return flag.", exc)
                try:
                    
                    fallback_query = f"""
                        SELECT
                            CompanyId AS customer_id,
                            Rec_Date AS order_date,
                            division_canonical,
                            item_rollup,
                            Sales_Order,
                            Revenue_usd,
                            Revenue,
                            GP_usd,
                            GP,
                            Term_GP_usd,
                            Term_GP,
                            SalesOrder_Currency,
                            USD_CAD_Conversion_rate
                        FROM fact_sales_line
                        WHERE Rec_Date <= '{cutoff_literal_param}'
                    """
                    line_df = pd.read_sql_query(fallback_query, engine)
                except Exception as fallback_exc:
                    logger.warning("Line-item feature fallback load failed: %s", fallback_exc)
                    line_df = pd.DataFrame()
        for w in window_months:
            start_dt = cutoff_dt - pd.DateOffset(months=w)
            effective_end = cutoff_dt
            try:
                if mask_tail_days is not None and int(mask_tail_days) > 0:
                    effective_end = cutoff_dt - pd.Timedelta(days=int(mask_tail_days))
            except Exception:
                effective_end = cutoff_dt
            if effective_end <= start_dt:
                sub = fd.head(0)[['customer_id', 'order_date', 'gross_profit']]
            else:
                mask = (fd['order_date'] > start_dt) & (fd['order_date'] <= effective_end)
                sub = fd.loc[mask, ['customer_id', 'order_date', 'gross_profit']]
            # Winsorize GP at config p
            gp_w = sub.groupby('customer_id')['gross_profit'].sum().rename('gp_sum_last_w').reset_index()
            # For mean, compute robustly
            gp_mean = sub.groupby('customer_id')['gross_profit'].mean().rename('gp_mean_last_w').reset_index()
            tx_n = sub.groupby('customer_id')['order_date'].count().rename('tx_count_last_w').reset_index()
            agg = tx_n.merge(gp_w, on='customer_id', how='outer').merge(gp_mean, on='customer_id', how='outer')
            agg.rename(columns={
                'tx_count_last_w': f'tx_count_last_{w}m',
                'gp_sum_last_w': f'gp_sum_last_{w}m',
                'gp_mean_last_w': f'gp_mean_last_{w}m',
            }, inplace=True)
            agg[f'avg_gp_per_tx_last_{w}m'] = agg[f'gp_sum_last_{w}m'] / agg[f'tx_count_last_{w}m'].replace(0, np.nan)
            agg[f'avg_gp_per_tx_last_{w}m'] = agg[f'avg_gp_per_tx_last_{w}m'].fillna(0.0)
            # Margin proxy for all scope
            col_all_gp = f'gp_sum_last_{w}m'
            agg[f'margin__all__gp_pct__{w}m'] = agg[col_all_gp].astype(float) / (agg[col_all_gp].abs().astype(float) + 1e-9)
            per_customer_frames.append(agg)

            # Division-specific aggregates + margin (gp_pct) proxy over window
            if use_alias:
                sub_div = fd.loc[mask & (fd['product_division'].astype(str).str.strip().str.casefold().isin(alias_divisions)), ['customer_id', 'order_date', 'gross_profit']]
            else:
                # If product_goal present and matches target, prefer that; else use division
                if 'product_goal' in fd.columns:
                    sub_goal = fd['product_goal'].astype(str).str.strip().str.casefold() == norm_division_name
                else:
                    sub_goal = False
                sub_div = fd.loc[mask & (sub_goal | (fd['product_division'].astype(str).str.strip() == norm_division_name)), ['customer_id', 'order_date', 'gross_profit']]
            tx_n_div = sub_div.groupby('customer_id')['order_date'].count().rename(f'rfm__div__tx_n__{w}m').reset_index()
            gp_sum_div = sub_div.groupby('customer_id')['gross_profit'].sum().rename(f'rfm__div__gp_sum__{w}m').reset_index()
            gp_mean_div = sub_div.groupby('customer_id')['gross_profit'].mean().rename(f'rfm__div__gp_mean__{w}m').reset_index()
            # Margin proxy: gp_pct = gp_sum / |gp_sum| + epsilon (since revenue is not available here)
            agg_div = tx_n_div.merge(gp_sum_div, on='customer_id', how='outer').merge(gp_mean_div, on='customer_id', how='outer')
            col_gp = f'rfm__div__gp_sum__{w}m'
            agg_div[f'margin__div__gp_pct__{w}m'] = agg_div[col_gp].astype(float) / (agg_div[col_gp].abs().astype(float) + 1e-9)
            per_customer_frames_div.append(agg_div)

            # Offset windows (end at cutoff - offset_days)
            try:
                cfgf = cfg.load_config().features
                if bool(getattr(cfgf, 'enable_offset_windows', True)):
                    offsets = list(getattr(cfgf, 'offset_days', [60]))
                    for off in offsets:
                        try:
                            off = int(off)
                        except Exception:
                            continue
                        end_off = cutoff_dt - pd.Timedelta(days=off)
                        start_off = end_off - pd.DateOffset(months=w)
                        if end_off <= start_off:
                            sub_off = fd.head(0)[['customer_id','order_date','gross_profit']]
                        else:
                            m_off = (fd['order_date'] > start_off) & (fd['order_date'] <= end_off)
                            sub_off = fd.loc[m_off, ['customer_id','order_date','gross_profit','product_division']]
                        # All-scope aggregates
                        gp_w_off = sub_off.groupby('customer_id')['gross_profit'].sum().rename(f'rfm__all__gp_sum__{w}m_off{off}d').reset_index()
                        gp_mean_off = sub_off.groupby('customer_id')['gross_profit'].mean().rename(f'rfm__all__gp_mean__{w}m_off{off}d').reset_index()
                        tx_n_off = sub_off.groupby('customer_id')['order_date'].count().rename(f'rfm__all__tx_n__{w}m_off{off}d').reset_index()
                        agg_off = tx_n_off.merge(gp_w_off, on='customer_id', how='outer').merge(gp_mean_off, on='customer_id', how='outer')
                        per_customer_frames.append(agg_off)
                        # Division-specific aggregates (match norm_division_name)
                        if use_alias:
                            sub_div_off = sub_off.loc[sub_off['product_division'].astype(str).str.strip().str.casefold().isin(alias_divisions), ['customer_id','order_date','gross_profit']]
                        else:
                            if 'product_goal' in sub_off.columns:
                                sub_goal_off = sub_off['product_goal'].astype(str).str.strip().str.casefold() == norm_division_name
                            else:
                                sub_goal_off = False
                            sub_div_off = sub_off.loc[sub_goal_off | (sub_off['product_division'].astype(str).str.strip() == norm_division_name), ['customer_id','order_date','gross_profit']]
                        tx_n_div_off = sub_div_off.groupby('customer_id')['order_date'].count().rename(f'rfm__div__tx_n__{w}m_off{off}d').reset_index()
                        gp_sum_div_off = sub_div_off.groupby('customer_id')['gross_profit'].sum().rename(f'rfm__div__gp_sum__{w}m_off{off}d').reset_index()
                        gp_mean_div_off = sub_div_off.groupby('customer_id')['gross_profit'].mean().rename(f'rfm__div__gp_mean__{w}m_off{off}d').reset_index()
                        agg_div_off = tx_n_div_off.merge(gp_sum_div_off, on='customer_id', how='outer').merge(gp_mean_div_off, on='customer_id', how='outer')
                        per_customer_frames_div.append(agg_div_off)
            except Exception:
                pass

        line_feature_frames = _compute_line_window_features(
            line_df,
            cutoff_dt=cutoff_dt,
            windows=window_months,
            target_division=norm_division_name,
            use_usd=use_usd_monetary,
            enable_canonical=enable_canonical,
            enable_margin=enable_margin,
            enable_currency=enable_currency,
            enable_diversity=enable_rollup_diversity,
            enable_returns=enable_returns_features,
            enable_order_comp=enable_order_comp,
            enable_trend=enable_trend_features,
        ) if not line_df.empty else []

        per_customer_frames.extend(line_feature_frames)
        # Monthly resample for slope/volatility over last 12 months
        last12_mask = (fd['order_date'] > (cutoff_dt - pd.DateOffset(months=12))) & (fd['order_date'] <= cutoff_dt)
        m = fd.loc[last12_mask, ['customer_id', 'order_date', 'gross_profit']].copy()
        m['ym'] = m['order_date'].values.astype('datetime64[M]')
        monthly = m.groupby(['customer_id', 'ym']).agg(month_gp=('gross_profit', 'sum'), month_tx=('gross_profit', 'count')).reset_index()

        def _slope_std(df: pd.DataFrame, value_col: str):
            # ensure 12 points by reindexing months
            months = pd.date_range((cutoff_dt - pd.DateOffset(months=11)).to_period('M').to_timestamp(),
                                   cutoff_dt.to_period('M').to_timestamp(), freq='MS')
            tmp = df.set_index('ym').reindex(months).fillna(0.0)
            y = tmp[value_col].values.astype(float)
            x = np.arange(len(y))
            if np.any(np.isfinite(y)) and len(y) >= 3:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0.0
            stdv = float(np.std(y))
            return pd.Series({'slope': slope, 'std': stdv})

        try:
            gp_dynamics = monthly.groupby('customer_id').apply(lambda df: _slope_std(df, 'month_gp'), include_groups=False).reset_index()
        except TypeError:
            gp_dynamics = monthly.groupby('customer_id').apply(lambda df: _slope_std(df, 'month_gp')).reset_index()
        gp_dynamics.rename(columns={'slope': 'gp_monthly_slope_12m', 'std': 'gp_monthly_std_12m'}, inplace=True)
        try:
            tx_dynamics = monthly.groupby('customer_id').apply(lambda df: _slope_std(df, 'month_tx'), include_groups=False).reset_index()
        except TypeError:
            tx_dynamics = monthly.groupby('customer_id').apply(lambda df: _slope_std(df, 'month_tx')).reset_index()
        tx_dynamics.rename(columns={'slope': 'tx_monthly_slope_12m', 'std': 'tx_monthly_std_12m'}, inplace=True)

        # Tenure and interpurchase intervals
        first_last = fd.groupby('customer_id').agg(first_order=('order_date', 'min'), last_order=('order_date', 'max')).reset_index()
        first_last['tenure_days'] = (cutoff_dt - first_last['first_order']).dt.days
        # Interpurchase intervals
        def _intervals(g: pd.DataFrame):
            dates = g['order_date'].sort_values().values
            if len(dates) < 2:
                return pd.Series({'ipi_median_days': 0.0, 'ipi_mean_days': 0.0, 'last_gap_days': float((cutoff_dt - g['order_date'].max()).days)})
            diffs = np.diff(dates).astype('timedelta64[D]').astype(int)
            return pd.Series({'ipi_median_days': float(np.median(diffs)), 'ipi_mean_days': float(np.mean(diffs)), 'last_gap_days': float((cutoff_dt - g['order_date'].max()).days)})

        try:
            ipi = fd.groupby('customer_id').apply(_intervals, include_groups=False).reset_index()
        except TypeError:
            ipi = fd.groupby('customer_id').apply(_intervals).reset_index()

        # Active months over last 24 months
        last24 = fd[(fd['order_date'] > (cutoff_dt - pd.DateOffset(months=24))) & (fd['order_date'] <= cutoff_dt)].copy()
        if not last24.empty:
            last24['ym'] = last24['order_date'].values.astype('datetime64[M]')
            active = last24.groupby('customer_id')['ym'].nunique().rename('lifecycle__all__active_months__24m').reset_index()
        else:
            active = pd.DataFrame(columns=['customer_id','lifecycle__all__active_months__24m'])

        # Seasonality (Q1..Q4 proportions over last 24 months)
        try:
            last24_mask = (fd['order_date'] > (cutoff_dt - pd.DateOffset(months=24))) & (fd['order_date'] <= cutoff_dt)
            s = fd.loc[last24_mask, ['customer_id', 'order_date']].copy()
            if not s.empty:
                s['quarter'] = s['order_date'].dt.quarter
                season_counts = s.groupby(['customer_id', 'quarter']).size().unstack(fill_value=0)
                # Ensure all quarters present
                for qnum in [1, 2, 3, 4]:
                    if qnum not in season_counts.columns:
                        season_counts[qnum] = 0
                season_counts = season_counts.rename(columns={1: 'q1_count_24m', 2: 'q2_count_24m', 3: 'q3_count_24m', 4: 'q4_count_24m'})
                season_counts['season_total_24m'] = season_counts[['q1_count_24m','q2_count_24m','q3_count_24m','q4_count_24m']].sum(axis=1)
                for q in ['q1', 'q2', 'q3', 'q4']:
                    denom = season_counts['season_total_24m'].replace(0, np.nan)
                    season_counts[f'{q}_share_24m'] = season_counts[f'{q}_count_24m'] / denom
                    season_counts[f'{q}_share_24m'] = season_counts[f'{q}_share_24m'].fillna(0.0)
                season_counts = season_counts.reset_index()[['customer_id', 'q1_share_24m', 'q2_share_24m', 'q3_share_24m', 'q4_share_24m']]
            else:
                season_counts = pd.DataFrame(columns=['customer_id','q1_share_24m','q2_share_24m','q3_share_24m','q4_share_24m'])
        except Exception:
            season_counts = pd.DataFrame(columns=['customer_id','q1_share_24m','q2_share_24m','q3_share_24m','q4_share_24m'])

        # Division-level features (last 12 months)
        try:
            known_divisions = list(division_set())
            if not known_divisions:
                known_divisions = ['Solidworks', 'Services', 'Simulation', 'Hardware']
        except Exception:
            known_divisions = ['Solidworks', 'Services', 'Simulation', 'Hardware']
        dl = fd.loc[last12_mask, ['customer_id', 'product_division', 'gross_profit']].copy()
        div_gp = dl.groupby(['customer_id', 'product_division'])['gross_profit'].sum().unstack(fill_value=0.0)
        div_gp = div_gp.reindex(columns=known_divisions, fill_value=0.0)
        div_gp = div_gp.add_prefix('gp_12m_')
        div_tx = dl.groupby(['customer_id', 'product_division']).size().unstack(fill_value=0)
        div_tx = div_tx.reindex(columns=known_divisions, fill_value=0).add_prefix('tx_12m_')
        div_df = div_gp.join(div_tx, how='outer').reset_index()
        div_df['gp_12m_total'] = div_df[[f'gp_12m_{d}' for d in known_divisions]].sum(axis=1)
        for d in known_divisions:
            div_df[f'{d.lower()}_gp_share_12m'] = div_df[f'gp_12m_{d}'] / div_df['gp_12m_total'].replace(0, np.nan)
            div_df[f'{d.lower()}_gp_share_12m'] = div_df[f'{d.lower()}_gp_share_12m'].fillna(0.0)

        # EB smoothing for target division share (optional)
        try:
            if cfgmod.features.use_eb_smoothing:
                target_col = f'{division_name.lower()}_gp_share_12m'
                if target_col in div_df.columns:
                    prior = float(div_df[target_col].mean())
                    alpha = 5.0
                    num = div_df[f'gp_12m_{division_name}'] + alpha * prior
                    den = div_df['gp_12m_total'] + alpha
                    div_df['xdiv__div__gp_share__12m'] = (num / den).fillna(0.0)
        except Exception:
            pass

        # Division recency (days since last division order)
        rec_div_list = []
        for d in known_divisions:
            target_norm = normalize_division(d)
            sub = fd.loc[
                fd['product_division'].astype(str).str.strip().str.casefold() == target_norm,
                ['customer_id', 'order_date']
            ]
            last_d = sub.groupby('customer_id')['order_date'].max().reset_index()
            last_d[f'days_since_last_{d.lower()}'] = (cutoff_dt - last_d['order_date']).dt.days
            rec_div_list.append(last_d[['customer_id', f'days_since_last_{d.lower()}']])
        rec_div = None
        if rec_div_list:
            rec_div = rec_div_list[0]
            for extra in rec_div_list[1:]:
                rec_div = rec_div.merge(extra, on='customer_id', how='outer')
        if rec_div is not None and rec_floor:
            try:
                for c in [c for c in rec_div.columns if c.startswith('days_since_last_')]:
                    rec_div[c] = pd.to_numeric(rec_div[c], errors='coerce').fillna(999).clip(lower=rec_floor)
            except Exception:
                pass

        # SKU-level features (last 12 months)
        important_skus = ['SWX_Core', 'SWX_Pro_Prem', 'Core_New_UAP', 'Pro_Prem_New_UAP', 'PDM', 'Simulation', 'Services', 'Training', 'Success Plan GP', 'Supplies', 'SW_Plastics', 'AM_Software', 'DraftSight', 'Fortus', 'HV_Simulation', 'CATIA', 'Delmia_Apriso']
        sl = fd.loc[last12_mask, ['customer_id', 'product_sku', 'gross_profit', 'quantity']].copy()
        sku_gp = sl.groupby(['customer_id', 'product_sku'])['gross_profit'].sum().unstack(fill_value=0.0)
        sku_gp = sku_gp.reindex(columns=important_skus, fill_value=0.0).add_prefix('sku_gp_12m_')
        sku_qty = sl.groupby(['customer_id', 'product_sku'])['quantity'].sum().unstack(fill_value=0.0)
        sku_qty = sku_qty.reindex(columns=important_skus, fill_value=0.0).add_prefix('sku_qty_12m_')
        sku_df = sku_gp.join(sku_qty, how='outer').reset_index()
        # GP per unit ratios
        for s in important_skus:
            gp_col = f'sku_gp_12m_{s}'
            qty_col = f'sku_qty_12m_{s}'
            ratio_col = f'sku_gp_per_unit_12m_{s}'
            if gp_col in sku_df.columns and qty_col in sku_df.columns:
                sku_df[ratio_col] = sku_df[gp_col] / sku_df[qty_col].replace(0, np.nan)
                sku_df[ratio_col] = sku_df[ratio_col].fillna(0.0)

        # Past Solidworks buyer flag (historical)
        sw_norm = normalize_division('Solidworks')
        past_swx = fd.loc[
            fd['product_division'].astype(str).str.strip().str.casefold() == sw_norm,
            ['customer_id']
        ].drop_duplicates()
        past_swx['ever_bought_solidworks'] = 1

        # Merge all extra frames to features_pd
        extra = features_pd[['customer_id']].copy()
        for df_merge in per_customer_frames + per_customer_frames_div + [gp_dynamics, tx_dynamics, first_last[['customer_id', 'tenure_days']], ipi, active, season_counts, div_df, sku_df, past_swx]:
            if df_merge is None or len(df_merge) == 0:
                continue
            extra = extra.merge(df_merge, on='customer_id', how='left')

        # Fill NaNs
        for col in extra.columns:
            if col == 'customer_id':
                continue
            extra[col] = extra[col].fillna(0)

        # Attach to features_pd
        features_pd = features_pd.merge(extra, on='customer_id', how='left')

        line_meta_df = _load_line_item_metadata(engine, cutoff_date)

        # --- Region (Branch) and Rep features from preserved raw data ---
        try:
            sl = _prepare_branch_source(line_meta_df)
            if sl.empty:
                sl = _load_branch_rep_curated(engine, cutoff_date)

            if sl is None or sl.empty:
                logger.info("Branch/Rep feature build skipped: no branch data available.")
            else:
                if "order_date" in sl.columns:
                    sl["order_date"] = pd.to_datetime(sl["order_date"], errors='coerce')
                sl["customer_id"] = sl["customer_id"].astype(str)
                branch_available = "branch" in sl.columns and sl["branch"].notna().any()
                rep_available = "rep" in sl.columns and sl["rep"].notna().any()
                if not branch_available and not rep_available:
                    logger.info("Branch/Rep feature build skipped: branch and rep columns missing.")
                else:
                    import re

                    def sanitize_key(text: str) -> str:
                        if text is None:
                            return "unknown"
                        key = str(text).lower().strip()
                        key = key.replace("&", " and ")
                        key = re.sub(r"[^0-9a-zA-Z]+", "_", key)
                        key = re.sub(r"_+", "_", key).strip("_")
                        return key or "unknown"

                    frames = []
                    if branch_available:
                        b = sl[['customer_id', 'branch']].copy()
                        b['branch'] = b['branch'].astype(str).str.strip()
                        b['count'] = 1
                        totals = b.groupby('customer_id')['count'].sum().rename('branch_tx_total')
                        if enable_order_comp:
                            branch_nuniq = b.groupby('customer_id')['branch'].nunique().rename('branch__nunique__life')
                            branch_top = (
                                b.groupby(['customer_id', 'branch'])['count'].sum().groupby('customer_id').max()
                                / totals.replace(0, np.nan)
                            ).rename('branch__top_share__life').replace([np.inf, -np.inf], np.nan).fillna(0.0)
                            frames.append(pd.concat([branch_nuniq, branch_top], axis=1))
                        top_branches = (
                            b['branch']
                            .value_counts()
                            .head(30)
                            .index
                            .tolist()
                        )
                        if top_branches:
                            b_top = (
                                b[b['branch'].isin(top_branches)]
                                .groupby(['customer_id', 'branch'])['count']
                                .sum()
                                .unstack(fill_value=0)
                            )
                            b_top = b_top.div(totals, axis=0).fillna(0.0)
                            b_top.columns = [f"branch_share_{sanitize_key(c)}" for c in b_top.columns]
                            frames.append(b_top)
                    if rep_available:
                        r = sl[['customer_id', 'rep']].copy()
                        r['rep'] = r['rep'].astype(str).str.strip()
                        r['count'] = 1
                        totals = r.groupby('customer_id')['count'].sum().rename('rep_tx_total')
                        if enable_order_comp:
                            rep_nuniq = r.groupby('customer_id')['rep'].nunique().rename('rep__nuniq__life')
                            rep_top = (
                                r.groupby(['customer_id', 'rep'])['count'].sum().groupby('customer_id').max()
                                / totals.replace(0, np.nan)
                            ).rename('rep__top_share__life').replace([np.inf, -np.inf], np.nan).fillna(0.0)
                            frames.append(pd.concat([rep_nuniq, rep_top], axis=1))
                        top_reps = (
                            r['rep']
                            .value_counts()
                            .head(50)
                            .index
                            .tolist()
                        )
                        if top_reps:
                            r_top = (
                                r[r['rep'].isin(top_reps)]
                                .groupby(['customer_id', 'rep'])['count']
                                .sum()
                                .unstack(fill_value=0)
                            )
                            r_top = r_top.div(totals, axis=0).fillna(0.0)
                            r_top.columns = [f"rep_share_{sanitize_key(c)}" for c in r_top.columns]
                            frames.append(r_top)

                    if frames:
                        br = frames[0]
                        for frame in frames[1:]:
                            br = br.join(frame, how='outer')
                        br = br.reset_index()
                        features_pd = features_pd.merge(br, on='customer_id', how='left')
                        for col in br.columns:
                            if col == 'customer_id':
                                continue
                            features_pd[col] = features_pd[col].fillna(0.0)
                    else:
                        logger.info("Branch/Rep feature build skipped: insufficient distribution data.")
        except Exception as e:
            logger.warning(f"Branch/Rep feature build failed: {e}")

        if enable_tag_als and not line_df.empty:
            tag_df = line_df.copy()
            tag_df["customer_id"] = tag_df["customer_id"].astype(str)
            revenue_series = None
            if use_usd_monetary and "Revenue_usd" in tag_df.columns:
                revenue_series = pd.to_numeric(tag_df["Revenue_usd"], errors="coerce")
            if revenue_series is None or revenue_series.isna().all():
                if "Revenue" in tag_df.columns:
                    revenue_series = pd.to_numeric(tag_df["Revenue"], errors="coerce")
                else:
                    revenue_series = pd.Series(0.0, index=tag_df.index)
            tag_df["revenue_value"] = revenue_series.fillna(0.0)
            tag_df["item_rollup"] = tag_df.get("item_rollup", pd.Series(["unknown"] * len(tag_df))).fillna("unknown").astype(str).str.strip().str.lower()
            top_rollups = (
                tag_df.groupby("item_rollup")["revenue_value"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .index
                .tolist()
            )
            if top_rollups:
                tag_pivot = (
                    tag_df[tag_df["item_rollup"].isin(top_rollups)]
                    .groupby(["customer_id", "item_rollup"])["revenue_value"]
                    .sum()
                    .unstack(fill_value=0.0)
                )
                totals = tag_df.groupby("customer_id")["revenue_value"].sum().replace(0.0, np.nan)
                tag_pivot = tag_pivot.div(totals, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                def _sanitize_tag(name: str) -> str:
                    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower())
                    return cleaned.strip("_") or "unknown"

                tag_pivot.columns = [f"tagals__share_{_sanitize_tag(col)}" for col in tag_pivot.columns]
                tag_pivot = tag_pivot.reset_index()
                features_pd = features_pd.merge(tag_pivot, on="customer_id", how="left")
                for col in tag_pivot.columns:
                    if col == "customer_id":
                        continue
                    features_pd[col] = features_pd[col].fillna(0.0)

        # --- Basket lift / affinity ---
        try:
            cfgmod = cfg.load_config()
            if cfgmod.features.use_market_basket:
                # Limit to feature window (e.g., last 12 months) and apply lag embargo to avoid adjacency
                try:
                    lag_days = int(getattr(cfgmod.features, 'affinity_lag_days', 60) or 60)
                except Exception:
                    lag_days = 60
                end_aff = cutoff_dt - pd.Timedelta(days=lag_days)
                last12_mask_lag = (fd['order_date'] > (cutoff_dt - pd.DateOffset(months=12))) & (fd['order_date'] <= end_aff)
                fd_win = fd.loc[last12_mask_lag, ['customer_id', 'product_sku', 'product_division']].dropna().copy()
                if not fd_win.empty:
                    fd_win['product_sku'] = fd_win['product_sku'].astype(str)
                    # Baseline: fraction of customers with target division activity in window
                    all_customers = set(fd_win['customer_id'].unique().tolist())
                    div_customers = set(fd_win.loc[fd_win['product_division'] == division_name, 'customer_id'].unique().tolist())
                    baseline = (len(div_customers) / max(1, len(all_customers)))
                    # Presence matrix per customer x SKU
                    has_sku = fd_win.drop_duplicates().assign(flag=1).pivot_table(index='customer_id', columns='product_sku', values='flag', fill_value=0)
                    # Compute lift per SKU with min support
                    min_support = 10
                    lift_weights = {}
                    supports = {}
                    # Use observed SKUs (cap to top-N by support to avoid blow-up)
                    sku_counts = has_sku.sum(axis=0).sort_values(ascending=False)
                    top_skus = sku_counts.index.tolist()
                    for s in top_skus:
                        supp = int(sku_counts.loc[s])
                        supports[s] = supp
                        if supp < min_support:
                            continue
                        custs = set(has_sku.index[has_sku[s] > 0].tolist())
                        inter = len(div_customers.intersection(custs))
                        p_cond = inter / max(1, supp)
                        lift = (p_cond / baseline) if baseline > 0 else 0.0
                        lift_weights[s] = float(lift)

                    # Aggregate to per-customer signals: max and mean lift of present SKUs (lagged exposure)
                    if lift_weights:
                        # Align DataFrame to lift columns only
                        cols = [c for c in has_sku.columns if c in lift_weights]
                        if cols:
                            w = pd.Series({c: lift_weights.get(c, 0.0) for c in cols})
                            present = has_sku[cols].astype(float)
                            weighted = present.mul(w, axis=1)
                            mb_lift_max = weighted.max(axis=1).rename(f'mb_lift_max_lag{lag_days}d')
                            # mean over present SKUs (avoid dividing by zero)
                            denom = present.sum(axis=1).replace(0.0, np.nan)
                            mb_lift_mean = (weighted.sum(axis=1) / denom).fillna(0.0).rename(f'mb_lift_mean_lag{lag_days}d')
                            mb_df = pd.concat([mb_lift_max, mb_lift_mean], axis=1).reset_index()
                            features_pd = features_pd.merge(mb_df, on='customer_id', how='left')
                            features_pd[f'mb_lift_max_lag{lag_days}d'] = features_pd[f'mb_lift_max_lag{lag_days}d'].fillna(0.0)
                            features_pd[f'mb_lift_mean_lag{lag_days}d'] = features_pd[f'mb_lift_mean_lag{lag_days}d'].fillna(0.0)

                        # Also export rule table for transparency
                        try:
                            rules = pd.DataFrame({
                                'sku': list(lift_weights.keys()),
                                'support': [supports.get(s, 0) for s in lift_weights.keys()],
                                'baseline_division_rate': baseline,
                                'lift': [lift_weights[s] for s in lift_weights.keys()],
                            })
                            rules.sort_values('lift', ascending=False).to_csv(
                                OUTPUTS_DIR / f"mb_rules_{division_name.lower()}_{(cutoff_date or '').replace('-', '')}.csv",
                                index=False
                            )
                        except Exception:
                            pass

                    # Backward-compatible aggregate affinity score
                    try:
                        if lift_weights and cols:
                            # Sum of lifts for present SKUs
                            affinity = weighted.sum(axis=1).rename(f'affinity__div__lift_topk__12m_lag{lag_days}d')
                            affinity_df = affinity.reset_index()
                            features_pd = features_pd.merge(affinity_df, on='customer_id', how='left')
                            coln = f'affinity__div__lift_topk__12m_lag{lag_days}d'
                            features_pd[coln] = features_pd[coln].fillna(0.0)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Basket lift computation failed: {e}")
    except Exception as e:
        # Non-fatal; proceed with base features if advanced features fail
        logger.warning(f"Advanced temporal features failed: {e}")

    # --- 3b. Asset features at cutoff (from Moneyball + item rollups) ---
    try:
        cfgmod = cfg.load_config()
        enabled = getattr(cfgmod.features, 'use_assets', None)
        # Determine build-on-demand preference (default True unless explicitly disabled)
        assets_cfg = getattr(cfgmod.features, 'assets', None)
        build_cfg = getattr(assets_cfg, 'build_on_demand', None) if assets_cfg is not None else None
        if build_cfg is None:
            build_cfg = getattr(cfgmod.features, 'build_assets_on_demand', None)
        build_on_demand = True if build_cfg is None else bool(build_cfg)
        # Avoid expensive builds for local SQLite engines unless explicitly overridden
        try:
            engine_is_sqlite = getattr(engine, 'dialect', None) and engine.dialect.name == 'sqlite'
        except Exception:
            engine_is_sqlite = False
        force_sqlite_assets = bool(getattr(cfgmod.features, 'force_assets_on_sqlite', False))
        if engine_is_sqlite and build_on_demand and not force_sqlite_assets:
            logger.info("Skipping asset build-on-demand for SQLite engine; set features.force_assets_on_sqlite=true to override.")
            build_on_demand = False
        # Default to ON when not explicitly disabled in config
        if cutoff_date and enabled is not False:
            logger.info("Asset features enabled (flag=%s); merging at cutoff %s", enabled, cutoff_date)
            # Read curated fact_assets; build on-demand if missing
            try:
                fact_assets_pd = pd.read_sql(select_all('fact_assets'), engine)
            except Exception:
                # Attempt to build and read again only if explicitly configured
                if build_on_demand:
                    try:
                        build_fact_assets(write=True)
                        fact_assets_pd = pd.read_sql(select_all('fact_assets'), engine)
                    except Exception as ee:
                        logger.warning(f"fact_assets unavailable: {ee}")
                        fact_assets_pd = pd.DataFrame()
                else:
                    logger.warning("fact_assets table not found and build_on_demand=False; skipping assets features")
                    fact_assets_pd = pd.DataFrame()

            if not fact_assets_pd.empty:
                from gosales.etl.assets import features_at_cutoff
                # Backward/forward compatible: function may return (roll, per) or (roll, per, extras)
                _out = features_at_cutoff(fact_assets_pd, cutoff_date)
                rollup_df = _out[0] if isinstance(_out, (list, tuple)) else _out
                per_df = _out[1] if isinstance(_out, (list, tuple)) and len(_out) > 1 else pd.DataFrame()
                extra_map = _out[2] if isinstance(_out, (list, tuple)) and len(_out) > 2 else {}

                # Pivot rollups into columns with safe names
                def safe(col: str) -> str:
                    return 'assets_rollup_' + str(col).strip().lower().replace(' ', '_').replace('/', '_')

                if not rollup_df.empty:
                    # Ensure consistent join key types
                    rollup_df['customer_id'] = rollup_df['customer_id'].astype(str)
                    features_pd['customer_id'] = features_pd['customer_id'].astype(str)
                    rollup_df.columns = [c if c == 'customer_id' else safe(c) for c in rollup_df.columns]
                    features_pd = features_pd.merge(rollup_df, on='customer_id', how='left')
                if not per_df.empty:
                    per_df['customer_id'] = per_df['customer_id'].astype(str)
                    features_pd['customer_id'] = features_pd['customer_id'].astype(str)
                    features_pd = features_pd.merge(per_df, on='customer_id', how='left')

                # Merge additional rollup frames (expiring windows, subs status)
                for key, df in (extra_map or {}).items():
                    if df is None or df.empty:
                        continue
                    df = df.copy()
                    df['customer_id'] = df['customer_id'].astype(str)
                    def prefix(col: str) -> str:
                        if col == 'customer_id':
                            return col
                        # key like 'expiring_30d' or 'on_subs' becomes assets_<key>_<rollup>
                        return f"assets_{key}_" + str(col).strip().lower().replace(' ', '_').replace('/', '_')
                    df.columns = [prefix(c) for c in df.columns]
                    features_pd = features_pd.merge(df, on='customer_id', how='left')

                # Fill NaNs for newly added features
                for c in features_pd.columns:
                    if c == 'customer_id':
                        continue
                    if features_pd[c].dtype.kind in 'fi':
                        features_pd[c] = features_pd[c].fillna(0.0)

                # Per-division ownership flags from assets at cutoff (active subs)
                try:
                    import re
                    # Build LUT from SKU mapping keys -> division
                    def _norm_key(s: str) -> str:
                        s = str(s or '').lower()
                        s = re.sub(r"[^0-9a-z]+", "_", s)
                        s = re.sub(r"_+", "_", s).strip('_')
                        return s
                    sku_map = get_sku_mapping() or {}
                    lut = {_norm_key(k): normalize_division(v.get('division')) for k, v in sku_map.items() if isinstance(v, dict) and 'division' in v}
                    # Heuristic fallback rules when rollup not in sku_map keys
                    def _heuristic_div(roll_norm: str) -> str | None:
                        r = roll_norm
                        # Solidworks core
                        if any(x in r for x in ("swx", "solidworks", "sw_core", "swx_core")):
                            return normalize_division("Solidworks")
                        # PDM/EPDM
                        if any(x in r for x in ("epdm", "pdm", "cad_editor")):
                            return normalize_division("PDM")
                        # Simulation
                        if "simulation" in r or "sim" in r:
                            return normalize_division("Simulation")
                        # CAM/CAMWorks
                        if "camworks" in r or r.startswith("cam"):
                            return normalize_division("CAMWorks")
                        # Electrical
                        if "electrical" in r or "schematic" in r:
                            return normalize_division("SW Electrical")
                        if any(x in r for x in ("training",)):
                            return normalize_division("Training")
                        if any(x in r for x in ("services",)):
                            return normalize_division("Services")
                        if any(x in r for x in ("success", "plan")):
                            return normalize_division("Success Plan")
                        if r.startswith("scan") or "scann" in r:
                            return normalize_division("Scanning")
                        # Printers/hardware ecosystem indicators
                        if any(x in r for x in ("fdm", "saf", "sla", "p3", "polyjet", "metals", "formlabs", "printer", "consumable", "spare", "repair", "am_", "3dp", "post_processing")):
                            return normalize_division("Hardware")
                        return None

                    # Collect assets_on_subs_* and assets_off_subs_* columns
                    on_cols = [c for c in features_pd.columns if c.startswith('assets_on_subs_')]
                    off_cols = [c for c in features_pd.columns if c.startswith('assets_off_subs_')]
                    add_cols: dict[str, pd.Series] = {}
                    if on_cols:
                        # Map each rollup column to a division
                        div_to_cols: dict[str, list[str]] = {}
                        for c in on_cols:
                            roll_norm = c.replace('assets_on_subs_', '')
                            div = lut.get(roll_norm)
                            if not div:
                                div = _heuristic_div(roll_norm)
                            if div:
                                div_to_cols.setdefault(div, []).append(c)
                        # Create boolean flags per division
                        for div, cols in div_to_cols.items():
                            if not cols:
                                continue
                            s = features_pd[cols].sum(axis=1)
                            colname = f"owns_assets_div_{normalize_division(div).lower()}"
                            add_cols[colname] = (pd.to_numeric(s, errors='coerce').fillna(0.0) > 0).astype('Int8')
                    # Former owner flags: any off_subs in division and no on_subs
                    if off_cols:
                        div_to_off: dict[str, list[str]] = {}
                        for c in off_cols:
                            roll_norm = c.replace('assets_off_subs_', '')
                            div = lut.get(roll_norm) or _heuristic_div(roll_norm)
                            if div:
                                div_to_off.setdefault(div, []).append(c)
                        # re-use on division to compare
                        div_to_on: dict[str, list[str]] = {}
                        for c in on_cols:
                            roll_norm = c.replace('assets_on_subs_', '')
                            div = lut.get(roll_norm) or _heuristic_div(roll_norm)
                            if div:
                                div_to_on.setdefault(div, []).append(c)
                        for div, off_list in div_to_off.items():
                            off_sum = features_pd[off_list].sum(axis=1)
                            on_sum = features_pd[div_to_on.get(div, [])].sum(axis=1) if div in div_to_on else 0
                            fname = f"former_owner_div_{normalize_division(div).lower()}"
                            add_cols[fname] = (
                                (pd.to_numeric(off_sum, errors='coerce').fillna(0.0) > 0)
                                & (pd.to_numeric(on_sum, errors='coerce').fillna(0.0) == 0)
                            ).astype('Int8')
                    if add_cols:
                        features_pd = pd.concat([features_pd, pd.DataFrame(add_cols, index=features_pd.index)], axis=1)
                except Exception as _e:
                    logger.warning(f"Per-division asset flags failed: {_e}")

                try:
                    added = [c for c in features_pd.columns if str(c).startswith('assets_')]
                    logger.info("Asset features added: %d", len(added))
                except Exception:
                    pass
                # Assets-based embedding fallback (i2v) for zero-ALS cohorts
                try:
                    from sklearn.decomposition import TruncatedSVD
                    # If ALS coverage is low or ALS columns missing, derive item2vec-like components from assets_rollup_*
                    als_cols_all = [c for c in features_pd.columns if str(c).startswith('als_f')]
                    als_cov = 0.0
                    if als_cols_all:
                        try:
                            als_cov = (features_pd[als_cols_all].abs().sum(axis=1) > 0).mean()
                        except Exception:
                            als_cov = 0.0
                    use_i2v = bool(getattr(cfgmod.features, 'use_item2vec', False))
                    try:
                        als_thr = float(getattr(getattr(cfgmod, 'whitespace', object()), 'als_coverage_threshold', 0.30))
                    except Exception:
                        als_thr = 0.30
                    roll_cols = [c for c in features_pd.columns if str(c).startswith('assets_rollup_')]
                    if roll_cols and (use_i2v or als_cov < als_thr):
                        R = features_pd[roll_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
                        ncomp = min(16, max(2, min(R.shape[0], R.shape[1]) - 1))
                        if ncomp >= 2:
                            svd = TruncatedSVD(n_components=ncomp, random_state=42)
                            Z = svd.fit_transform(R)
                            i2v = pd.DataFrame(Z, index=features_pd.index, columns=[f'i2v_f{i}' for i in range(ncomp)])
                            # Create i2v columns once; they are supplemental features for cold rows
                            for c in i2v.columns:
                                if c not in features_pd.columns:
                                    features_pd[c] = i2v[c].astype(float)
                except Exception as _e:
                    logger.warning(f"Assets-based i2v fallback failed: {_e}")

                # Optional Assets-ALS factorization (guarded for size)
                try:
                    use_assets_als = bool(getattr(cfgmod.features, 'use_assets_als', False))
                    roll_cols = [c for c in features_pd.columns if str(c).startswith('assets_rollup_')]
                    max_rows = int(getattr(getattr(cfgmod.features, 'assets_als', object()), 'max_rows', 20000)) if hasattr(cfgmod.features, 'assets_als') else 20000
                    max_cols = int(getattr(getattr(cfgmod.features, 'assets_als', object()), 'max_cols', 200)) if hasattr(cfgmod.features, 'assets_als') else 200
                    factors = int(getattr(getattr(cfgmod.features, 'assets_als', object()), 'factors', 16)) if hasattr(cfgmod.features, 'assets_als') else 16
                    iters = int(getattr(getattr(cfgmod.features, 'assets_als', object()), 'iters', 4)) if hasattr(cfgmod.features, 'assets_als') else 4
                    reg = float(getattr(getattr(cfgmod.features, 'assets_als', object()), 'reg', 0.1)) if hasattr(cfgmod.features, 'assets_als') else 0.1
                    if use_assets_als and roll_cols:
                        m, n = features_pd.shape[0], len(roll_cols)
                        if m <= max_rows and n <= max_cols and m > 2 and n > 2:
                            import numpy as _np
                            R = features_pd[roll_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=_np.float32)
                            f = min(factors, max(2, min(m, n) - 1))
                            rng = _np.random.default_rng(42)
                            X = rng.standard_normal((m, f), dtype=_np.float32) * 0.01
                            Y = rng.standard_normal((n, f), dtype=_np.float32) * 0.01
                            identity = _np.eye(f, dtype=_np.float32)
                            # ALS iterations with normal equations
                            for _ in range(iters):
                                # Update X (users)
                                YtY = Y.T @ Y + reg * identity
                                Yt = Y.T
                                X = _np.linalg.solve(YtY, (Yt @ R.T)).T
                                # Update Y (items)
                                XtX = X.T @ X + reg * identity
                                Xt = X.T
                                Y = _np.linalg.solve(XtX, (Xt @ R)).T
                            als_cols = {f'als_assets_f{i}': X[:, i] for i in range(f)}
                            features_pd = pd.concat([features_pd, pd.DataFrame(als_cols, index=features_pd.index)], axis=1)
                        else:
                            logger.info("Assets-ALS skipped due to matrix size (rows=%d, cols=%d)", m, n)
                except Exception as _e:
                    logger.warning(f"Assets-ALS factorization failed: {_e}")
    except Exception as e:
        logger.warning(f"Asset features failed: {e}")

    # Flags / indicators aggregated from the line-item fact
    try:
        agg_flags = _compute_flags_from_line_meta(line_meta_df, cutoff_date)

        if agg_flags.empty:
            logger.info("Flag aggregation skipped: no ACR/New sources available.")
        else:
            agg_flags['customer_id'] = agg_flags['customer_id'].astype(str)
            features_pd['customer_id'] = features_pd['customer_id'].astype(str)
            features_pd = features_pd.merge(agg_flags, on='customer_id', how='left')
            for c in ['ever_acr', 'ever_new_customer']:
                if c in features_pd.columns:
                    features_pd[c] = pd.to_numeric(features_pd[c], errors='coerce').fillna(0).astype(int)
                else:
                    features_pd[c] = 0
    except Exception as e:
        logger.warning(f"Flag aggregation failed: {e}")

    # Optionally join ALS embeddings (feature period <= cutoff)
    try:
        if cfgmod.features.use_als_embeddings and cutoff_date:
            # Guard on DB schema having 'quantity' to avoid costly failures
            db_has_quantity = False
            try:
                from sqlalchemy import inspect as _insp
                _cols = {c.get('name') for c in _insp(engine).get_columns('fact_transactions')}
                db_has_quantity = 'quantity' in (_cols or set())
            except Exception:
                db_has_quantity = False
            lag_days = 0
            try:
                lag_days = int(getattr(cfgmod.features, 'affinity_lag_days', 60) or 60)
            except Exception:
                lag_days = 60
            als_df = pl.DataFrame() if not db_has_quantity else customer_als_embeddings(
                engine,
                cutoff_date,
                factors=16,
                lookback_months=cfgmod.features.als_lookback_months,
                lag_days=lag_days,
            )
            if not als_df.is_empty():
                # Ensure type consistency before joining
                als_pd = als_df.to_pandas()
                # Ensure customer_id types match
                als_pd["customer_id"] = als_pd["customer_id"].astype(str)
                features_pd["customer_id"] = features_pd["customer_id"].astype(str)

                # Join on customer_id with consistent types
                features_pd = features_pd.merge(als_pd, on='customer_id', how='left')
                for c in [c for c in features_pd.columns if str(c).startswith('als_f')]:
                    features_pd[c] = pd.to_numeric(features_pd[c], errors='coerce').fillna(0.0)
    except Exception as e:
        logger.warning(f"ALS embedding join failed (non-blocking): {e}")

    # Convert back to polars
    features = pl.from_pandas(features_pd)

    # --- 4. Combine Features and Target ---
    # Start with all customers, then left-join the features and the target variable.
    # Align join key dtypes explicitly
    features = features.with_columns(pl.col("customer_id").cast(pl.Utf8) if "customer_id" in features.columns else pl.lit(None))
    customers = customers.with_columns(pl.col("customer_id").cast(pl.Utf8))

    feature_matrix = (
        customers.lazy()
        .join(features.lazy(), on="customer_id", how="left")
        .join(division_buyers, on="customer_id", how="left")
        .with_columns([
            pl.col("bought_in_division").fill_null(0).cast(pl.Int8),
        ])
        .collect()
    )

    # Fast-path: on large SQLite datasets, return minimal feature matrix to cap memory/time (configurable)
    try:
        _is_sqlite = getattr(engine, 'dialect', None) and engine.dialect.name == 'sqlite'
    except Exception:
        _is_sqlite = False
    if _is_sqlite:
        try:
            fast_thr = int(getattr(cfg.load_config().features, 'fastpath_minimal_return_rows', 10_000_000))
        except Exception:
            fast_thr = 10_000_000
        if fast_thr <= 0:
            effective_fast_thr = float("inf")
        else:
            # In test/local SQLite with configured external DB, apply a conservative cap to keep memory bounded
            try:
                configured_engine = str(getattr(cfg.load_config().database, 'engine', '')).strip().lower()
            except Exception:
                configured_engine = ''
            effective_fast_thr = fast_thr
            if configured_engine and configured_engine != 'sqlite':
                effective_fast_thr = min(fast_thr, 50_000)
        try:
            nrows_fd = len(feature_data)
        except Exception:
            nrows_fd = None
        if nrows_fd is not None and nrows_fd > effective_fast_thr:
            logger.warning(
                "Returning minimal feature matrix: rows=%d exceeds limit=%d (fastpath_minimal_return_rows=%d)",
                nrows_fd, effective_fast_thr, fast_thr,
            )
            return feature_matrix

    # Fill nulls for all other columns in pandas for easier handling
    # Drop datetime columns before materializing to Pandas to limit memory footprint
    date_columns = [col for col, dtype in feature_matrix.schema.items() if dtype in (pl.Date, pl.Datetime)]
    if date_columns:
        feature_matrix = feature_matrix.drop(date_columns)

    # Downcast wide numeric blocks to float32/int32 to reduce memory usage prior to pandas conversion
    feature_matrix = feature_matrix.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32),
    ])

    feature_matrix_pd = feature_matrix.to_pandas()

    # Optional: augment with cutoff-safe NetSuite customer features (territory/contact/etc.)
    try:
        from gosales.features.ns_customer_features import build_ns_customer_features
        cutoff_for_ns = cutoff_date if cutoff_date else pd.Timestamp.today().date().isoformat()
        ns_pl = build_ns_customer_features(cutoff_for_ns)
        ns_pd = ns_pl.to_pandas()
        if not ns_pd.empty:
            feature_matrix_pd = feature_matrix_pd.merge(ns_pd, on="customer_id", how="left")
            # Handle nulls by family
            cust_cat = [c for c in feature_matrix_pd.columns if c.startswith("cust_cat_")]
            cust_feat = [c for c in feature_matrix_pd.columns if c.startswith("cust_feat_")]
            for c in cust_cat:
                feature_matrix_pd[c] = feature_matrix_pd[c].fillna("missing").astype(str)
            for c in cust_feat:
                feature_matrix_pd[c] = pd.to_numeric(feature_matrix_pd[c], errors='coerce').fillna(0)
    except Exception as _e:
        # Non-fatal; continue without NS augmentation
        pass
    
    # Drop the date columns that are no longer needed for ML
    date_columns = [col for col in feature_matrix_pd.columns if 'date' in col.lower()]
    if date_columns:
        feature_matrix_pd.drop(columns=date_columns, inplace=True, errors="ignore")
    
    # Note: Defer global fill until after missingness flags are added
    # so `_missing` indicators reflect original NaNs.
    # Map to naming scheme for key features
    # Recency
    if 'days_since_last_order' in feature_matrix_pd.columns:
        feature_matrix_pd['rfm__all__recency_days__life'] = feature_matrix_pd['days_since_last_order']
    div_rec_col = f'days_since_last_{division_name}_order'
    if div_rec_col in feature_matrix_pd.columns:
        feature_matrix_pd['rfm__div__recency_days__life'] = feature_matrix_pd[div_rec_col]
    # Cycle-aware transforms: log-recency and hazard/decay with configurable half-lives
    try:
        # Log-recency
        if 'rfm__all__recency_days__life' in feature_matrix_pd.columns:
            feature_matrix_pd['rfm__all__log_recency__life'] = np.log1p(pd.to_numeric(feature_matrix_pd['rfm__all__recency_days__life'], errors='coerce').fillna(999.0))
        if 'rfm__div__recency_days__life' in feature_matrix_pd.columns:
            feature_matrix_pd['rfm__div__log_recency__life'] = np.log1p(pd.to_numeric(feature_matrix_pd['rfm__div__recency_days__life'], errors='coerce').fillna(999.0))
        # Hazard/decay transforms
        half_lives = list(cfg.load_config().features.recency_decay_half_lives_days or [30, 90, 180])
        for hl in half_lives:
            try:
                hl = float(hl) if float(hl) > 0 else 30.0
            except Exception:
                hl = 30.0
            if 'rfm__all__recency_days__life' in feature_matrix_pd.columns:
                d = pd.to_numeric(feature_matrix_pd['rfm__all__recency_days__life'], errors='coerce').fillna(999.0)
                feature_matrix_pd[f'rfm__all__recency_decay__hl{int(hl)}'] = np.exp(-d / hl)
            if 'rfm__div__recency_days__life' in feature_matrix_pd.columns:
                dd = pd.to_numeric(feature_matrix_pd['rfm__div__recency_days__life'], errors='coerce').fillna(999.0)
                feature_matrix_pd[f'rfm__div__recency_decay__hl{int(hl)}'] = np.exp(-dd / hl)
    except Exception:
        pass
    # RFM windows (all scope)
    for w in window_months:
        if f'tx_count_last_{w}m' in feature_matrix_pd.columns:
            feature_matrix_pd[f'rfm__all__tx_n__{w}m'] = feature_matrix_pd[f'tx_count_last_{w}m']
        if f'gp_sum_last_{w}m' in feature_matrix_pd.columns:
            feature_matrix_pd[f'rfm__all__gp_sum__{w}m'] = feature_matrix_pd[f'gp_sum_last_{w}m']
        if f'gp_mean_last_{w}m' in feature_matrix_pd.columns:
            feature_matrix_pd[f'rfm__all__gp_mean__{w}m'] = feature_matrix_pd[f'gp_mean_last_{w}m']
    # Fallback: if key RFM columns are missing, recompute directly from fd
    try:
        for w in window_months:
            start_dt = cutoff_dt - pd.DateOffset(months=w)
            mask = (fd['order_date'] > start_dt) & (fd['order_date'] <= cutoff_dt)
            sub = fd.loc[mask, ['customer_id', 'gross_profit']]
            # tx_n
            if f'rfm__all__tx_n__{w}m' not in feature_matrix_pd.columns:
                tx_n = sub.groupby('customer_id')['gross_profit'].count().rename(f'rfm__all__tx_n__{w}m').reset_index()
                feature_matrix_pd = feature_matrix_pd.merge(tx_n, on='customer_id', how='left')
                feature_matrix_pd[f'rfm__all__tx_n__{w}m'] = feature_matrix_pd[f'rfm__all__tx_n__{w}m'].fillna(0)
            # gp_sum
            if f'rfm__all__gp_sum__{w}m' not in feature_matrix_pd.columns:
                gp_sum = sub.groupby('customer_id')['gross_profit'].sum().rename(f'rfm__all__gp_sum__{w}m').reset_index()
                feature_matrix_pd = feature_matrix_pd.merge(gp_sum, on='customer_id', how='left')
                feature_matrix_pd[f'rfm__all__gp_sum__{w}m'] = feature_matrix_pd[f'rfm__all__gp_sum__{w}m'].fillna(0.0)
            # gp_mean
            if f'rfm__all__gp_mean__{w}m' not in feature_matrix_pd.columns:
                gp_mean = sub.groupby('customer_id')['gross_profit'].mean().rename(f'rfm__all__gp_mean__{w}m').reset_index()
                feature_matrix_pd = feature_matrix_pd.merge(gp_mean, on='customer_id', how='left')
                feature_matrix_pd[f'rfm__all__gp_mean__{w}m'] = feature_matrix_pd[f'rfm__all__gp_mean__{w}m'].fillna(0.0)
    except Exception:
        pass

    # Window deltas: 12m vs previous 12m (from 24m)
    try:
        cfgf = cfg.load_config().features
        if bool(getattr(cfgf, 'enable_window_deltas', True)):
            # All-scope deltas
            if all(c in feature_matrix_pd.columns for c in ['rfm__all__gp_sum__12m', 'rfm__all__gp_sum__24m']):
                last12 = pd.to_numeric(feature_matrix_pd['rfm__all__gp_sum__12m'], errors='coerce').fillna(0.0)
                tot24 = pd.to_numeric(feature_matrix_pd['rfm__all__gp_sum__24m'], errors='coerce').fillna(0.0)
                prev12 = (tot24 - last12).clip(lower=0.0)
                feature_matrix_pd['rfm__all__gp_sum__delta_12m_prev12m'] = last12 - prev12
                feature_matrix_pd['rfm__all__gp_sum__ratio_12m_prev12m'] = (last12 / (prev12 + 1e-9)).replace([np.inf, -np.inf], 0.0)
            if all(c in feature_matrix_pd.columns for c in ['rfm__all__tx_n__12m', 'rfm__all__tx_n__24m']):
                last12 = pd.to_numeric(feature_matrix_pd['rfm__all__tx_n__12m'], errors='coerce').fillna(0.0)
                tot24 = pd.to_numeric(feature_matrix_pd['rfm__all__tx_n__24m'], errors='coerce').fillna(0.0)
                prev12 = (tot24 - last12).clip(lower=0.0)
                feature_matrix_pd['rfm__all__tx_n__delta_12m_prev12m'] = last12 - prev12
                feature_matrix_pd['rfm__all__tx_n__ratio_12m_prev12m'] = (last12 / (prev12 + 1e-9)).replace([np.inf, -np.inf], 0.0)
            # Division-scope deltas
            if all(c in feature_matrix_pd.columns for c in ['rfm__div__gp_sum__12m', 'rfm__div__gp_sum__24m']):
                last12 = pd.to_numeric(feature_matrix_pd['rfm__div__gp_sum__12m'], errors='coerce').fillna(0.0)
                tot24 = pd.to_numeric(feature_matrix_pd['rfm__div__gp_sum__24m'], errors='coerce').fillna(0.0)
                prev12 = (tot24 - last12).clip(lower=0.0)
                feature_matrix_pd['rfm__div__gp_sum__delta_12m_prev12m'] = last12 - prev12
                feature_matrix_pd['rfm__div__gp_sum__ratio_12m_prev12m'] = (last12 / (prev12 + 1e-9)).replace([np.inf, -np.inf], 0.0)
            if all(c in feature_matrix_pd.columns for c in ['rfm__div__tx_n__12m', 'rfm__div__tx_n__24m']):
                last12 = pd.to_numeric(feature_matrix_pd['rfm__div__tx_n__12m'], errors='coerce').fillna(0.0)
                tot24 = pd.to_numeric(feature_matrix_pd['rfm__div__tx_n__24m'], errors='coerce').fillna(0.0)
                prev12 = (tot24 - last12).clip(lower=0.0)
                feature_matrix_pd['rfm__div__tx_n__delta_12m_prev12m'] = last12 - prev12
                feature_matrix_pd['rfm__div__tx_n__ratio_12m_prev12m'] = (last12 / (prev12 + 1e-9)).replace([np.inf, -np.inf], 0.0)
    except Exception:
        pass

    # Fallback: ensure margin proxy columns exist based on rfm__all__gp_sum__{w}m
    try:
        for w in window_months:
            col_sum = f'rfm__all__gp_sum__{w}m'
            col_margin = f'margin__all__gp_pct__{w}m'
            if (col_sum in feature_matrix_pd.columns) and (col_margin not in feature_matrix_pd.columns):
                s = pd.to_numeric(feature_matrix_pd[col_sum], errors='coerce').fillna(0.0)
                feature_matrix_pd[col_margin] = s.astype(float) / (s.abs().astype(float) + 1e-9)
    except Exception:
        pass
    # Lifecycle naming
    if 'tenure_days' in feature_matrix_pd.columns:
        feature_matrix_pd['lifecycle__all__tenure_days__life'] = feature_matrix_pd['tenure_days']
        # Tenure months and buckets for cycle awareness
        try:
            t = pd.to_numeric(feature_matrix_pd['tenure_days'], errors='coerce').fillna(0.0)
            feature_matrix_pd['lifecycle__all__tenure_months__life'] = (t / 30.0).astype(float)
            # Buckets: <3m, 3-6m, 6-12m, 1-2y, >=2y
            bins = [0, 90, 180, 365, 730, np.inf]
            labels = ['lt3m','3to6m','6to12m','1to2y','ge2y']
            b = pd.cut(t, bins=bins, labels=labels, right=True, include_lowest=True)
            for lab in labels:
                feature_matrix_pd[f'lifecycle__all__tenure_bucket__{lab}'] = (b == lab).astype(int)
        except Exception:
            pass
    if 'last_gap_days' in feature_matrix_pd.columns:
        feature_matrix_pd['lifecycle__all__gap_days__life'] = feature_matrix_pd['last_gap_days']
    # Diversity/division counts
    try:
        last12_mask = (fd['order_date'] > (cutoff_dt - pd.DateOffset(months=12))) & (fd['order_date'] <= cutoff_dt)
        dl = fd.loc[last12_mask, ['customer_id', 'product_division', 'product_sku']].copy()
        div_n = dl.groupby('customer_id')['product_division'].nunique().rename('xdiv__all__division_nunique__12m').reset_index()
        sku_all = dl.groupby('customer_id')['product_sku'].nunique().rename('diversity__all__sku_nunique__12m').reset_index()
        sku_div = dl.loc[dl['product_division'] == division_name].groupby('customer_id')['product_sku'].nunique().rename('diversity__div__sku_nunique__12m').reset_index()
        feature_matrix_pd = feature_matrix_pd.merge(div_n, on='customer_id', how='left').merge(sku_all, on='customer_id', how='left').merge(sku_div, on='customer_id', how='left')
        for col in ['xdiv__all__division_nunique__12m','diversity__all__sku_nunique__12m','diversity__div__sku_nunique__12m']:
            feature_matrix_pd[col] = feature_matrix_pd[col].fillna(0)
    except Exception:
        pass
    # Seasonality renames
    for q in ['q1','q2','q3','q4']:
        col = f'{q}_share_24m'
        if col in feature_matrix_pd.columns:
            feature_matrix_pd[f'season__all__{q}_share__24m'] = feature_matrix_pd[col]

    # Returns features (12m, target division)
    try:
        last12_mask = (fd['order_date'] > (cutoff_dt - pd.DateOffset(months=12))) & (fd['order_date'] <= cutoff_dt)
        sub12 = fd.loc[last12_mask & (fd['product_division'].astype(str).str.strip() == norm_division_name), ['customer_id', 'gross_profit']].copy()
        sub12['is_return'] = (pd.to_numeric(sub12['gross_profit'], errors='coerce') < 0).astype(int)
        ret_counts = sub12.groupby('customer_id')['is_return'].sum().rename('returns__div__return_tx_n__12m').reset_index()
        tx_counts = sub12.groupby('customer_id')['is_return'].count().rename('tx_n_12m_div').reset_index()
        ret = ret_counts.merge(tx_counts, on='customer_id', how='left')
        ret['returns__div__return_rate__12m'] = ret['returns__div__return_tx_n__12m'] / ret['tx_n_12m_div'].replace(0, np.nan)
        ret['returns__div__return_rate__12m'] = ret['returns__div__return_rate__12m'].fillna(0.0)
        feature_matrix_pd = feature_matrix_pd.merge(ret[['customer_id','returns__div__return_tx_n__12m','returns__div__return_rate__12m']], on='customer_id', how='left')
        feature_matrix_pd['returns__div__return_tx_n__12m'] = feature_matrix_pd['returns__div__return_tx_n__12m'].fillna(0)
        feature_matrix_pd['returns__div__return_rate__12m'] = feature_matrix_pd['returns__div__return_rate__12m'].fillna(0.0)
    except Exception:
        pass

    # Returns features (12m, all divisions)
    try:
        last12_mask = (fd['order_date'] > (cutoff_dt - pd.DateOffset(months=12))) & (fd['order_date'] <= cutoff_dt)
        sub12a = fd.loc[last12_mask, ['customer_id', 'gross_profit']].copy()
        sub12a['is_return'] = (pd.to_numeric(sub12a['gross_profit'], errors='coerce') < 0).astype(int)
        ret_counts_a = sub12a.groupby('customer_id')['is_return'].sum().rename('returns__all__return_tx_n__12m').reset_index()
        tx_counts_a = sub12a.groupby('customer_id')['is_return'].count().rename('tx_n_12m_all').reset_index()
        reta = ret_counts_a.merge(tx_counts_a, on='customer_id', how='left')
        reta['returns__all__return_rate__12m'] = reta['returns__all__return_tx_n__12m'] / reta['tx_n_12m_all'].replace(0, np.nan)
        reta['returns__all__return_rate__12m'] = reta['returns__all__return_rate__12m'].fillna(0.0)
        feature_matrix_pd = feature_matrix_pd.merge(reta[['customer_id','returns__all__return_tx_n__12m','returns__all__return_rate__12m']], on='customer_id', how='left')
        feature_matrix_pd['returns__all__return_tx_n__12m'] = feature_matrix_pd['returns__all__return_tx_n__12m'].fillna(0)
        feature_matrix_pd['returns__all__return_rate__12m'] = feature_matrix_pd['returns__all__return_rate__12m'].fillna(0.0)
    except Exception:
        pass

    # Diversity across windows: SKU nunique (all/div)
    try:
        for w in window_months:
            maskw = (fd['order_date'] > (cutoff_dt - pd.DateOffset(months=w))) & (fd['order_date'] <= cutoff_dt)
            dlw = fd.loc[maskw, ['customer_id', 'product_division', 'product_sku']].copy()
            if not dlw.empty:
                sku_all_w = dlw.groupby('customer_id')['product_sku'].nunique().rename(f'diversity__all__sku_nunique__{w}m').reset_index()
                dlw['product_division'] = dlw['product_division'].astype(str).str.strip()
                sku_div_w = dlw.loc[dlw['product_division'] == norm_division_name].groupby('customer_id')['product_sku'].nunique().rename(f'diversity__div__sku_nunique__{w}m').reset_index()
                feature_matrix_pd = feature_matrix_pd.merge(sku_all_w, on='customer_id', how='left').merge(sku_div_w, on='customer_id', how='left')
                feature_matrix_pd[f'diversity__all__sku_nunique__{w}m'] = feature_matrix_pd[f'diversity__all__sku_nunique__{w}m'].fillna(0)
                feature_matrix_pd[f'diversity__div__sku_nunique__{w}m'] = feature_matrix_pd[f'diversity__div__sku_nunique__{w}m'].fillna(0)
    except Exception:
        pass

    # Winsorize monetary gp_sum features based on config (stable quantiles)
    try:
        p = cfg.load_config().features.gp_winsor_p
        for w in window_months:
            for scope in ['all','div']:
                col = f'rfm__{scope}__gp_sum__{w}m'
                if col in feature_matrix_pd.columns:
                    s = pd.to_numeric(feature_matrix_pd[col], errors='coerce').fillna(0.0)
                    lower = float(np.quantile(s.values, 0.0))
                    upper = float(np.quantile(s.values, p))
                    feature_matrix_pd[col] = s.clip(lower=lower, upper=upper)
    except Exception:
        pass
    # Add missingness flags if configured (single concat to avoid fragmentation)
    try:
        if cfg.load_config().features.add_missingness_flags:
            cols = [c for c in feature_matrix_pd.columns if c not in ('customer_id','bought_in_division')]
            if cols:
                # Compute NaN mask just-in-time before any global fill, so flags
                # represent original missingness.
                missing_mask = feature_matrix_pd[cols].isna()
                flags_df = missing_mask.astype(np.int8)
                flags_df.columns = [f"{c}_missing" for c in flags_df.columns]
                feature_matrix_pd = pd.concat([feature_matrix_pd, flags_df], axis=1)
    except Exception:
        pass

    # Fill nulls globally after flags are created to ensure downstream typing
    # Preserve recency semantics: missing recency should be large (not zero), so warm gating remains correct.
    try:
        # First, explicitly fill recency-related columns with a high sentinel
        recency_cols = [
            'rfm__all__recency_days__life',
            'rfm__div__recency_days__life',
            'days_since_last_order',
            f'days_since_last_{division_name}_order',
        ]
        for col in recency_cols:
            if col in feature_matrix_pd.columns:
                feature_matrix_pd[col] = pd.to_numeric(feature_matrix_pd[col], errors='coerce').fillna(999.0)
        # Then, fill remaining NaNs with 0 to preserve prior behavior for other features
        feature_matrix_pd = feature_matrix_pd.fillna(0)
    except Exception:
        pass
    
    # Convert back to polars
    feature_matrix = pl.from_pandas(feature_matrix_pd)

    # Join with customer industry data
    try:
        logger.info("Joining industry data with feature matrix...")
        customers_with_industry_pd = pd.read_sql("""
            SELECT customer_id, industry, industry_sub 
            FROM dim_customer 
            WHERE industry IS NOT NULL
        """, engine)
        
        if not customers_with_industry_pd.empty:
            # Normalise text
            customers_with_industry_pd['industry'] = customers_with_industry_pd['industry'].astype(str).str.strip()
            customers_with_industry_pd['industry_sub'] = customers_with_industry_pd['industry_sub'].astype(str).str.strip()

            # Top-N categories
            top_industries = customers_with_industry_pd['industry'].value_counts().head(20).index.tolist()
            top_subs = customers_with_industry_pd['industry_sub'].value_counts().head(30).index.tolist()

            # Helper to sanitize feature names for LightGBM (alnum + underscore only)
            import re
            def sanitize_key(text: str) -> str:
                if text is None:
                    return "unknown"
                key = str(text).lower()
                key = key.replace("&", " and ")
                key = re.sub(r"[^0-9a-zA-Z]+", "_", key)
                key = re.sub(r"_+", "_", key).strip("_")
                if not key:
                    key = "unknown"
                return key

            # Industry dummies
            industry_key_map = {industry: sanitize_key(industry) for industry in top_industries}
            for industry, key in industry_key_map.items():
                customers_with_industry_pd[f"is_{key}"] = (customers_with_industry_pd['industry'] == industry).astype(int)

            # Sub-industry dummies
            sub_key_map = {sub: sanitize_key(sub) for sub in top_subs}
            for sub, key in sub_key_map.items():
                customers_with_industry_pd[f"is_sub_{key}"] = (customers_with_industry_pd['industry_sub'] == sub).astype(int)

            # Interaction examples: industry × services engagement will be created post-join

            # Convert to polars and join
            industry_features = pl.from_pandas(customers_with_industry_pd)
            # Ensure string columns are Utf8 to avoid PyString/Arrow warnings downstream
            try:
                industry_features = industry_features.with_columns([
                    pl.col("industry").cast(pl.Utf8, strict=False),
                    pl.col("industry_sub").cast(pl.Utf8, strict=False),
                ])
            except Exception:
                pass
            feature_columns = ["customer_id","industry","industry_sub"] + \
                [f"is_{industry_key_map[i]}" for i in top_industries] + \
                [f"is_sub_{sub_key_map[s]}" for s in top_subs]

            feature_matrix = feature_matrix.join(
                industry_features.select(feature_columns),
                on="customer_id",
                how="left"
            ).fill_null(0)
            try:
                # Re-assert Utf8 after join in case engine conversion introduced Arrow-backed strings
                feature_matrix = feature_matrix.with_columns([
                    pl.col("industry").cast(pl.Utf8, strict=False),
                    pl.col("industry_sub").cast(pl.Utf8, strict=False),
                ])
            except Exception:
                pass
            
            logger.info(f"Successfully joined industry data. Added {len(top_industries)} industry and {len(top_subs)} sub-industry dummies.")

            # Pooled/hierarchical encoders for sparse industries (non-leaky: pre-cutoff history only)
            try:
                cfgf = cfg.load_config().features
                if bool(getattr(cfgf, 'pooled_encoders_enable', True)):
                    lookback_m = int(getattr(cfgf, 'pooled_encoders_lookback_months', 24))
                    alpha_ind = float(getattr(cfgf, 'pooled_alpha_industry', 50.0))
                    alpha_sub = float(getattr(cfgf, 'pooled_alpha_sub', 50.0))
                    # Build last-<lookback_m> months transaction slice
                    fd2 = feature_data.copy()
                    fd2['order_date'] = pd.to_datetime(fd2['order_date'])
                    start_lb = cutoff_dt - pd.DateOffset(months=lookback_m)
                    mask_lb = (fd2['order_date'] > start_lb) & (fd2['order_date'] <= cutoff_dt)
                    hist = fd2.loc[mask_lb, ['customer_id','order_date','product_division','product_sku','gross_profit']].copy()
                    # Attach industry/sub to each transaction
                    cust_ind = customers_with_industry_pd[['customer_id','industry','industry_sub']].copy()
                    hist = hist.merge(cust_ind, on='customer_id', how='left')
                    # Target membership per tx
                    if use_custom_targets:
                        hist['is_target'] = hist['product_sku'].astype(str).isin(target_skus)
                    else:
                        hist['is_target'] = hist['product_division'].astype(str).str.strip() == norm_division_name
                    # Industry-level counts and rates
                    ind_tx = hist.groupby('industry', dropna=False)['order_date'].count().rename('tx_n').reset_index()
                    ind_ttx = hist.loc[hist['is_target']].groupby('industry', dropna=False)['order_date'].count().rename('tx_n_target').reset_index()
                    ind_gp = hist.groupby('industry', dropna=False)['gross_profit'].sum().rename('gp_sum').reset_index()
                    ind_tgp = hist.loc[hist['is_target']].groupby('industry', dropna=False)['gross_profit'].sum().rename('gp_sum_target').reset_index()
                    ind = ind_tx.merge(ind_ttx, on='industry', how='left').merge(ind_gp, on='industry', how='left').merge(ind_tgp, on='industry', how='left').fillna(0.0)
                    # Global priors
                    g_tx = float(ind['tx_n'].sum())
                    g_ttx = float(ind['tx_n_target'].sum())
                    g_gp = float(ind['gp_sum'].sum())
                    g_tgp = float(ind['gp_sum_target'].sum())
                    p_tx_global = (g_ttx / g_tx) if g_tx > 0 else 0.0
                    p_gp_global = (g_tgp / g_gp) if g_gp > 0 else 0.0
                    # Smoothed industry encoders
                    ind['enc__industry__tx_rate_24m_smooth'] = (ind['tx_n_target'] + alpha_ind * p_tx_global) / (ind['tx_n'] + alpha_ind)
                    # For GP share smoothing, weight prior by magnitude to be on same scale
                    ind['enc__industry__gp_share_24m_smooth'] = (ind['gp_sum_target'] + alpha_ind * p_gp_global * ind['gp_sum']) / (ind['gp_sum'] + alpha_ind)
                    ind_enc = ind[['industry','enc__industry__tx_rate_24m_smooth','enc__industry__gp_share_24m_smooth']]
                    # Sub-industry encoders with hierarchical shrink to parent industry
                    sub_tx = hist.groupby('industry_sub', dropna=False)['order_date'].count().rename('tx_n').reset_index()
                    sub_ttx = hist.loc[hist['is_target']].groupby('industry_sub', dropna=False)['order_date'].count().rename('tx_n_target').reset_index()
                    sub_gp = hist.groupby('industry_sub', dropna=False)['gross_profit'].sum().rename('gp_sum').reset_index()
                    sub_tgp = hist.loc[hist['is_target']].groupby('industry_sub', dropna=False)['gross_profit'].sum().rename('gp_sum_target').reset_index()
                    sub = sub_tx.merge(sub_ttx, on='industry_sub', how='left').merge(sub_gp, on='industry_sub', how='left').merge(sub_tgp, on='industry_sub', how='left').fillna(0.0)
                    # Map sub -> dominant parent industry
                    parents = hist.groupby('industry_sub')['industry'].agg(lambda s: s.value_counts().index[0] if not s.value_counts().empty else None).reset_index().rename(columns={'industry':'parent_industry'})
                    sub = sub.merge(parents, on='industry_sub', how='left')
                    sub = sub.merge(ind_enc, left_on='parent_industry', right_on='industry', how='left', suffixes=('','_parent'))
                    sub['p_tx_parent'] = sub['enc__industry__tx_rate_24m_smooth'].fillna(p_tx_global)
                    sub['p_gp_parent'] = sub['enc__industry__gp_share_24m_smooth'].fillna(p_gp_global)
                    sub['enc__industry_sub__tx_rate_24m_smooth'] = (sub['tx_n_target'] + alpha_sub * sub['p_tx_parent']) / (sub['tx_n'] + alpha_sub)
                    sub['enc__industry_sub__gp_share_24m_smooth'] = (sub['gp_sum_target'] + alpha_sub * sub['p_gp_parent'] * sub['gp_sum']) / (sub['gp_sum'] + alpha_sub)
                    sub_enc = sub[['industry_sub','enc__industry_sub__tx_rate_24m_smooth','enc__industry_sub__gp_share_24m_smooth']]
                    # Join encoders onto feature matrix by industry keys
                    # Ensure string dtypes on both sides and numeric encoder columns to avoid conversion errors
                    if 'industry' in feature_matrix.columns:
                        feature_matrix = feature_matrix.with_columns(
                            pl.col('industry').cast(pl.Utf8, strict=False).fill_null("")
                        )
                    if 'industry_sub' in feature_matrix.columns:
                        feature_matrix = feature_matrix.with_columns(
                            pl.col('industry_sub').cast(pl.Utf8, strict=False).fill_null("")
                        )

                    ind_enc_pl = pl.from_pandas(ind_enc)
                    if 'industry' in ind_enc_pl.columns:
                        ind_enc_pl = ind_enc_pl.with_columns(pl.col('industry').cast(pl.Utf8, strict=False))
                        ind_enc_pl = ind_enc_pl.with_columns(pl.col('industry').fill_null(""))
                        # Cast all non-key columns to Float64
                        for _c in [c for c in ind_enc_pl.columns if c != 'industry']:
                            ind_enc_pl = ind_enc_pl.with_columns(pl.col(_c).cast(pl.Float64, strict=False))

                    sub_enc_pl = pl.from_pandas(sub_enc)
                    if 'industry_sub' in sub_enc_pl.columns:
                        sub_enc_pl = sub_enc_pl.with_columns(pl.col('industry_sub').cast(pl.Utf8, strict=False))
                        sub_enc_pl = sub_enc_pl.with_columns(pl.col('industry_sub').fill_null(""))
                        for _c in [c for c in sub_enc_pl.columns if c != 'industry_sub']:
                            sub_enc_pl = sub_enc_pl.with_columns(pl.col(_c).cast(pl.Float64, strict=False))

                    if 'industry' in feature_matrix.columns and 'industry' in ind_enc_pl.columns:
                        feature_matrix = feature_matrix.join(ind_enc_pl, on='industry', how='left')
                    if 'industry_sub' in feature_matrix.columns and 'industry_sub' in sub_enc_pl.columns:
                        feature_matrix = feature_matrix.join(sub_enc_pl, on='industry_sub', how='left')
            except Exception as e:
                try:
                    _silence = bool(getattr(cfg.load_config().features, 'silence_pooled_encoder_warnings', False))
                except Exception:
                    _silence = False
                msg = f"Pooled encoder features failed (non-blocking): {e}"
                if _silence:
                    try:
                        logger.debug(msg)
                    except Exception:
                        pass
                else:
                    logger.warning(msg)
        else:
            logger.warning("No industry data available for joining.")
            
    except Exception as e:
        logger.warning(f"Could not join industry data: {e}")

    # Example interaction features (post-join) if present
    try:
        interaction_cols = [c for c in feature_matrix.columns if c.startswith("is_") or c.startswith("is_sub_")]
        # Normalizers and derived metrics
        max_services = float(feature_matrix["total_services_gp"].max()) if "total_services_gp" in feature_matrix.columns else 1.0
        max_avg_gp = float(feature_matrix["avg_transaction_gp"].max()) if "avg_transaction_gp" in feature_matrix.columns else 1.0
        max_diversity = float(feature_matrix["product_diversity_score"].max()) if "product_diversity_score" in feature_matrix.columns else 1.0
        # Growth ratio
        if "gp_2024" in feature_matrix.columns and "gp_2023" in feature_matrix.columns:
            feature_matrix = feature_matrix.with_columns(
                (feature_matrix["gp_2024"].cast(pl.Float64) / (feature_matrix["gp_2023"].cast(pl.Float64) + 1.0)).alias("growth_ratio_24_over_23")
            )
        # Build interactions (limit to first 12 flags to control dimensionality)
        if interaction_cols:
            if "total_services_gp" in feature_matrix.columns:
                svc_norm = feature_matrix["total_services_gp"].cast(pl.Float64) / (max_services or 1.0)
                for c in interaction_cols[:12]:
                    feature_matrix = feature_matrix.with_columns((svc_norm * feature_matrix[c].cast(pl.Float64)).alias(f"{c}_x_services"))
            if "avg_transaction_gp" in feature_matrix.columns:
                avg_norm = feature_matrix["avg_transaction_gp"].cast(pl.Float64) / (max_avg_gp or 1.0)
                for c in interaction_cols[:12]:
                    feature_matrix = feature_matrix.with_columns((avg_norm * feature_matrix[c].cast(pl.Float64)).alias(f"{c}_x_avg_gp"))
            if "product_diversity_score" in feature_matrix.columns:
                div_norm = feature_matrix["product_diversity_score"].cast(pl.Float64) / (max_diversity or 1.0)
                for c in interaction_cols[:12]:
                    feature_matrix = feature_matrix.with_columns((div_norm * feature_matrix[c].cast(pl.Float64)).alias(f"{c}_x_diversity"))
            if "growth_ratio_24_over_23" in feature_matrix.columns:
                gr = feature_matrix["growth_ratio_24_over_23"].cast(pl.Float64)
                for c in interaction_cols[:12]:
                    feature_matrix = feature_matrix.with_columns((gr * feature_matrix[c].cast(pl.Float64)).alias(f"{c}_x_growth"))
    except Exception:
        pass

    # Optional segment-specific feature pruning via allowlist
    try:
        _cfg = cfg.load_config()
        segs = [str(s).strip().lower() for s in getattr(getattr(_cfg, 'population', object()), 'build_segments', ["warm","cold"]) if str(s).strip()]
        allow_map = dict(getattr(getattr(_cfg, 'features', object()), 'segment_feature_allowlist', {}) or {})
        if len(segs) == 1:
            seg = segs[0]
            patterns = [str(p) for p in allow_map.get(seg, []) if str(p)]
            if patterns:
                essentials = {"customer_id", "bought_in_division", "industry", "industry_sub"}
                keep_cols = []
                for c in feature_matrix.columns:
                    cs = str(c)
                    if cs in essentials:
                        keep_cols.append(cs)
                        continue
                    if any(pat in cs for pat in patterns):
                        keep_cols.append(cs)
                if keep_cols:
                    # Ensure essentials are present
                    for e in list(essentials):
                        if e in feature_matrix.columns and e not in keep_cols:
                            keep_cols.append(e)
                    feature_matrix = feature_matrix.select(keep_cols)
    except Exception:
        pass

    logger.info(f"Successfully created feature matrix for division: {division_name}.")
    logger.info(f"Total customers processed: {feature_matrix.height}")
    positive_cases = feature_matrix.filter(pl.col('bought_in_division') == 1).height
    logger.info(f"Customers who bought in {division_name}: {positive_cases}")
    
    # Emit feature catalog artifact for auditing
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        fm_pd = feature_matrix.to_pandas()
        catalog = []
        for col in fm_pd.columns:
            dtype = str(fm_pd[col].dtype)
            non_null = int(fm_pd[col].notna().sum())
            coverage = round(non_null / len(fm_pd), 6) if len(fm_pd) else 0.0
            # Human-readable descriptions for important feature families
            if col == 'customer_id':
                desc = "Primary key for customer"
            elif col == 'bought_in_division':
                desc = f"Target: bought in {division_name} during prediction window"
            elif str(col).startswith('assets_subs_share_'):
                desc = "Assets: per-rollup subscription share (on / (on+off)) at cutoff"
            elif str(col).startswith('assets_on_subs_share_'):
                desc = "Assets: composition share across rollups among ON subscriptions at cutoff"
            elif str(col).startswith('assets_off_subs_share_'):
                desc = "Assets: composition share across rollups among OFF subscriptions at cutoff"
            elif str(col).startswith('assets_expiring_30d_'):
                desc = "Assets: quantity expiring within 30 days by rollup at cutoff"
            elif str(col).startswith('assets_expiring_60d_'):
                desc = "Assets: quantity expiring within 60 days by rollup at cutoff"
            elif str(col).startswith('assets_expiring_90d_'):
                desc = "Assets: quantity expiring within 90 days by rollup at cutoff"
            else:
                desc = "feature"
            catalog.append({"name": col, "dtype": dtype, "coverage": coverage, "description": desc})
        # Include cutoff in catalog filename for determinism across cutoffs
        fname = f"feature_catalog_{division_name.lower()}_{(cutoff_date or '').replace('-', '')}.csv" if cutoff_date else f"feature_catalog_{division_name.lower()}.csv"
        pd.DataFrame(catalog).to_csv(OUTPUTS_DIR / fname, index=False)
        logger.info("Wrote feature catalog to outputs directory.")
    except Exception:
        pass

    # Cast customer_id back to integer type where possible for tests and downstream consumers
    try:
        if 'customer_id' in feature_matrix.columns:
            feature_matrix = feature_matrix.with_columns(pl.col('customer_id').cast(pl.Int64, strict=False))
    except Exception:
        pass

    return feature_matrix


if __name__ == "__main__":
    db_engine = get_db_connection()
    # Example: Build the feature matrix for the 'Solidworks' division
    feature_matrix = create_feature_matrix(db_engine, "Solidworks")
    if not feature_matrix.is_empty():
        print("Feature Matrix Head:")
        print(feature_matrix.head().to_pandas().to_string())
        print("\nFeature Matrix Shape:")
        print(feature_matrix.shape)
