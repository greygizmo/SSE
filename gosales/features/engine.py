import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
from gosales.utils.db import get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils import config as cfg
from gosales.utils.paths import OUTPUTS_DIR
from gosales.features.als_embed import customer_als_embeddings
from gosales.etl.sku_map import division_set
from gosales.utils.normalize import normalize_division

logger = get_logger(__name__)

def create_feature_matrix(engine, division_name: str, cutoff_date: str = None, prediction_window_months: int = 6):
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
    if cutoff_date:
        logger.info(f"Using cutoff date: {cutoff_date} (features from data <= cutoff)")
        logger.info(f"Target: purchases in {prediction_window_months} months after cutoff")

    # --- 1. Load Base Data ---
    try:
        division_filter = ", ".join(f"'{d}'" for d in division_set())
        base_cols = "customer_id, order_date, product_division, product_sku, gross_profit"

        def _read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
            chunks = pd.read_sql_query(sql, engine, params=params, chunksize=100_000)
            frames = [chunk for chunk in chunks]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)

        if cutoff_date:
            feature_sql = (
                f"SELECT {base_cols} FROM fact_transactions "
                f"WHERE order_date <= :cutoff AND product_division IN ({division_filter})"
            )
            feature_data = _read_sql(feature_sql, {"cutoff": cutoff_date})
            feature_data["order_date"] = pd.to_datetime(feature_data["order_date"])

            from dateutil.relativedelta import relativedelta

            cutoff_dt = pd.to_datetime(cutoff_date)
            prediction_end = (cutoff_dt + relativedelta(months=prediction_window_months)).strftime("%Y-%m-%d")
            pred_sql = (
                "SELECT customer_id, order_date, product_division FROM fact_transactions "
                "WHERE order_date > :cutoff AND order_date <= :pred_end "
                f"AND product_division IN ({division_filter})"
            )
            prediction_data = _read_sql(
                pred_sql, {"cutoff": cutoff_date, "pred_end": prediction_end}
            )
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
            feature_sql = (
                f"SELECT {base_cols} FROM fact_transactions "
                f"WHERE product_division IN ({division_filter})"
            )
            feature_data = _read_sql(feature_sql)
            feature_data["order_date"] = pd.to_datetime(feature_data["order_date"])
            prediction_data = feature_data.copy()

        transactions = pl.from_pandas(feature_data)
        if "customer_id" in transactions.columns:
            transactions = transactions.with_columns(
                pl.col("customer_id").cast(pl.Int64, strict=False)
            )

        customers_pd = pd.read_sql_query(
            "SELECT customer_id FROM dim_customer", engine
        )
        customers_pd["customer_id"] = pd.to_numeric(
            customers_pd["customer_id"], errors="coerce"
        ).astype("Int64")
        customers = pl.from_pandas(customers_pd).with_columns(
            pl.col("customer_id").cast(pl.Int64, strict=False)
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
        # Use prediction window data for target labels (normalize division)
        pred_div = prediction_data['product_division'].astype(str).str.strip()
        prediction_buyers_df = prediction_data.loc[pred_div == norm_division_name, 'customer_id'].unique()
        division_buyers_pd = pd.DataFrame({'customer_id': prediction_buyers_df, 'bought_in_division': 1})
        # Enforce integer customer_id for join compatibility
        if 'customer_id' in division_buyers_pd.columns:
            division_buyers_pd['customer_id'] = pd.to_numeric(division_buyers_pd['customer_id'], errors='coerce').astype('Int64')
        division_buyers = pl.from_pandas(division_buyers_pd).with_columns(pl.col('customer_id').cast(pl.Int64, strict=False)).lazy()
        logger.info(f"Target: {len(prediction_buyers_df)} customers bought {division_name} in prediction window")
    else:
        # Original behavior: ever bought in historical data
        division_buyers = (
            transactions.lazy()
            .filter(pl.col("product_division") == norm_division_name)
            .select("customer_id")
            .unique()
            .with_columns(pl.lit(1).cast(pl.Int8).alias("bought_in_division"))
        )

    # --- 3. Engineer Behavioral Features ---
    # Get the current date for recency calculations
    current_date = datetime.now().date()

    features = (
        transactions.lazy()
        .group_by("customer_id")
        .agg([
            # Recency Features
            pl.col("order_date").max().alias("last_order_date"),
            pl.col("order_date").filter(pl.col("product_division") == division_name).max().alias(f"last_{division_name}_order_date"),
            
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
                days_diff.append((current_date - date.date()).days)
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
                days_diff.append((current_date - date.date()).days)
        features_pd[f'days_since_last_{division_name}_order'] = days_diff
    else:
        features_pd[f'days_since_last_{division_name}_order'] = 999  # Default for customers with no orders in division
    
    # --- 3a. Windowed RFM and temporal dynamics (pandas) ---
    try:
        cfgmod = cfg.load_config()
        cutoff_dt = pd.to_datetime(cutoff_date) if cutoff_date else transactions_pd['order_date'].max()
        fd = feature_data.copy()
        fd['order_date'] = pd.to_datetime(fd['order_date'])
        fd = fd.sort_values(['customer_id', 'order_date'])

        window_months = cfgmod.features.windows_months or [3, 6, 12, 24]
        per_customer_frames = []
        per_customer_frames_div = []
        for w in window_months:
            start_dt = cutoff_dt - pd.DateOffset(months=w)
            mask = (fd['order_date'] > start_dt) & (fd['order_date'] <= cutoff_dt)
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
            sub_div = fd.loc[mask & (fd['product_division'].astype(str).str.strip() == norm_division_name), ['customer_id', 'order_date', 'gross_profit']]
            tx_n_div = sub_div.groupby('customer_id')['order_date'].count().rename(f'rfm__div__tx_n__{w}m').reset_index()
            gp_sum_div = sub_div.groupby('customer_id')['gross_profit'].sum().rename(f'rfm__div__gp_sum__{w}m').reset_index()
            gp_mean_div = sub_div.groupby('customer_id')['gross_profit'].mean().rename(f'rfm__div__gp_mean__{w}m').reset_index()
            # Margin proxy: gp_pct = gp_sum / |gp_sum| + epsilon (since revenue is not available here)
            agg_div = tx_n_div.merge(gp_sum_div, on='customer_id', how='outer').merge(gp_mean_div, on='customer_id', how='outer')
            col_gp = f'rfm__div__gp_sum__{w}m'
            agg_div[f'margin__div__gp_pct__{w}m'] = agg_div[col_gp].astype(float) / (agg_div[col_gp].abs().astype(float) + 1e-9)
            per_customer_frames_div.append(agg_div)

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
            sub = fd.loc[fd['product_division'].astype(str).str.strip() == normalize_division(d), ['customer_id', 'order_date']]
            last_d = sub.groupby('customer_id')['order_date'].max().reset_index()
            last_d[f'days_since_last_{d.lower()}'] = (cutoff_dt - last_d['order_date']).dt.days
            rec_div_list.append(last_d[['customer_id', f'days_since_last_{d.lower()}']])
        rec_div = None
        if rec_div_list:
            rec_div = rec_div_list[0]
            for extra in rec_div_list[1:]:
                rec_div = rec_div.merge(extra, on='customer_id', how='outer')

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
        past_swx = fd.loc[fd['product_division'].astype(str).str.strip() == 'Solidworks', ['customer_id']].drop_duplicates()
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

        # --- Region (Branch) and Rep features from sales_log (feature period only) ---
        try:
            sl = pd.read_sql("SELECT CustomerId, [Rec Date] AS rec_date, Branch, Rep FROM sales_log", engine)
            sl['rec_date'] = pd.to_datetime(sl['rec_date'], errors='coerce')
            sl['customer_id'] = pd.to_numeric(sl['CustomerId'], errors='coerce').astype('Int64')
            sl = sl.dropna(subset=['customer_id'])
            sl = sl[sl['rec_date'] <= cutoff_dt]
            # Top branches and reps
            top_branches = sl['Branch'].astype(str).str.strip().value_counts().head(30).index.tolist()
            top_reps = sl['Rep'].astype(str).str.strip().value_counts().head(50).index.tolist()

            import re
            def sanitize_key(text: str) -> str:
                if text is None:
                    return "unknown"
                key = str(text).lower().strip()
                key = key.replace("&", " and ")
                key = re.sub(r"[^0-9a-zA-Z]+", "_", key)
                key = re.sub(r"_+", "_", key).strip("_")
                if not key:
                    key = "unknown"
                return key

            # Branch share features
            b = sl[['customer_id', 'Branch']].copy()
            b['Branch'] = b['Branch'].astype(str).str.strip()
            b['count'] = 1
            b_tot = b.groupby('customer_id')['count'].sum().rename('branch_tx_total')
            b_top = b[b['Branch'].isin(top_branches)].groupby(['customer_id', 'Branch'])['count'].sum().unstack(fill_value=0)
            # Normalize to shares
            b_top = b_top.div(b_tot, axis=0).fillna(0.0)
            b_top.columns = [f"branch_share_{sanitize_key(c)}" for c in b_top.columns]

            # Rep share features
            r = sl[['customer_id', 'Rep']].copy()
            r['Rep'] = r['Rep'].astype(str).str.strip()
            r['count'] = 1
            r_tot = r.groupby('customer_id')['count'].sum().rename('rep_tx_total')
            r_top = r[r['Rep'].isin(top_reps)].groupby(['customer_id', 'Rep'])['count'].sum().unstack(fill_value=0)
            r_top = r_top.div(r_tot, axis=0).fillna(0.0)
            r_top.columns = [f"rep_share_{sanitize_key(c)}" for c in r_top.columns]

            br = b_top.join(r_top, how='outer').reset_index()
            features_pd = features_pd.merge(br, on='customer_id', how='left')
            for col in br.columns:
                if col == 'customer_id':
                    continue
                features_pd[col] = features_pd[col].fillna(0.0)
        except Exception as e:
            logger.warning(f"Branch/Rep feature build failed: {e}")

        # --- Basket lift / affinity ---
        try:
            cfgmod = cfg.load_config()
            if cfgmod.features.use_market_basket:
                # Limit to feature window (e.g., last 12 months) to avoid leakage
                fd_win = fd.loc[last12_mask, ['customer_id', 'product_sku', 'product_division']].dropna().copy()
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

                    # Aggregate to per-customer signals: max and mean lift of present SKUs
                    if lift_weights:
                        # Align DataFrame to lift columns only
                        cols = [c for c in has_sku.columns if c in lift_weights]
                        if cols:
                            w = pd.Series({c: lift_weights.get(c, 0.0) for c in cols})
                            present = has_sku[cols].astype(float)
                            weighted = present.mul(w, axis=1)
                            mb_lift_max = weighted.max(axis=1).rename('mb_lift_max')
                            # mean over present SKUs (avoid dividing by zero)
                            denom = present.sum(axis=1).replace(0.0, np.nan)
                            mb_lift_mean = (weighted.sum(axis=1) / denom).fillna(0.0).rename('mb_lift_mean')
                            mb_df = pd.concat([mb_lift_max, mb_lift_mean], axis=1).reset_index()
                            features_pd = features_pd.merge(mb_df, on='customer_id', how='left')
                            features_pd['mb_lift_max'] = features_pd['mb_lift_max'].fillna(0.0)
                            features_pd['mb_lift_mean'] = features_pd['mb_lift_mean'].fillna(0.0)

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
                            affinity = weighted.sum(axis=1).rename('affinity__div__lift_topk__12m')
                            affinity_df = affinity.reset_index()
                            features_pd = features_pd.merge(affinity_df, on='customer_id', how='left')
                            features_pd['affinity__div__lift_topk__12m'] = features_pd['affinity__div__lift_topk__12m'].fillna(0.0)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Basket lift computation failed: {e}")
    except Exception as e:
        # Non-fatal; proceed with base features if advanced features fail
        logger.warning(f"Advanced temporal features failed: {e}")

    # Flags / indicators aggregated from sales_log: ACR and New (robust to missing/variant names)
    try:
        # Read only needed columns, but fallback to * if not present
        try:
            sl_flags = pd.read_sql("SELECT CustomerId, \"Rec Date\" AS rec_date, * FROM sales_log", engine)
        except Exception:
            sl_flags = pd.read_sql("SELECT * FROM sales_log", engine)
            if 'Rec Date' in sl_flags.columns:
                sl_flags = sl_flags.rename(columns={'Rec Date': 'rec_date'})

        # Coerce types early
        sl_flags['rec_date'] = pd.to_datetime(sl_flags.get('rec_date'), errors='coerce')
        sl_flags['customer_id'] = pd.to_numeric(
            sl_flags.get('CustomerId', sl_flags.get('customer_id')),
            errors='coerce'
        ).astype('Int64')
        if cutoff_date and 'rec_date' in sl_flags.columns:
            sl_flags = sl_flags[sl_flags['rec_date'] <= cutoff_dt]

        def _normalize(col: str) -> str:
            return str(col).strip().lower().replace('[', '').replace(']', '')

        cols_norm = { _normalize(c): c for c in sl_flags.columns }
        # Allow a few common variants
        acr_col = cols_norm.get('acr') or cols_norm.get('is_acr') or cols_norm.get('acr_flag')
        new_col = cols_norm.get('new') or cols_norm.get('is_new') or cols_norm.get('new_customer')

        agg_parts = []
        if isinstance(acr_col, str) and acr_col in sl_flags.columns:
            acr_num = pd.to_numeric(sl_flags[acr_col], errors='coerce').fillna(0).astype('Int8')
            ever_acr = acr_num.groupby(sl_flags['customer_id']).max().astype('Int8').reset_index(name='ever_acr')
            agg_parts.append(ever_acr)
        if isinstance(new_col, str) and new_col in sl_flags.columns:
            new_num = pd.to_numeric(sl_flags[new_col], errors='coerce').fillna(0).astype('Int8')
            ever_new = new_num.groupby(sl_flags['customer_id']).max().astype('Int8').reset_index(name='ever_new_customer')
            agg_parts.append(ever_new)

        if agg_parts:
            agg_flags = agg_parts[0]
            for ap in agg_parts[1:]:
                agg_flags = agg_flags.merge(ap, on='customer_id', how='outer')
        else:
            # No flags available; create empty frame with customer_id only
            agg_flags = sl_flags[['customer_id']].dropna().drop_duplicates().copy()

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
            als_df = customer_als_embeddings(
                engine,
                cutoff_date,
                factors=16,
                lookback_months=cfgmod.features.als_lookback_months,
            )
            if not als_df.is_empty():
                # Join on customer_id
                als_pd = als_df.to_pandas()
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
    features = features.with_columns(pl.col("customer_id").cast(pl.Int64, strict=False) if "customer_id" in features.columns else pl.lit(None))
    customers = customers.with_columns(pl.col("customer_id").cast(pl.Int64, strict=False))

    feature_matrix = (
        customers.lazy()
        .join(features.lazy(), on="customer_id", how="left")
        .join(division_buyers, on="customer_id", how="left")
        .with_columns([
            pl.col("bought_in_division").fill_null(0).cast(pl.Int8),
        ])
        .collect()
    )

    # Fill nulls for all other columns in pandas for easier handling
    feature_matrix_pd = feature_matrix.to_pandas()
    
    # Drop the date columns that are no longer needed for ML
    date_columns = [col for col in feature_matrix_pd.columns if 'date' in col.lower()]
    feature_matrix_pd = feature_matrix_pd.drop(columns=date_columns)
    
    # Fill nulls and ensure proper data types
    feature_matrix_pd = feature_matrix_pd.fillna(0)
    # Map to naming scheme for key features
    # Recency
    if 'days_since_last_order' in feature_matrix_pd.columns:
        feature_matrix_pd['rfm__all__recency_days__life'] = feature_matrix_pd['days_since_last_order']
    div_rec_col = f'days_since_last_{division_name}_order'
    if div_rec_col in feature_matrix_pd.columns:
        feature_matrix_pd['rfm__div__recency_days__life'] = feature_matrix_pd[div_rec_col]
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
                zeros = np.zeros((len(feature_matrix_pd), len(cols)), dtype=np.int8)
                flags = pd.DataFrame(zeros, columns=[f"{c}_missing" for c in cols], index=feature_matrix_pd.index)
                feature_matrix_pd = pd.concat([feature_matrix_pd, flags], axis=1)
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
            feature_columns = ["customer_id"] + \
                [f"is_{industry_key_map[i]}" for i in top_industries] + \
                [f"is_sub_{sub_key_map[s]}" for s in top_subs]

            feature_matrix = feature_matrix.join(
                industry_features.select(feature_columns),
                on="customer_id",
                how="left"
            ).fill_null(0)
            
            logger.info(f"Successfully joined industry data. Added {len(top_industries)} industry and {len(top_subs)} sub-industry dummies.")
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
            desc = (
                "Primary key for customer" if col == 'customer_id' else (
                    f"Target: bought in {division_name} during prediction window" if col == 'bought_in_division' else "feature"
                )
            )
            catalog.append({"name": col, "dtype": dtype, "coverage": coverage, "description": desc})
        # Include cutoff in catalog filename for determinism across cutoffs
        fname = f"feature_catalog_{division_name.lower()}_{(cutoff_date or '').replace('-', '')}.csv" if cutoff_date else f"feature_catalog_{division_name.lower()}.csv"
        pd.DataFrame(catalog).to_csv(OUTPUTS_DIR / fname, index=False)
        logger.info("Wrote feature catalog to outputs directory.")
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
