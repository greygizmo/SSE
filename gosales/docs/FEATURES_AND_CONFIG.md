# Feature Families and Configuration Guide

This document summarizes the engineered feature families used by GoSales and how to control them via configuration.

## Feature Families

- Recency
  - `rfm__all|div__recency_days__life`: Days since last order (all/division)
  - `rfm__all|div__log_recency__life`: Log-transform of recency days (stabilizes heavy tails)
  - `rfm__all|div__recency_decay__hl{30|90|180}`: Exponential decay with half-lives in days

- RFM Windows (All and Division scope)
  - `rfm__all|div__tx_n__{3|6|12|24}m`: Transaction counts
  - `rfm__all|div__gp_sum__{3|6|12|24}m`: GP sums (winsorized)
  - `rfm__all|div__gp_mean__{3|6|12|24}m`: GP means

- Offset Windows (decorrelate from boundary)
  - Same RFM windows ending at `cutoff - offset_days`, e.g., `__12m_off60d`

- Window Deltas (trend without adjacency)
  - 12m vs previous 12m (from 24m totals):
  - `rfm__all|div__{gp_sum,tx_n}__delta_12m_prev12m`, `...__ratio_12m_prev12m`

- Tenure
  - `lifecycle__all__tenure_days__life`, tenure months, and bucket indicators: `lt3m, 3to6m, 6to12m, 1to2y, ge2y`

- Industry/Sub Dummies (top-N)
  - `is_<industry>`, `is_sub_<sub>`

- Pooled/Hiera Encoders (non-leaky; pre-cutoff only)
  - Industry-level: `enc__industry__tx_rate_24m_smooth`, `enc__industry__gp_share_24m_smooth`
  - Sub-industry (shrunk to parent industry): `enc__industry_sub__tx_rate_24m_smooth`, `enc__industry_sub__gp_share_24m_smooth`

- Affinity (Market Basket with lag)
  - Computes SKU presence matrix up to `cutoff - affinity_lag_days`, derives per-SKU lift toward the target division.
  - Aggregates to per-customer signals:
    - `mb_lift_max_lag{N}d`, `mb_lift_mean_lag{N}d`
    - `affinity__div__lift_topk__12m_lag{N}d`
  - Avoids near-boundary adjacency via the lag (default N=60).

- Diversity & Dynamics
  - `diversity__*`, `xdiv__*`, monthly slopes/std for GP and TX over 12m

- Assets
  - `assets_expiring_{30|60|90}d_*`, `assets_*_subs_share_*`

- ALS Embeddings
  - `als_f*` (if enabled)

- SKU Aggregates (12m)
  - `sku_gp_12m_*`, `sku_qty_12m_*`, `sku_gp_per_unit_12m_*`

## Configuration Reference (highlights)

All options live in `gosales/config.yaml`. Key entries:

- features
  - `windows_months`: e.g., `[3, 6, 12, 24]`
  - `gp_winsor_p`: winsor upper quantile for GP sums
  - `recency_floor_days`: floor for recency to reduce adjacency
  - `recency_decay_half_lives_days`: half-lives for hazard decays
  - `enable_offset_windows`, `offset_days`: build offset windows
  - `enable_window_deltas`: build 12m vs previous 12m deltas
  - `affinity_lag_days`: embargo days for affinity exposures (default: 60)
  - `pooled_encoders_enable`: enable pooled encoders
  - `pooled_encoders_lookback_months`: lookback for encoders (default: 24)
  - `pooled_alpha_industry`, `pooled_alpha_sub`: smoothing strength
  - `use_assets`, `use_als_embeddings`, `use_market_basket`, `add_missingness_flags`

- modeling
  - `models`: search order (e.g., `[lgbm, logreg]`), selection is metric-driven
  - `safe_divisions`: per-division SAFE policy
  - `top_k_percents`, `capacity_percent`: business thresholds
  - `calibration_methods`: `platt` and/or `isotonic`

- validation
  - `gauntlet_*`: e.g., mask tail days, purge days, label buffer
  - thresholds: `shift14_epsilon_*`, `ablation_epsilon_*`

See the UI â€œFeature Guideâ€ tab for a rendered view of the current configuration.

