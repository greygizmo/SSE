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
  - Memory guards (for local SQLite runs):
    - `sqlite_skip_advanced_rows`: when feature rows exceed this, advanced extras are skipped. Default: 10,000,000.
    - `fastpath_minimal_return_rows`: when feature rows exceed this, a minimal feature matrix is returned early. Default: 10,000,000.
  - 2025-09 update: defaults for `add_missingness_flags`, `use_market_basket`, `use_als_embeddings`, and `pooled_encoders_enable` remain `true`; the scorer now batches to `float32` slices automatically, so leave these on unless you are debugging feature sources.
    - Both emit WARN logs when triggered with row counts and thresholds. In repo tests, a conservative cap is applied automatically for local SQLite when your configured `database.engine` != `sqlite` to keep peak memory under control.

- modeling
  - `models`: search order (e.g., `[lgbm, logreg]`), selection is metric-driven
  - `safe_divisions`: per-division SAFE policy
  - `top_k_percents`, `capacity_percent`: business thresholds
  - `calibration_methods`: `platt` and/or `isotonic`

- validation
  - `gauntlet_*`: e.g., mask tail days, purge days, label buffer
  - thresholds: `shift14_epsilon_*`, `ablation_epsilon_*`

### Whitespace (Phase-4)

- `weights`: base blend weights `[p_icp_pct, lift_norm, als_norm, EV_norm]`.
- `als_coverage_threshold`: minimum ALS coverage to avoid shrinking ALS weight; below this threshold ALS is downweighted and, if available, item2vec is used to backfill.
- `segment_columns`: optional list of segment keys (e.g., `['industry','size_bin']`) to apply coverage‑aware blending per segment; falls back to global when segment rows < `segment_min_rows`.
- `segment_min_rows`: minimum rows per segment required to use segment‑specific weights.
- `bias_division_max_share_topN`: max share per division in the selected top-percent capacity; enforced by rebalancer.

#### ALS Centroids (division-specific)

- Owner ALS centroid is computed per division from rows marked `owned_division_pre_cutoff == True` when available and cached under `gosales/outputs/` as `als_owner_centroid_<division>.npy`.
- Assets-ALS centroid is also cached per division as `assets_als_owner_centroid_<division>.npy` when assets signals are present.
- During ranking, ALS similarity uses the division’s cached centroid when `division_name` is present; fallback order is: per-division cache → in-group owned rows → in-group mean of valid rows. When no division context is available, the legacy global `als_owner_centroid.npy` is used if present.
- This prevents cross-division leakage of centroids and makes results reproducible per division; tests assert distinct centroids and scores across divisions even if one division has no owners.

See the UI “Feature Guide” tab for a rendered view of the current configuration.

---

## Recent Changes

- Post_Processing model added as a first-class target. Artifacts live under `gosales/models/post_processing_model` and include `metadata.json` with `division`, `cutoff_date`, and `prediction_window_months`.
- Scoring aligns per-division score frames before concatenation to avoid schema width mismatches.
- Scoring introduces adaptive batch mode: wide feature matrices (>=50k columns equivalent) are scored in float32 Polars slices so ALS, market-basket, missingness flags, and pooled encoders stay enabled on laptops without spiking memory.
- Scoring prefers curated DB connection (where curated fact tables exist), with safe fallback to the primary connection.
- Trainer always emits `metadata.json` (class balance, features, cutoff/window), even when degenerate probabilities abort artifact write.
- Prequential evaluator sanitizes features to numeric floats to prevent dtype errors with LightGBM.
- Label auto-widening in `gosales.labels.targets` now honors SKU-targeted models (e.g., `Printers`), reusing `sku_targets` during widening so only intended SKUs contribute to positives.
- Whitespace ranking now persists and uses division-specific ALS owner centroids and assets-ALS centroids (`als_owner_centroid_<division>.npy`, `assets_als_owner_centroid_<division>.npy`). Falls back safely when division context is missing. Guards added in tests to prevent cross-division centroid leakage.

### Quick Commands

```powershell
# Train (example: Post_Processing)
python -m gosales.models.train --division Post_Processing --cutoffs "2024-06-30,2024-12-31" --window-months 6 --models lgbm

# Score all divisions + whitespace (curated DB preferred)
python -m gosales.pipeline.score_customers

# Prequential horizon curves
python -m gosales.pipeline.prequential_eval --division Post_Processing --train-cutoff 2024-12-31 --start 2025-01 --end 2025-12 --window-months 6
```

