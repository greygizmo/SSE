### Phase 2 To-Do (feature library vs playbook)

- Config & windows
  - Add `features` section to config (windows_months, gp_winsor_p, add_missingness_flags, use_* toggles). TODO

- Guardrails helpers
  - Add `filter_to_cutoff(df, cutoff)` central helper; assert all blocks use it. TODO
  - Add winsorization helper for monetary columns. TODO

- Feature blocks (minimum champion set)
  - RFM (div/all): recency_days__life, tx_n/gp_sum/gp_mean__{3m,6m,12m,24m}. TODO
  - Trajectory: monthly gp_sum & tx_n slope/std over 12m. TODO
  - Cross-division: division_nunique__12m, gp_share__12m, recency_days__life per division. TODO
  - Diversity: sku_nunique__{3m,6m,12m,24m}. TODO
  - Lifecycle: tenure_days__life, gap_days__life, active_months__24m. TODO
  - Margin & returns: gp_pct__{W}, return_rate__12m, return_tx_n__12m. TODO
  - Seasonality: quarter shares or sin/cos. TODO

- Optional blocks (toggled)
  - Affinity (market-basket) score (already partially present) behind toggle. TODO
  - EB smoothing for ratios behind toggle. TODO
  - ALS embeddings (skip by default). TODO

- Artifacts
  - Write `features_{division}_{cutoff}.parquet` and `feature_catalog_{division}_{cutoff}.csv`. PARTIAL (catalog exists)
  - Write `feature_stats_{division}_{cutoff}.json` (coverage, null %, winsor caps). TODO
  - Deterministic sort + checksum. TODO

- CLI
  - Add `gosales/features/build.py` with flags: `--division`, `--cutoff`, `--windows`, `--config`, toggles. TODO

- Caching
  - Emit `customer_month_{cutoff}.parquet` for monthly gp_sum/tx_n aggregates. TODO

- Tests
  - Golden rows for windows; winsorization; determinism (checksum stable). TODO


