### Phase 2 To-Do (feature library vs playbook)

- Config & windows
  - Add `features` section to config (windows_months, gp_winsor_p, add_missingness_flags, use_* toggles). DONE

- Guardrails helpers
  - Add `filter_to_cutoff(df, cutoff)` central helper; assert all blocks use it. PARTIAL (helper added)
  - Add winsorization helper for monetary columns. DONE

- Feature blocks (minimum champion set)
  - RFM (div/all): recency_days__life, tx_n/gp_sum/gp_mean__{3m,6m,12m,24m}. PARTIAL (added several, needs full coverage)
  - Trajectory: monthly gp_sum & tx_n slope/std over 12m. DONE
  - Cross-division: division_nunique__12m, gp_share__12m, recency_days__life per division. PARTIAL (share EB for target div)
  - Diversity: sku_nunique__{3m,6m,12m,24m}. PARTIAL (12m implemented)
  - Lifecycle: tenure_days__life, gap_days__life, active_months__24m. PARTIAL (tenure/gap implemented)
  - Margin & returns: gp_pct__{W}, return_rate__12m, return_tx_n__12m. TODO
  - Seasonality: quarter shares or sin/cos. DONE (quarter shares)

- Optional blocks (toggled)
  - Affinity (market-basket) score (already partially present) behind toggle. DONE (toggle wired)
  - EB smoothing for ratios behind toggle. DONE (for division gp_share)
  - ALS embeddings (skip by default). TODO

- Artifacts
  - Write `features_{division}_{cutoff}.parquet` and `feature_catalog_{division}_{cutoff}.csv`. DONE (CLI)
  - Write `feature_stats_{division}_{cutoff}.json` (coverage, null %, winsor caps). DONE (coverage)
  - Deterministic sort + checksum. PARTIAL (sort done; checksum TODO)

- CLI
  - Add `gosales/features/build.py` with flags: `--division`, `--cutoff`, `--windows`, `--config`, toggles. DONE

- Caching
  - Emit `customer_month_{cutoff}.parquet` for monthly gp_sum/tx_n aggregates. DONE (builder in features/cache.py)

- Tests
  - Golden rows for windows; winsorization; determinism (checksum stable). TODO


