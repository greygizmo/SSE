# Test Suite Overview and Importance

This document summarizes the intent and criticality of the automated tests. They are grouped by pipeline phase and operational concerns so engineers can quickly understand the guardrails that protect each stage of the GoSales Engine.

## Phase 0–1: ETL, Contracts, and Labeling
- `test_cleaners.py`
  - Currency/date coercion produces reliable floats and timestamps to protect aggregates and time filters.
- `test_parse_and_keys.py`
  - Parsing and surrogate key builders yield canonical formats and deterministic IDs.
- `test_contracts.py`
  - Required columns, PK uniqueness, and date bounds are enforced to block schema drift.
- `test_fact_transactions_exists.py`
  - Ensures the `fact_transactions` table exists post‑ingest.
- `test_sku_map.py`
  - SKU mapping covers critical divisions/metadata to prevent misclassification.
- `test_labels.py`
  - Denylist thresholds, expansion vs all‑customer modes, and censoring flags keep labels leakage‑safe.

## Phase 2: Feature Engineering
- `test_als_embedding_window.py`
  - ALS embeddings respect lookback windows; no stale transactions leak forward.
- `test_feature_matrix_memory.py`
  - Guardrails on memory use during feature generation to keep CI stable.
- `test_features.py`
  - Window/target alignment, CLI checksum emissions, and config override persistence.

## Phase 3: Modeling and Calibration
- `test_phase3_train_safe_calibration.py`
  - Sparse‑positive cohorts skip calibration with a recorded reason; diagnostics remain complete.

## Phase 4: Whitespace Ranking / Next‑Best‑Action
- `test_whitespace_score.py`
  - Uses aggregated queries (no `SELECT *`) and avoids O(customers×divisions) Python loops.
- `test_whitespace_missing_divisions.py`
  - Filters invalid divisions and preserves continuous scores.
- `test_phase4_rank_normalization.py` (updated)
  - Zero‑signal rows cannot outrank txn‑ALS accounts; fallbacks are clipped below genuine ALS maxima.
- `test_phase4_weight_scaling_and_als.py` (updated)
  - Asset‑only rows use full `assets_norm`; mixed‑signal rows follow blend weights.
- `test_build_lift.py`
  - Market‑basket lift mining remains functional and aligned to ranking inputs.

## Phase 5: Validation and Holdout Monitoring
- `test_phase5_ks_snapshot.py`
  - KS statistics are written for train vs holdout.
- `test_phase5_scenarios_and_segments.py`
  - Scenario calculators and segment outputs (per‑rep/hybrid) are consistent.
- `test_deciles_constant.py`
  - Decile aggregation behaves correctly under low‑variance scenarios.

## Phase 6: Orchestration, Config, and Observability
- `test_phase6_config_and_registry.py`
  - Config validation, run registry/manifest writing, and whitespace weight normalization.
- `test_db_connection.py`
  - Azure→SQLite fallbacks and engine URL overrides.
- `test_monitoring_data_collector.py`
  - Recent alerts and performance metrics discovered from artifacts; data-quality scoring now reacts to missing metrics, gate failures, and alert severity rather than defaulting to 99; lineage pulls only from real run manifests and returns empty when absent; fallbacks documented.
- `test_monitoring_type_consistency.py`
  - Mixed‑type samples are penalized; uniform types score high.
- `test_score_all_pruning.py`
  - Target derivation and legacy model pruning keep disk usage in check.
- `test_score_customers.py::test_dim_customer_cache_isolation_across_engines`
  - `dim_customer` cache is per‑connection and immutable by callers.
- `test_types.py`
  - Type enforcement preserves schema contracts and handles empties gracefully.
