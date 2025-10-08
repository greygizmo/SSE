# Serious flaws identified in GoSales Engine

This review highlights five issues that materially impact correctness, observability, or scalability of the GoSales Engine codebase.

## 1. Global dim_customer cache leaks across database connections
`_get_dim_customer` caches the contents of `dim_customer` the first time it runs and unconditionally returns that cached DataFrame for all subsequent calls, regardless of which SQLAlchemy engine is passed in.【F:gosales/pipeline/score_customers.py†L74-L106】 When the scorer is invoked against more than one tenant/database in the same Python process (e.g., sequential divisions, on-prem vs. cloud), the second call will silently reuse customer names from the previous connection. That contaminates scoring outputs with the wrong customer roster and leaks data across tenants.

## 2. Whitespace heuristic reads entire fact table into memory
`generate_whitespace_opportunities` issues `SELECT * FROM fact_transactions` and materialises the full result set in pandas before converting to Polars and looping customer-by-division to emit opportunities.【F:gosales/pipeline/score_customers.py†L1032-L1099】 On realistic datasets (millions of rows, dozens of divisions) this approach will exceed memory, thrash the process, and make the CLI unusable. The algorithm also performs an O(customers × divisions) Python loop, which cannot scale to production volumes.

## 3. Monitoring cannot see real validation metrics
The monitoring collector only scans the top-level `OUTPUTS_DIR` for files named `validation_metrics*.json`.【F:gosales/monitoring/data_collector.py†L552-L558】 Real validation outputs are written under `outputs/validation/<division>/<cutoff>/metrics.json`, so the monitor never discovers them.【F:gosales/tests/test_phase5_per_feature_psi_highlight.py†L59-L65】 As a result the telemetry stack operates on stale defaults and misses actual validation failures.

## 4. Data-quality score is hard-coded to look healthy
Even when no validation artifacts are found (the common case given flaw #3), `_calculate_data_quality_score` returns 99.0 and never drops below 90, regardless of pipeline health.【F:gosales/monitoring/data_collector.py†L48-L84】 This hard-coded optimism masks outages, causing dashboards to report excellent quality when validations are missing or failing.

## 5. Monitoring fabricates lineage statistics
When no `run_context_*.json` is present, `_collect_data_lineage` fabricates a canned lineage table with fixed record counts, durations, and data sources.【F:gosales/monitoring/data_collector.py†L560-L597】 Presenting invented metrics as real lineage misleads operators and breaks auditability; it should instead signal that lineage is unavailable.

Addressing these flaws is critical for trustworthy scoring, operational monitoring, and the ability to run the pipeline at production scale.
