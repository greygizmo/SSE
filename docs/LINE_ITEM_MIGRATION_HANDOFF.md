# Line-Item Migration Handoff (for the next Codex agent)

Branch: refactor/etl-line-item-sources
Updated: 2025-10-16
Contact: GoSales Engine maintainers

This document summarizes the current status, key decisions, code entry points, and the exact next tasks to continue the migration from legacy views to the new line-item sources. Use this as your single source of truth to proceed efficiently and deterministically.

## Executive Summary
- New fact is line-item grain from `dbo.table_saleslog_detail`.
- Canonical order identifier: `Sales_Order` (not `SalesOrderId`).
- COGS = `Amount2`; GP columns = `GP`, `Term_GP`.
- Currency normalization: write `_usd` fields (Revenue, Amount2/COGS, GP, Term_GP) using `USD_CAD_Conversion_rate` when `SalesOrder_Currency == 'CAD'`.
- Division mapping (Goals): we continue to call them “Divisions” in the repo, but the canonical categories come from the new Goal taxonomy. Do not use the `Division` column in `dbo.table_saleslog_detail`.
  - Primary sources:
    - `dbo.analytics_order_tags` (NEW): transaction-aware tag bridge with columns `Id` (product internal id), `tag` (item rollup), and `goal` (Goal category).
    - `dbo.analytics_product_tags`: static mapping at the rollup level (`item_rollup` ? `Goal`) used to connect rollups to quantity/GP columns in the (legacy) sales log views and to rollups in product info.
  - Precedence for resolving canonical Division (Goal):
    1) `analytics_order_tags` by `Item_internalid` ? (`tag`,`goal`)
    2) `analytics_product_tags` by `item_rollup` ? `Goal`
    3) Fallback: `Unknown`
  - Throughout the codebase and docs we will continue to say “Division”, but mean “Goal” from the above sources.
- Artifacts now written (when enabled): `fact_sales_line` and `fact_sales_header` (derived aggregation) alongside legacy curated outputs.

## What’s Already Implemented
- ETL scaffolding: `gosales/etl/sales_line.py`
  - Loads sales detail, dedupes snapshots by latest `last_update` per (`Sales_Order`, `Item_internalid`, `Revenue`, `Amount2`, `GP`, `Term_GP`).
  - Keeps the approved column list; aliases `COGS`; adds USD-normalized fields.
  - `summarise_sales_header(df)` aggregates to a header-level fact for parity checks.
- Integrated into star build: `gosales/etl/build_star.py`
  - Adds `fact_sales_line` and `fact_sales_header` when toggled on.
  - CLI flags: `--use-line-items/--no-use-line-items` to force enable/disable at runtime.
- Pipeline toggles: `gosales/pipeline/score_all.py`
  - Accepts `--use-line-items/--no-use-line-items` and propagates to ETL.
- Config schema: `gosales/utils/config.py`
  - Adds `etl.line_items.{use_line_item_facts,sources,dedupe,behavior}` with defaults.
- Goal-driven division mapping:
  - `fact_sales_line` now surfaces `item_rollup`, `division_goal`, and `division_canonical` columns. `dbo.analytics_order_tags` is applied first (matched on `Item_internalid`), with `dbo.analytics_product_tags` and legacy division as deterministic fallbacks. Coverage is logged per run.
  - `fact_transactions` consumes the same mapping, writing canonical `product_division` and a diagnostic `product_goal` column (Draftsight -> Solidworks, GeoMagic -> Scanning) while preserving legacy division names for features.
- Line-item behavior toggles:
  - `etl.line_items.behavior.exclude_line_types` filters disallowed `Rev_type` categories (e.g., `tax`, `freight`, `discount_only`) before aggregation.
  - `etl.line_items.behavior.return_treatment` supports `net_amount` (default), `exclude_returns`, and `separate_flag` (adds an `is_return_line` boolean in `fact_sales_line`).
  - `etl.line_items.behavior.kit_handling == "prefer_children"` trims kit parent rows when kit indicators are present; falls back gracefully when signals are absent.
- Feature engine alignment:
  - Branch/rep share metrics and ACR/New flags now source from `fact_sales_line`, falling back to legacy `fact_sales_log_raw`/`sales_log` only when the line-item fact is disabled.
  - Transaction ALS embeddings now use line-item interactions from `fact_transactions`, combine `quantity` and positive `gross_profit` (via `log1p`), apply a pre-cutoff embargo of `features.affinity_lag_days`, and restrict to `features.als_lookback_months`.
  - New line-item feature families delivered across the feature engine:
    - Canonical division aggregates and revenue shares (`xdiv__canon_*`), including Unknown share and USD-based margin rates (`margin__*`).
    - Rollup/tag diversity, returns intensity, and currency mix signals derived from `fact_sales_line` windows.
    - Order composition metrics (lines/order, revenue per line/order) plus expanded branch/rep exposure shares.
    - Optional tag share features (`tagals__share_*`) gated by `features.enable_tag_als` for lightweight rollup embeddings.
- Documentation:
  - Migration plan: `docs/migration_line_item_etl_plan.md` (status, keep-list, currency, CLI flags, outstanding tasks).
  - Interrogation workbook: `docs/migration_line_item_table_interrogation.md` (parity results, captured CSVs, progress log).
- Validation automation:
  - `python -m gosales.validation.line_item_parity` aggregates legacy vs line-item division totals and stores CSV/JSON outputs under `gosales/outputs/validation/line_item_parity/` with pass/fail summaries.
  - Holdout sourcing defaults to curated `fact_sales_line` (canonical divisions) when line-items are enabled, providing canonical buyer lists and per-customer holdout GP; override via `validation.holdout_db_object` when comparing against legacy views.

## How To Run Locally
1) Build the curated layer with line-items enabled
- `PYTHONPATH="$PWD" python -m gosales.etl.build_star --config gosales/config.yaml --use-line-items`

2) Or run end-to-end scoring with the toggle
- `PYTHONPATH="$PWD" python -m gosales.pipeline.score_all --use-line-items`

3) Inspect artifacts
- DB (curated): tables `fact_sales_line`, `fact_sales_header` (plus legacy curated tables).
- Parquet: `gosales/data/curated/fact/fact_sales_line.parquet`, `gosales/data/curated/fact/fact_sales_header.parquet`.
- Diagnostics: `gosales/docs/appendices/migration_line_item/` (column inventories, parity CSVs, monthly counts, goal coverage, join samples).

4) Run automated parity diagnostics
- `PYTHONPATH="$PWD" python -m gosales.validation.line_item_parity --config gosales/config.yaml --cutoff "2024-12-31"`

## Current Parity State (high level)
- Using `Sales_Order` as the key, latest parity extract shows:
  - 10,635 orders with residual differences > $0.01.
  - 1,729 orders exist only in the new line-item outputs (legacy view omits them).
- Main driver: legacy header includes manual adjustments not present in the raw line-item fact. Treat as expected unless business rules require a modeled adjustment layer.

## Key Decisions (do not change without discussion)
- Use `Sales_Order` for order identity across the system.
- COGS = `Amount2` (source of truth for cost).
- USD normalization using `USD_CAD_Conversion_rate` when `SalesOrder_Currency == 'CAD'`.
- Division mapping via tags: `analytics_product_tags.Goal` is canonical; if a rollup lacks a goal, map to `Unknown` until defined.

## Files You’ll Touch Most Often
- ETL (line fact): `gosales/etl/sales_line.py`
- ETL (star build & toggles): `gosales/etl/build_star.py`
- Pipeline toggles: `gosales/pipeline/score_all.py`
- Config types & loading: `gosales/utils/config.py`
- Docs: `docs/migration_line_item_etl_plan.md`, `docs/migration_line_item_table_interrogation.md`

## Fact Tables at a Glance
- `fact_sales_line`  
  - Grain: one row per deduped item line (`Sales_Order`, `Item_internalid`, `Revenue`, `Amount2`, `GP`, `Term_GP`, latest `last_update`).  
  - Use when you need item/brand/division granularity, bucket features, ALS/market-basket inputs, or per-line monetary signals.  
  - Contains the approved keep-list, `COGS` alias (`Amount2`), and USD-normalized monetary columns.
- `fact_sales_header`  
  - Grain: one row per `Sales_Order`, aggregated deterministically from `fact_sales_line`.  
  - Use for parity vs the legacy header view, transaction-level features (order counts, recency, header revenue/GP), or lightweight analytics without scanning all lines.  
  - Reflects the raw line fact; manual adjustments present in the legacy view are intentionally not reintroduced.
- Keep both: `fact_sales_line` remains the authoritative source; `fact_sales_header` is a deterministic convenience/compatibility layer that speeds migration, parity checks, and transaction-grain features without sacrificing the item detail.

## Config Cheat-Sheet (gosales/config.yaml)
- DB objects: database.source_tables.{sales_detail, product_info, product_tags, analytics_order_tags, customer_asset_rollups}
- Line-items toggle: etl.line_items.use_line_item_facts: false|true
- Dedupe keys: etl.line_items.dedupe.{order_column, item_column, revenue_column, cogs_column, gross_profit_column, term_gross_profit_column, last_update_column}
- Behavior: etl.line_items.behavior.{exclude_line_types, return_treatment, kit_handling, manual_adjustments_documented}
- NEW: add `database.source_tables.analytics_order_tags: dbo.analytics_order_tags` and include it in `database.allowed_identifiers`.
- Line-item sources block now includes `order_tags: dbo.analytics_order_tags` (overridable per environment).
 - Feature toggles: `features.use_usd_monetary`, `enable_canonical_division_features`, `enable_margin_features`, `enable_currency_mix`, `enable_rollup_diversity`, `enable_returns_features`, `enable_order_composition`, `enable_trend_features`, `enable_tag_als`.
- Validation defaults: `validation.holdout_db_object: fact_sales_line` ensures holdouts align with canonical divisions; override or disable when running legacy comparisons.

### Division/Goal Sources (authoritative)
- `analytics_order_tags(Id, tag, goal)`: transaction-aware mapping by product internal id (`Id` ? line `Item_internalid`). Fast path to count items sold and infer Goal at the line level.
- `analytics_product_tags(item_rollup, Goal, …)`: rollup-to-Goal mapping used to resolve Divisions when order-level tags are missing.
- We continue to refer to Goals as “Divisions” in features/models/outputs for naming continuity.

## Next Actions (in priority order)
1) Header aggregation policy (documented in plan)
- If required, add a lightweight "adjusted header" layer to include/exclude manual adjustments by rule (currently we treat legacy adjustments as expected variance and keep raw line-items authoritative).

2) Validation automation adoption
- Wire `gosales.validation.line_item_parity` into CI/ops manifests and document acceptance thresholds for division deltas.
- Align holdout sourcing to curated `fact_sales_line` (or override via config) when generating forward-validation baselines.

3) Holdout sourcing hardening
3) Holdout sourcing hardening
- Point validation holdouts at the line-item source (database.source_tables.sales_detail) and regenerate baseline checks.
- Parity deltas within agreed thresholds at the division level.
- Feature distribution stability (PSI/KS) within configured bands.
- Gains/AUC/PR in tolerance relative to historical runs.
- Determinism: repeated runs with same config produce identical checksums.

## Reference Commands
- Build star (line-items off):
  - `PYTHONPATH="$PWD" python -m gosales.etl.build_star --config gosales/config.yaml --no-use-line-items`
- Build star (line-items on):
  - `PYTHONPATH="$PWD" python -m gosales.etl.build_star --config gosales/config.yaml --use-line-items`
- Score all (line-items on):
  - `PYTHONPATH="$PWD" python -m gosales.pipeline.score_all --use-line-items`
- Probe tables & parity (safe metadata extracts):
  - `PYTHONPATH="$PWD" python scripts/line_item_source_probe.py --sample-top 50`

## Known Gaps / Risks
- Some rollups may lack a `Goal` tag; they are currently treated as `Unknown` until business mapping is provided.
- Optional `Customer_asset_rollups` probe is skipped if schema mismatch is detected; use only as a validator during migration.
- Manual adjustments in the legacy header view are intentionally not replicated at the line level.

## Handoff Notes
- Keep this handoff doc and the migration plan in sync with each change.
- Follow Conventional Commits, keep changes toggled and reversible.
- If uncertain, ask for confirmation before changing the join/identifier rules, keep-list, or currency logic.





