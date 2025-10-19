# GoSales ETL Migration Plan: Transition to Line-Item Sources and Division Standardization

Version: 0.1 (planning draft)
Owner: GoSales Engine maintainers
Status: Proposed (no code changes yet)

This document is the authoritative runbook to migrate the GoSales Engine from legacy header-level views to new, line-item–level tables, while standardizing division mapping via analytics product tags. It is designed to be followed step-by-step and includes explicit gates, diagnostics, and rollback points to ensure determinism, leakage-safety, and end-to-end resilience.

## Executive Summary

- Replace `dbo.saleslog` with `dbo.table_saleslog_detail` as the primary sales fact (line-level grain). Reconstruct header-level aggregations where needed in a controlled, deterministic way.
- Replace `dbo.[Moneyball Assets]` and `dbo.items_category_limited` with `dbo.table_Product_Info_cleaned_headers`, augmented by `dbo.analytics_product_tags` for the canonical Division field (`Goal`).
- Treat `dbo.Customer_asset_rollups` as an optional validator/shortcut during migration; plan to compute these rollups from line items for transparency and leakage-safety.
- Introduce config toggles to enable dual-run A/B comparisons, quick rollback, and controlled cutovers without breaking existing CLIs or downstream artifacts.

## Non‑Negotiables and Invariants

1. Leakage-safety: All joins and aggregations must be cutoff-aware (no post-cutoff data leaks); enforce filters before joins.
2. Determinism: Outputs must be reproducible given the same inputs and config. Sort deterministically before writing artifacts.
3. Resilience: CLIs must run end-to-end even when optional artifacts are missing (ALS/SHAP/telemetry). Prefer graceful fallbacks over hard failures.
4. Config-first: No hard-coded credentials/paths. All environment-dependent settings live in `gosales/config.yaml` (and `config_no_cal.yaml`) and `.env`.
5. Observability: Emit structured logs and validation artifacts documenting assumptions, seeds, and toggles used per run.

## Scope and Source Mapping

| Legacy Source | New Source | Notes |
| --- | --- | --- |
| `dbo.saleslog` | `dbo.table_saleslog_detail` | Migrates from header/transaction grain to line-item grain. Header views reconstructed deterministically.
| `dbo.[Moneyball Assets]` | `dbo.table_Product_Info_cleaned_headers` | Product metadata, including item rollups and attributes.
| `dbo.items_category_limited` | `dbo.table_Product_Info_cleaned_headers` | Item rollups and categories now sourced here.
| Divisions via rollups | `dbo.analytics_order_tags`, `dbo.analytics_product_tags` (`Goal`) | Canonical Division prioritizes transaction-level tags, then rollup-level `Goal`. Define precedence and multi-goal strategy.
| `dbo.Customer_asset_rollups` | Optional (validator/shortcut) | Keep as optional reference; plan deterministic recomputation from lines.

## Design Overview

### Grain and Star Schema

- Fact: `fact_sales_line`
  - Grain: One row per sales line, using a stable unique key (e.g., `(transaction_id, line_number)` or `line_id`).
  - Core fields: `transaction_id`, `line_id/line_number`, `tran_date`, `customer_id`, `item_id`, `quantity`, `unit_price`, `line_amount`, `discount_amount`, `tax_amount`, `ship_amount`, `currency`, flags (`is_return`, `is_tax`, `is_freight`, `is_kit_parent/child`).
  - Division resolution fields now persist `division_goal` / canonical division columns derived from product/tag joins.

- Derived View: `fact_sales_header`
  - Built from `fact_sales_line` by grouping to header-level where needed by existing consumers (summing eligible amounts, excluding tax/freight/discount-only per rules).

- Dimensions
  - `dim_product` from `dbo.table_Product_Info_cleaned_headers` (canonical product key, sku, name, rollup, type, active, brand, uom, effective dates).
  - `dim_product_tags` bridge mapping product key → `Goal` (Division). Handle 1:n with a clear strategy (default: primary-only).
  - `dim_customer` as today, unchanged interfaces; ensure keys consistent with sales facts.
  - `dim_date` standard.

### Fact `fact_sales_line` Specification

- Initial scaffolding for line-item ingestion lives in `gosales/etl/sales_line.py`; integration with the curated star build now writes `fact_sales_line` (table + parquet) whenever `etl.line_items.use_line_item_facts` is enabled.
- `table_saleslog_detail` is treated as the authoritative transaction source, with `Sales_Order` as the canonical order identifier.
- COGS is surfaced directly via `Amount2`, with USD-normalised counterparts added for monetary columns.
- Expose canonical division columns (`item_rollup`, `division_goal`, `division_canonical`) by applying `dbo.analytics_order_tags` first (per `Item_internalid`), then `dbo.analytics_product_tags` as fallback so downstream features consume Goal-aware partitions without losing legacy naming.
- `summarise_sales_header()` aggregates the line-item fact back to a `fact_sales_header` table for parity checks against the legacy view.

- **Grain & Keys**
  - One row per unique (`Sales_Order`, `Item_internalid`, `Revenue`, `Amount2`, `GP`, `Term_GP`) after collapsing multiple snapshots by keeping the row with the latest `last_update`.
  - Preserve supplemental audit fields as needed (e.g., `Invoice`, `PurchaseOrderId`), but `SalesOrderId` will be dropped in favor of `Sales_Order` as the canonical join key.

- **Dedupe Process**
  - Partition raw rows by (`Sales_Order`, `Item_internalid`, `Revenue`, `Amount2`, `GP`, `Term_GP`); order partitions by `last_update` (descending) and retain `ROW_NUMBER() = 1`.
  - Persist a `is_latest_snapshot` flag for audit/backfill scenarios.

- **Core Columns**
  - `order_number`: `Sales_Order` (primary identifier for header-level reporting, joins, and parity analysis).
  - `item_id`: `Item_internalid`.
  - `tran_date`: `Rec_Date`.
  - `revenue`: `Revenue`.
  - `cogs`: expose `Amount2` (source column) as cost-of-goods-sold.
  - `gross_profit`: `GP`.
  - `term_gross_profit`: `Term_GP`.
  - `manual_adjustment_flag`: default `False` (set `True` if/when manual adjustments come through a designated column).
  - Retain legacy seat/qty columns (`*_Qty`) and GP variants to unblock downstream features.

- **Business Rules**
  - Accept that legacy header view may include manual adjustments not present in this fact; document variance and provide reconciliation extracts.
  - Respect `etl.line_items.behavior` toggles: filter disallowed `Rev_type` values, control return handling (`net_amount`/`exclude_returns`/`separate_flag`), and prefer kit children when indicators are present. Log coverage and fallbacks when signals are unavailable.
  - Apply configuration-driven exclusions for `is_tax`, `is_freight`, `is_discount_only`, `is_return`, `is_kit_parent/child` when deriving header-level metrics.

### Column Keep-List (initial)

Keep the following columns from `dbo.table_saleslog_detail` for modeling, features, and audits (subject to currency normalization):

`Rec_Date`, `Sales_Order`, `Branch`, `Division`, `New_Business`, `New`, `Rep`, `Referral_NS_Field`, `Rev_type`, `Item_internalid`,
`Amount2` (as COGS), `Revenue`,
`SWX_Core`, `SWX_Core_Qty`, `Core_New_UAP`, `Core_New_UAP_Qty`, `SWX_Pro_Prem`, `SWX_Pro_Prem_Qty`, `Pro_Prem_New_UAP`, `Pro_Prem_New_UAP_Qty`,
`Simulation`, `Simulation_Qty`, `CAMWorks_Seats`, `CAMWorks_Seats_Qty`, `Misc_Seats`, `Misc_Seats_Qty`, `EPDM_CAD_Editor_Seats`, `EPDM_CAD_Editor_Seats_Qty`,
`HV_Simulation`, `HV_Simulation_Qty`, `SW_Electrical`, `SW_Electrical_Qty`, `SW_Inspection`, `SW_Inspection_Qty`, `_3DX_Revenue`, `_3DX_Revenue_Qty`,
`Training`, `Training_Qty`, `Services`, `Services_Qty`, `Fortus`, `Fortus_Qty`, `_1200_Elite_Fortus250`, `_1200_Elite_Fortus250_Qty`,
`uPrint`, `uPrint_Qty`, `Altium_PCBWorks`, `Altium_PCBWorks_Qty`, `SWX_Core_Seats`, `SWX_Pro_Prem_Seats`, `Core_New_UAP_Seats`, `Pro_Prem_New_UAP_Seats`,
`GP`, `Term_GP`, `Invoice_Date`, `Created_date`, `CompanyId`,
`SW_Plastics`, `SW_Plastics_Qty`, `CATIA`, `CATIA_Qty`, `FDM`, `FDM_Qty`, `AM_Software`, `AM_Software_Qty`, `Polyjet`, `Polyjet_Qty`,
`P3`, `P3_Qty`, `SLA`, `SLA_Qty`, `SAF`, `SAF_Qty`, `Metals`, `Metals_Qty`, `_3DP_Software`, `_3DP_Software_Qty`, `_3DP_Training`, `_3DP_Training_Qty`,
`Post_Processing`, `Post_Processing_Qty`, `Consumables`, `Consumables_Qty`, `FormLabs`, `FormLabs_Qty`, `Other_Misc`, `Other_Misc_Qty`,
`Spare_Parts_Repair_Parts_Time_Materials`, `Spare_Parts_Repair_Parts_Time_materials_Qty`, `Creaform`, `Creaform_Qty`, `Artec`, `Artec_Qty`,
`Success_Plan`, `Success_Plan_Qty`, `Success_Plan_Level`, `AM_Support`, `AM_Support_Qty`, `Delmia`, `Delmia_Qty`, `CPE_YXC_Renewal`, `CPE_YXC_Renewal_Qty`,
`SalesOrder_Currency`, `USD_CAD_Conversion_rate`, `Draftsight`, `GeoMagic`, `Draftsight_Qty`, `GeoMagic_Qty`.

Derived fields surfaced by the scaffolding:

- `COGS` (alias of `Amount2`).
- `Revenue_usd`, `Amount2_usd`, `GP_usd`, `Term_GP_usd` (currency-normalised amounts).
- `manual_adjustment_flag` (currently defaulting to `False`; future toggles may drive this).

### Currency Normalization

- Normalize monetary fields to USD for consistency:
  - If `SalesOrder_Currency = 'CAD'`, compute USD-adjusted fields as `*_usd = * * USD_CAD_Conversion_rate` for `Revenue`, `Amount2` (COGS), `GP`, and `Term_GP`.
  - Otherwise set `*_usd = *`.
  - Surface both raw and USD-normalized values in the fact to preserve auditability.

### Division Roll-ups for New Buckets

- Ensure brand/category buckets such as `Draftsight` and `GeoMagic` roll into the appropriate Divisions via the `analytics_product_tags.Goal` mapping.
- For direct bucket columns, derive per-division aggregates in features by summing the relevant columns according to the mapping, keeping backward-compatible division names.

### Outstanding Tasks (as of migration planning)

- [x] Implement currency normalization in the `fact_sales_line` build and surface `*_usd` columns (available via `gosales/etl/sales_line.py` and written when the line-item toggle is enabled).
- [ ] Evaluate new bucket columns (Draftsight, GeoMagic, etc.) for feature engineering paths once divisions are mapped.
- [x] Recompute header-level aggregates using the deduped fact and validate against legacy adjustments (`fact_sales_header` now written alongside `fact_sales_line`).
- [x] Add configuration toggles and adapters to switch between legacy view and new line-item fact during rollout phases (`score_all.py --use-line-items/--no-use-line-items`, `build_star.py --use-line-items`).
- [x] Ship automated division parity CLI (`python -m gosales.validation.line_item_parity`) writing CSV/JSON deltas under `gosales/outputs/validation/line_item_parity/`.
- [x] Default holdout sourcing to curated `fact_sales_line` with canonical divisions (override via validation.holdout_db_object).

### Division Mapping

- Canonical Division: `Goal` derived from `dbo.analytics_order_tags` (per `Item_internalid`) when available; otherwise fall back to `dbo.analytics_product_tags`.
- Precedence rule (configurable): `order_tags.goal` → `product_tags.Goal` → product header rollup → `"Unknown"`.
- Multi-goal strategy (configurable): `primary_only` (default), `prefer_business_priority`, or `explode_multi` (line duplicated per division with weights = 1/n unless configured otherwise).

## Interrogation Plan (Read‑Only, Safe Queries)

Objective: Understand schemas, keys, shapes, and parity with minimal load. Use small samples, recent date windows, and metadata queries. Avoid full scans.

### 1) Column Metadata

```
-- table_saleslog_detail
SELECT c.name AS column_name, t.name AS data_type, c.max_length, c.is_nullable
FROM sys.columns c
JOIN sys.types t ON c.user_type_id = t.user_type_id
WHERE c.object_id = OBJECT_ID('[dbo].[table_saleslog_detail]')
ORDER BY c.column_id;

-- table_Product_Info_cleaned_headers
SELECT c.name AS column_name, t.name AS data_type, c.max_length, c.is_nullable
FROM sys.columns c
JOIN sys.types t ON c.user_type_id = t.user_type_id
WHERE c.object_id = OBJECT_ID('[dbo].[table_Product_Info_cleaned_headers]')
ORDER BY c.column_id;

-- analytics_product_tags
SELECT c.name AS column_name, t.name AS data_type, c.max_length, c.is_nullable
FROM sys.columns c
JOIN sys.types t ON c.user_type_id = t.user_type_id
WHERE c.object_id = OBJECT_ID('[dbo].[analytics_product_tags]')
ORDER BY c.column_id;
```

### 2) Candidate Keys and Uniqueness

```
-- Sales line uniqueness: try (transaction_id, line_number). If duplicates exist, inspect alternative keys (e.g., line_id).
SELECT TOP (1) 1
FROM (
  SELECT transaction_id, line_number, COUNT(*) AS cnt
  FROM dbo.table_saleslog_detail WITH (NOLOCK)
  GROUP BY transaction_id, line_number
  HAVING COUNT(*) > 1
) dups;

-- Product key consistency (choose stable key present in both sources)
SELECT TOP (100) item_id, item_sku, item_rollup
FROM dbo.table_Product_Info_cleaned_headers WITH (NOLOCK);

-- Tag multiplicity per product
SELECT product_key_column, COUNT(DISTINCT Goal) AS goals
FROM dbo.analytics_product_tags WITH (NOLOCK)
GROUP BY product_key_column
ORDER BY goals DESC;
```

### 3) Shape, Volume, and Date Coverage

```
-- Monthly line counts (last 24 months)
SELECT FORMAT(tran_date, 'yyyy-MM') AS yyyymm, COUNT_BIG(*) AS line_count
FROM dbo.table_saleslog_detail WITH (NOLOCK)
WHERE tran_date >= DATEADD(MONTH, -24, GETDATE())
GROUP BY FORMAT(tran_date, 'yyyy-MM')
ORDER BY yyyymm;

-- Distinct entities (last 12 months)
SELECT COUNT(DISTINCT customer_id) AS customers,
       COUNT(DISTINCT item_id) AS items,
       COUNT(DISTINCT transaction_id) AS transactions
FROM dbo.table_saleslog_detail WITH (NOLOCK)
WHERE tran_date >= DATEADD(MONTH, -12, GETDATE());
```

### 4) Parity vs Legacy Header View (if available)

```
DECLARE @start DATE = DATEADD(MONTH, -6, GETDATE());
DECLARE @end   DATE = GETDATE();

;WITH line_header AS (
  SELECT transaction_id,
         SUM(extended_amount) AS total_amount,
         MIN(tran_date) AS tran_date
  FROM dbo.table_saleslog_detail WITH (NOLOCK)
  WHERE tran_date BETWEEN @start AND @end
  GROUP BY transaction_id
)
SELECT 'only_in_lines' AS diff_side, COUNT(*) AS n
FROM line_header lh
LEFT JOIN dbo.saleslog s ON s.transaction_id = lh.transaction_id
WHERE s.transaction_id IS NULL
UNION ALL
SELECT 'only_in_saleslog', COUNT(*)
FROM dbo.saleslog s
LEFT JOIN line_header lh ON lh.transaction_id = s.transaction_id
WHERE lh.transaction_id IS NULL
UNION ALL
SELECT 'mismatched_amounts', COUNT(*)
FROM dbo.saleslog s
JOIN line_header lh ON lh.transaction_id = s.transaction_id
WHERE ABS(ISNULL(s.total_amount,0) - ISNULL(lh.total_amount,0)) > 0.01;
```

### 5) Division Tag Coverage and Strategy Inputs

```
-- Coverage of Goal
SELECT COUNT(*) AS rows,
       SUM(CASE WHEN Goal IS NULL OR Goal = '' THEN 1 ELSE 0 END) AS null_goal
FROM dbo.analytics_product_tags WITH (NOLOCK);

-- Multi-goal products and whether a primary/priority exists
SELECT product_key_column,
       COUNT(*) AS tag_rows,
       COUNT(DISTINCT Goal) AS goal_count
FROM dbo.analytics_product_tags WITH (NOLOCK)
GROUP BY product_key_column
ORDER BY goal_count DESC, tag_rows DESC;
```

### 6) Joinability: Sales ↔ Product Info

```
SELECT TOP (200)
       s.item_id,
       p.item_id AS p_item_id,
       p.item_sku,
       p.item_rollup
FROM dbo.table_saleslog_detail s WITH (NOLOCK)
LEFT JOIN dbo.table_Product_Info_cleaned_headers p
  ON p.item_id = s.item_id
WHERE s.tran_date >= DATEADD(MONTH, -3, GETDATE())
ORDER BY s.tran_date DESC;
```

> Notes: Use `WITH (NOLOCK)` only for exploration. Production ETL should use consistent isolation and idempotent reads.

## Implementation Plan (Step‑By‑Step)

### Phase 0 — Prepare and Gate

- Create branch `refactor/etl-line-item-sources`.
- Add config toggles (see below) with defaults that preserve current behavior until explicitly flipped.
- Define acceptance thresholds for A/B parity (see Validation section).
- Document run contexts and seeds.

### Phase 1 — Build Line-Item Fact and Header Reconstruction

- Implement `fact_sales_line` from `dbo.table_saleslog_detail` with:
  - Strict cutoff filter on `tran_date` applied before joins.
  - Stable unique key; fail fast if uniqueness violated with actionable log.
  - Line classification rules (configurable) to tag `is_tax`, `is_freight`, `is_discount_only`, `is_return`, `is_kit_parent/child`.
  - Deterministic sort before write: `customer_id, tran_date, transaction_id, line_id/line_number`.
  - Deduplicate raw rows by retaining the latest `last_update` per (`SalesOrderId`, `Item_internalid`, `Revenue_account_use`, `Revenue`) to collapse replicated line snapshots.
- Implement `fact_sales_header` by grouping `fact_sales_line`:
  - Exclude `tax/freight/discount_only` per config.
  - Returns handled as `net_amount` by default; configurable.
- Instrument table interrogation via `scripts/line_item_source_probe.py` (outputs captured under `docs/appendices/migration_line_item/` and logged in the workbook).

### Phase 2 — Integrate Product Info and Division Tags

- Build `dim_product` from `dbo.table_Product_Info_cleaned_headers`.
- Resolve Division with `dbo.analytics_product_tags`:
  - Precedence: tags `Goal` → product header rollup → `Unknown`.
  - Multi-goal strategy: `primary_only` (default) unless business chooses otherwise.
- Provide a compatibility mapping to legacy `item_rollup` when needed.

### Phase 3 — Adapters and Backward Compatibility

- Add `gosales/etl/adapters.py` exposing functions used by legacy feature code:
  - `get_sales_header_like(cutoff, config)` → DataFrame shaped like legacy `saleslog` aggregations.
  - `get_item_rollup_like(config)` → mapping consistent with legacy expectations.
  - `get_division(config)` → canonical Division per precedence rules.
- Keep existing CLIs running by switching sources under the adapter before refactoring downstream code.

### Phase 4 — Features and Labels on Line Grain

- Update RFM and temporal aggregations to use `fact_sales_line` (or header view where defined), with explicit rules for:
  - Whether multiple lines on same transaction count once (transaction frequency) or per line (line frequency).
  - Exclusion of tax/freight lines in monetary features.
- Division-aware features using `Goal`.
- Branch/rep share and ACR/New flag features now source from `fact_sales_line`; legacy fallbacks to `sales_log`/`fact_orders` have been removed (line-item toggle required).
- Re-evaluate ALS and market-basket features using line-level baskets grouped by `transaction_id` (seed any stochastic steps).
- Labels: continue to enforce censoring windows; define label grain (customer-period / customer-division) clearly.

ALS updates (line-item aware)
- Transaction ALS now trains on `fact_transactions` with weights combining quantity and positive gross_profit via `log1p` transforms to emphasize meaningful purchases while tempering extremes.
- Embedding windows are restricted to `lookback_months` before the cutoff and honor an embargo of `features.affinity_lag_days` to avoid near-cutoff leakage.
- Assets ALS remains optional and is size-guarded; it uses rollup columns at cutoff and does not include post-cutoff information.

### Phase 5 — Validation, Drift, and A/B Parity

- Dual-run old (views) vs new (tables) in the same environment for selected cutoffs and windows.
- Compare aggregates (per division): revenue, active customers, transactions, adoption counts.
- Compute PSI/KS on key features; log deltas with thresholds.
- Emit diff artifacts in `gosales/outputs/validation/...` and summarize in logs.

### Phase 6 – Flip Defaults and Deprecate Legacy Views

- Switch `etl.line_items.use_line_item_facts: true` and set source tables as defaults.
- Keep a fallback flag for one release cycle; then remove legacy view dependencies.
- Update docs and notify downstream consumers.

## Configuration (Proposed Keys and Defaults)

Add to `gosales/config.yaml` and mirror in `gosales/config_no_cal.yaml`. Validate in `gosales.utils.config` with helpful errors.

```yaml
sources:
  sales_detail: "[dbo].[table_saleslog_detail]"
  product_info: "[dbo].[table_Product_Info_cleaned_headers]"
  product_tags: "[dbo].[analytics_product_tags]"
  customer_asset_rollups: "[dbo].[Customer_asset_rollups]"  # optional

etl:
  line_items:
    use_line_item_facts: true  # default true; disable only for emergency rollback
    sources:
      sales_detail: "[dbo].[table_saleslog_detail]"
      product_info: "[dbo].[table_Product_Info_cleaned_headers]"
      product_tags: "[dbo].[analytics_product_tags]"
      order_tags: "[dbo].[analytics_order_tags]"
      customer_asset_rollups: "[dbo].[Customer_asset_rollups]"
    dedupe:
      order_column: "Sales_Order"
      item_column: "Item_internalid"
      revenue_column: "Revenue"
      cogs_column: "Amount2"
      gross_profit_column: "GP"
      term_gross_profit_column: "Term_GP"
      last_update_column: "last_update"
    behavior:
      exclude_line_types: ["tax", "freight", "discount_only"]
      return_treatment: "net_amount"  # options: net_amount | exclude_returns | separate_flag
      kit_handling: "prefer_children"
      manual_adjustments_documented: true
  division:
    precedence: ["product_tags", "product_info_rollup"]
    multi_goal_strategy: "primary_only"  # primary_only | prefer_business_priority | explode_multi
  validation:
    dual_run: true
    holdout_db_object: "fact_sales_line"
    holdout_source: "auto"
    parity_thresholds:
      revenue_abs_pct: 0.5   # allowed absolute pct diff in revenue aggregates
      transactions_abs_pct: 0.5
      adoption_abs_pct: 1.0

random:
  seed: 1729
```

> Notes:
> - Keep path resolution via `gosales.utils.paths` only; do not hardcode filesystem locations.
> - Ensure toggles are surfaced by CLIs and documented in `--help`.

## Replacement Rules (Old → New)

- `dbo.saleslog` → `fact_sales_header` built from `fact_sales_line`:
  - Amounts: sum of `line_amount` excluding `tax/freight/discount_only` (configurable).
  - Transaction count: distinct `transaction_id` within window (unless configured otherwise).
  - Recency: min(`tran_date`) or max(`tran_date`) per definition; must be explicitly set and documented.

- `dbo.[Moneyball Assets]` and `dbo.items_category_limited` → `dim_product` + `dim_product_tags`:
  - Any `item_rollup` fields sourced from product headers.
  - Canonical Division from tags (`Goal`), with precedence/fallback rules.

- `dbo.Customer_asset_rollups` → optional validator:
  - Compare our computed rollups per customer-division to the view; log row deltas and aggregate diffs.
  - Plan to fully deprecate after one release when parity is stable and business-accepted.

## Feature Engineering Adjustments

- Monetary features: sum line amounts per rules; currency normalization if multi-currency exists.
- Frequency features: define whether per-transaction or per-line; expose as config.
- Line-item feature families (all toggleable under `features.*`):
  - Canonical division revenue/GP windows (`xdiv__canon_*`), USD margin rates (`margin__*`), and currency mix metrics (`currency__*`).
  - Rollup/tag diversity, returns intensity, and optional tag-share embeddings (`diversity__rollup_*`, `returns__*`, `tagals__share_*`).
  - Order composition features (lines/order, revenue per line/order) plus branch/rep coverage shares for top assignments.
  - All new columns zero-fill and respect config toggles: `use_usd_monetary`, `enable_canonical_division_features`, `enable_margin_features`, `enable_currency_mix`, `enable_rollup_diversity`, `enable_returns_features`, `enable_order_composition`, `enable_trend_features`, `enable_tag_als`.
- Recency features: compute from `tran_date` at the chosen grain (customer or customer-division).
- Adoption features: derive from line items joined to Division; create binary adoption flags and counts per cutoff window.
- Embeddings/ALS: base on baskets per transaction/customer-period; seed randomness; zero-fill when disabled or missing.

## Labels and Censoring

- Ensure censoring windows are applied before any forward look.
- Explicitly define label grain (customer-period or customer-division) and ensure that training windows do not leak post-cutoff information.
- Maintain denylist handling unchanged; validate columns exist after refactor.

## Modeling, Ranking, and Cooldowns

- Refresh feature catalogs to match any renamed/added features; keep backward compatibility via adapters until fully migrated.
- Rank by canonical Division (`Goal`); ensure deterministic tie-breaking (`customer_id`, `division`, `stable_secondary`).
- Cooldown/capacity logic unchanged; inputs must remain deterministic.

## Validation, Drift, and A/B Parity Gates

Gates are mandatory; do not flip defaults if any gate fails.

1. Schema validation: no missing required columns; candidate key uniqueness holds for `fact_sales_line`.
2. Aggregate parity (per division over validation windows):
   - Revenue absolute percentage delta ≤ `revenue_abs_pct` threshold.
   - Transactions absolute percentage delta ≤ `transactions_abs_pct` threshold.
   - Adoption counts absolute percentage delta ≤ `adoption_abs_pct` threshold.
3. Distributional stability: PSI or KS for top 20 predictive features within acceptable bands; investigate any outliers.
4. Model quality: gains/AUC/PR remain within historical tolerance bands (documented in metrics output).
5. Determinism: Repeat runs with same config produce identical checksums for artifacts.

Emit validation artifacts under `gosales/outputs/validation/...` and summarize the decision in logs and this doc (appendix entries).

## Observability and Diagnostics

- Structured logging via `gosales.utils.logger.get_logger` with `extra` fields: source toggles, cutoff, window, exclusion rules, counts dropped by line types.
- Diagnostics artifacts:
  - A/B diff CSVs per division (revenue, transactions, adoption, top N items).
  - Mismatch summaries vs `Customer_asset_rollups`.
  - Run-context manifest (config snapshot, seeds, code version/branch).

## Risks and Mitigations

- Multi-goal products (1:n mapping): default to `primary_only`. If business needs multi-division attribution, use `explode_multi` with careful downstream handling.
- Freight/tax/discount lines: misclassification can skew monetary features. Keep explicit classification rules and tests.
- Returns/credits: set treatment to `net_amount` by default; revisit for churn-sensitive labels.
- Kits/bundles: avoid double-counting; prefer child lines over parent kit placeholders.
- Late-arriving data: reinforce cutoff filters; log late-arrival counts.
- Performance: prefer vectorized operations; avoid quadratic loops; add indexes if DB-side transformations added.

## Step-by-Step Runbook (Operator View)

1) Create branch (once approved):
   - `git checkout -b refactor/etl-line-item-sources`

2) Add config keys (no behavior change yet) and run static checks:
   - Update `gosales/config.yaml` and `gosales/config_no_cal.yaml` with proposed keys.
   - `ruff check gosales` and `black gosales` (formatting only).

3) Interrogate tables (off-hours, safe queries):
   - Capture column metadata and sample shapes into `docs/` as appendices.

4) Implement `fact_sales_line` and `fact_sales_header` behind toggles:
4) Implement `fact_sales_line` and `fact_sales_header` behind toggles:
   - Ensure the line-item toggle remains enabled during validation runs (disable only for emergency rollback).

5) Integrate product info and tags; define Division strategy; expose adapter functions.

6) Dual-run A/B for selected divisions and cutoffs:
   - `PYTHONPATH="$PWD" python -m gosales.pipeline.rank_whitespace --cutoff "YYYY-MM-DD" --window-months 6 --config gosales/config.yaml`
   - Compare outputs; store diffs under `gosales/outputs/validation/`.

7) If all gates pass, flip defaults:
   - Maintain the line-item default as true; legacy fallbacks are now removed after validation.

8) Deprecate legacy view dependencies and update docs/tests accordingly.
- All CLIs (`gosales.pipeline.*`, `gosales.validation.*`, `gosales.whitespace.*`) must remain operational regardless of toggle settings.
- `gosales.pipeline.score_all` and `gosales.etl.build_star` now accept `--use-line-items/--no-use-line-items` flags that map to the runtime toggle (`etl.line_items.use_line_item_facts`). Extend remaining entry points as needed.
- Surface division mapping options via CLI or config; default to config.

## Testing Strategy

- Unit tests for adapters to guarantee legacy contract shapes.
- Targeted tests for line classification rules and header reconstruction (tax/freight/returns/kits).
- Feature and label determinism tests (checksum assertions for sample fixtures).
- End-to-end tests for ranking determinism and division mapping.
- Skip/mark slow suites (ALS/SHAP) unless explicitly enabled; seed randomness.
- Validation CLI: `PYTHONPATH="$PWD" python -m gosales.validation.line_item_parity --config gosales/config.yaml --cutoff "<ISO-date>"` (writes division delta CSV/JSON under `gosales/outputs/validation/line_item_parity/`).

Run locally:

```
pytest gosales/tests -q
pytest gosales/tests/test_phase4_rank_normalization.py -q
```

## Acceptance Criteria (Go/No-Go)

- All tests green under both old and new sources (via dual-run).
- All validation gates pass within thresholds; any exceptions documented and signed off.
- Deterministic artifacts (checksums unchanged across repeated runs with same config).
- Documentation updated (this plan, artifact catalog, feature/label docs).
- Optional: parity confirmed against `dbo.Customer_asset_rollups`, with deltas understood or reconciled.

## Appendix A — SQL Playbook (Safe Sampling)

```
-- Quick sample of recent lines
SELECT TOP (200) *
FROM dbo.table_saleslog_detail WITH (NOLOCK)
ORDER BY tran_date DESC, transaction_id, line_number;

-- Distinct Goals inventory
SELECT Goal, COUNT(*) AS rows
FROM dbo.analytics_product_tags WITH (NOLOCK)
GROUP BY Goal
ORDER BY rows DESC;
```

## Appendix B — Field Mapping Checklist (To Be Completed During Interrogation)

- Sales lines required fields and their column names in `table_saleslog_detail`:
  - transaction key: `...`
  - line key: `...`
  - customer key: `...`
  - product key: `...`
  - dates: `...`
  - quantities/prices/amounts: `...`
  - line type/flags: `...`

- Product info required fields in `table_Product_Info_cleaned_headers`:
  - stable product key: `...`
  - sku/name/rollup/type/active/uom/brand: `...`

- Division mapping in `analytics_product_tags`:
  - product key used: `...`
  - `Goal` primary flag/priority: `...`
  - multiplicity rule: `...`

Complete and check in this section once interrogation is done.

