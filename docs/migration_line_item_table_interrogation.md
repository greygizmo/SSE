# Table Interrogation Workbook - Line-Item ETL Migration

This workbook accompanies `docs/migration_line_item_etl_plan.md` and captures the schema discovery, key validation, and coverage checks for the new source tables. Fill in each section with query results, observations, and follow-up actions to maintain an auditable trail.

> Note: The queries below are read-only and scoped to metadata or limited time windows to avoid heavy load. Execute them in a staging window whenever possible. Record execution timestamps and credentials used (without embedding secrets) for reproducibility. You can automate these probes via `scripts/line_item_source_probe.py`, which will write sanitized CSV outputs under `docs/appendices/migration_line_item/`.

## Environment Checklist

- Server: `db-goeng-netsuite-prod`
- Database: `dbo`
- Authentication method: _[record here]_ (e.g., Windows auth / SQL auth)
- Tool used: _[SQL Server Management Studio / sqlcmd / PowerShell Invoke-Sqlcmd]_
- Execution window: _[YYYY-MM-DD hh:mm]_ (local timezone)
- 2025-10-15 14:17 (local): Automated probe via `scripts/line_item_source_probe.py` failed. Error: `RuntimeError: Failed to connect to Azure SQL via pyodbc ... [IM002] Data source name not found and no default driver specified`. Action: verify Microsoft ODBC Driver 18/17 for SQL Server is installed and accessible to 64-bit Python, or configure alternative driver/DSN before rerunning.
- 2025-10-15 14:29 (local): Manual credential test shows driver resolved but SQL authentication failed (`Login failed for user 'treid'`). Update `.env` with a working SQL login (format may require fully qualified user) or verify password, then re-run the probe script.
- 2025-10-15 16:10 (local): Probe script connected successfully; generated column metadata CSVs under `gosales/docs/appendices/migration_line_item/` but halted on duplicate-key scan because `transaction_id`/`line_number` columns do not exist. Follow-up: re-run after updating script to use `SalesOrderId`/dedupe keys.
- 2025-10-15 16:38 (local): Updated probe script (with customizable keys) ran end-to-end; all diagnostics except the optional customer rollup sample succeeded. New CSVs stamped `20251015_2238xx__*.csv` saved under `gosales/docs/appendices/migration_line_item/`.
- 2025-10-16 16:42 (local): `build_star` now writes the new `fact_sales_line` artifact when `etl.line_items.use_line_item_facts` is enabled (parquet + database table); parity extracts updated using `Sales_Order` identifiers (`20251015_230700__parity_by_order.csv`).
- `fact_sales_header` is produced alongside the line fact via `summarise_sales_header()` for header-level reconciliation.
## 1. Column Metadata

### Summary Findings (2025-10-15)

- ODBC Driver 18 connectivity restored after credential update; subsequent probes succeed.
- Column metadata captured for sales detail (147 columns), product info (17 columns), and product tags (38 columns); CSV snapshots saved under `gosales/docs/appendices/migration_line_item/`.
- Sales detail lacks `transaction_id`/`line_number`; treat (`SalesOrderId`, `Item_internalid`) as working grain and dedupe by latest `last_update`.

### `dbo.table_saleslog_detail`

```sql
SELECT c.name AS column_name,
       t.name AS data_type,
       c.max_length,
       c.is_nullable,
       c.column_id
FROM sys.columns c
JOIN sys.types t ON c.user_type_id = t.user_type_id
WHERE c.object_id = OBJECT_ID('[dbo].[table_saleslog_detail]')
ORDER BY c.column_id;
```

- Observations:
  - Table is line-grained with legacy header fields plus new numeric signals; primary order identifiers are `Sales_Order` (text) and `SalesOrderId` (int).
  - No `transaction_id` column exists; we must treat `SalesOrderId` + item identifiers as the new grain.
  - Row-level dedupe fields present: `Created_date`, `last_update`, `PurchaseOrderId`; duplicates observed share `SalesOrderId`/`Item_internalid` but differ in `last_update`.
  - Duplicate scan confirms duplicates exist (`has_duplicates = 1`); top offenders show counts of 4 rows per order-item, reinforcing the need to keep the most recent `last_update`.
  - Monetary columns (`Revenue`, `Amount2`, etc.) are already numeric; ensure we select latest `last_update` per order-item to avoid double-counting.

### `dbo.table_Product_Info_cleaned_headers`

```sql
SELECT c.name AS column_name,
       t.name AS data_type,
       c.max_length,
       c.is_nullable,
       c.column_id
FROM sys.columns c
JOIN sys.types t ON c.user_type_id = t.user_type_id
WHERE c.object_id = OBJECT_ID('[dbo].[table_Product_Info_cleaned_headers]')
ORDER BY c.column_id;
```

- Observations:
  - Contains canonical product identifiers (`internalid`, `Product_Internal_ID`) plus rollup (`item_rollup`) in a single table.
  - Columns are stored as text; we must coerce numeric IDs (`internalid`, `Product_Internal_ID`) when loading.
  - This schema can replace `dbo.[Moneyball Assets]` and `dbo.items_category_limited`; ensure we retain `Status`/`IsRenewable` flags for asset logic.

### `dbo.analytics_product_tags`

```sql
SELECT c.name AS column_name,
       t.name AS data_type,
       c.max_length,
       c.is_nullable,
       c.column_id
FROM sys.columns c
JOIN sys.types t ON c.user_type_id = t.user_type_id
WHERE c.object_id = OBJECT_ID('[dbo].[analytics_product_tags]')
ORDER BY c.column_id;
```

- Observations:
  - Minimal schema: `item_rollup` -> `Goal` plus one-hot indicator columns for legacy categories.
  - Only 42 item rollups present; `Goal` missing for exactly one rollup (needs business mapping).
  - No explicit priority flag--mapping is 1:1 today, so we can treat `Goal` as canonical division but keep logic extensible for future multi-goal cases.

## 2. Candidate Keys & Uniqueness Checks

### Sales Line Grain

```sql
-- Validate uniqueness of (transaction_id, line_number)
SELECT TOP (1) 1
FROM (
  SELECT transaction_id,
         line_number,
         COUNT(*) AS cnt
  FROM dbo.table_saleslog_detail WITH (NOLOCK)
  GROUP BY transaction_id, line_number
  HAVING COUNT(*) > 1
) dups;

-- If duplicates exist, inspect alternative key (e.g., line_id)
SELECT TOP (100)
       transaction_id,
       line_number,
       line_id,
       COUNT(*) AS cnt
FROM dbo.table_saleslog_detail WITH (NOLOCK)
GROUP BY transaction_id, line_number, line_id
HAVING COUNT(*) > 1
ORDER BY cnt DESC;
```

- Findings:
  - Legacy keys (`transaction_id`/`line_number`) are absent; we will use `Sales_Order` (text) as the canonical order identifier and `Item_internalid` as the item key.
  - Numeric `SalesOrderId` is retained only in source data; downstream facts and joins should rely on `Sales_Order`.
  - Using `Sales_Order + Item_internalid` still yields duplicates (up to 4 identical rows) differing only by `last_update` (see `sales_line_duplicate_examples.csv`).
  - Proposed dedupe: partition by (`Sales_Order`, `Item_internalid`, `Revenue`, `Amount2`, `GP`, `Term_GP`) and retain the row with max(`last_update`). Need validation against other attribute combos (e.g., success-plan vs seat lines).

### Division Tag Multiplicity

```sql
SELECT product_key_column,
       COUNT(*) AS tag_rows,
       COUNT(DISTINCT Goal) AS goal_count
FROM dbo.analytics_product_tags WITH (NOLOCK)
GROUP BY product_key_column
ORDER BY goal_count DESC, tag_rows DESC;
```

- Findings:
  - Current data shows 42 rollups, each mapping to exactly one `Goal` (no multi-goal cases yet).
  - One rollup lacks a goal; document and resolve (fallback to `Unknown` until business mapping supplied).
  - Retain infrastructure for multi-goal handling in case additional tags arrive later.

## 3. Shape and Coverage

### Monthly Line Counts

```sql
SELECT FORMAT(tran_date, 'yyyy-MM') AS yyyymm,
       COUNT_BIG(*) AS line_count
FROM dbo.table_saleslog_detail WITH (NOLOCK)
WHERE tran_date >= DATEADD(MONTH, -24, GETDATE())
GROUP BY FORMAT(tran_date, 'yyyy-MM')
ORDER BY yyyymm;
```

- Notes:
  - Monthly line counts (last 24 months) span roughly 7.3K–11.2K lines (2023-10 through 2025-09), with the current month (2025-10) partially loaded at 2.9K lines.
  - Volume roughly 1.4-1.6x header counts due to line-level granularity--expect higher totals than legacy view.

### Distinct Entities (recent 12 months)

```sql
SELECT COUNT(DISTINCT customer_id) AS customers,
       COUNT(DISTINCT item_id) AS items,
       COUNT(DISTINCT transaction_id) AS transactions
FROM dbo.table_saleslog_detail WITH (NOLOCK)
WHERE tran_date >= DATEADD(MONTH, -12, GETDATE());
```

- Notes:
  - Last 12 months: 22,984 distinct customers (`CompanyId`), 5,200 distinct items (`Item_internalid`), 45,846 sales orders.
  - Customer counts align with legacy view; item cardinality higher due to inclusion of hardware/service SKUs previously aggregated.

## 4. Legacy Parity Diagnostics (if `dbo.saleslog` accessible)

```sql
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

- Outcome:
  - Legacy `dbo.saleslog` only retains header-level revenue per order; line table (after dedupe) sums to materially higher revenue because it includes per-item detail.
  - Parity summary (last 6 months): 1,748 orders appear only in the new line table, none exist solely in the legacy view, and 8,905 orders have mismatched totals.
  - Six-month aggregate check: deduped line table totals approx. $253.3M across 23,113 orders vs legacy view $150.3M across 21,365 orders. Investigate whether legacy excluded certain buckets (e.g., seat renewals) or if additional filters (tax/freight, credit handling) are required.
- Detailed parity extract (`20251015_230700__parity_by_order.csv`) shows 10,635 orders with residual differences > $0.01; top variances correspond to large renewal bundles where the legacy view recorded only a subset of revenue/GP.
  - Orders with missing legacy revenue (i.e., present only in the new fact): 1,729.
  - Business note: legacy `dbo.saleslog` captures manual revenue adjustments; treat the observed deltas as expected when comparing to the raw line-item fact.
  - Need bespoke parity check: aggregate deduped line items to headers and compare to legacy totals, expecting alignment once duplicates removed and non-revenue lines filtered.
  - Observed example: SalesOrderId `9598491` yields 971.25 in legacy view vs 971.25 + 3,885.00 in line table--confirm whether old view intentionally excluded certain revenue buckets (e.g., seat renewals).
- Legacy view lacks `Revenue_account_use`, so revenue-account parity can only be reported from the new table; capture new-only totals separately for business validation (still pending mapping).
- Currency normalization (CAD -> USD) still needs to be implemented in the fact build; monitor the USD-adjusted fields once available.

## 5. Division Coverage and Strategy Inputs

```sql
SELECT COUNT(*) AS rows,
       SUM(CASE WHEN Goal IS NULL OR Goal = '' THEN 1 ELSE 0 END) AS null_goal
FROM dbo.analytics_product_tags WITH (NOLOCK);

SELECT Goal,
       COUNT(*) AS rows
FROM dbo.analytics_product_tags WITH (NOLOCK)
GROUP BY Goal
ORDER BY rows DESC;
```

- Outcome:
  - `analytics_product_tags` has 42 rows; 1 rollup with null/blank `Goal`.
  - Top Goals by row count: Printers (9), Specialty Software (8), CAD (5), Printer Accessorials (4), Training/Services (4); each rollup still maps to a single `Goal`.
  - Division taxonomy aligns with legacy `item_rollup` categories; we can migrate division logic with minimal friction.

## 6. Joinability - Sales vs Product Info

```sql
SELECT TOP (200)
       s.item_id,
       p.item_id AS p_item_id,
       p.item_sku,
       p.item_rollup,
       s.tran_date
FROM dbo.table_saleslog_detail s WITH (NOLOCK)
LEFT JOIN dbo.table_Product_Info_cleaned_headers p
  ON p.item_id = s.item_id
WHERE s.tran_date >= DATEADD(MONTH, -3, GETDATE())
ORDER BY s.tran_date DESC;
```

- Observations:
  - Joins on `Item_internalid` succeed for sampled recent rows; no null matches observed in the spot check.
  - Need to account for items lacking `item_rollup` in product headers (if any) to avoid orphaned lines.
  - Sample join extract (`sales_vs_product_join_sample.csv`) includes order_id, item_key, product_key, product rollup, line amount, and transaction date for quick audits.
  - `SalesOrderId` + item metadata confirm we must reconstruct transaction header metrics after dedupe.

## 7. Optional - Customer Asset Rollups Benchmark

```sql
SELECT TOP (200)
       car.customer_id,
       car.item_rollup,
       car.asset_count,
       car.last_purchase_date
FROM dbo.Customer_asset_rollups car WITH (NOLOCK)
ORDER BY car.last_purchase_date DESC;
```

- Observations:
  - View remains available and can validate customer-level adoption counts, but schemas differ (rollup still header-grained).
  - Use during migration as QA reference; long term prefer recomputing rollups from deduped line fact.
  - Automated probe currently skips the extraction because expected columns (`customer_id`, `item_rollup`, etc.) are absent--confirm actual schema before relying on scripted samples.

## Appendices

- Appendix A: Raw query outputs (store snapshots under `docs/appendices/migration_line_item/` as CSV/Markdown).
- Captured CSVs (2025-10-15/16 runs, stamped `20251015_2238xx__*` and `20251015_230700__*`): `sales_detail_columns`, `product_info_columns`, `product_tags_columns`, `sales_line_duplicate_scan`, `sales_line_duplicate_examples`, `sales_line_monthly_counts`, `sales_line_distinct_entities`, `product_tag_multiplicity`, `legacy_parity_summary`, `parity_by_order`, `division_goal_coverage`, `division_goal_inventory`, `sales_vs_product_join_sample` under `gosales/docs/appendices/migration_line_item/`.
- Appendix B: Summary of field mappings (populate template in `migration_line_item_etl_plan.md`).
- Appendix C: Decisions log (e.g., selected Division strategy, agreed thresholds).

> Remember to remove or redact any sensitive values before committing outputs. The workbook should contain only metadata, counts, and anonymized samples.
