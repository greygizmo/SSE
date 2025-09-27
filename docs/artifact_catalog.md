# GoSales Artifact Catalog

This catalog enumerates every artifact the GoSales Engine produces across the pipeline and explains why each file matters. Use it as a runbook when reviewing outputs, wiring downstream consumers, or triaging anomalies.

## Phase 0 — ETL, Star Schema, and Contracts

| Artifact | Location Pattern | Purpose | Importance |
| --- | --- | --- | --- |
| `fact_transactions.parquet` | `gosales/outputs/star/` | Curated transactional fact table with enrichment and fallback logic applied. | Forms the backbone for features, labels, and validation; provides a leakage-safe transaction history.【F:README.md†L10-L13】 |
| `dim_customer.parquet` | `gosales/outputs/star/` | Cleansed customer dimension with fuzzy resolution for missing keys. | Establishes a consistent customer universe and joins metadata used by features and reporting.【F:README.md†L10-L13】 |
| `fact_events.parquet` | `gosales/outputs/star/` | Eventized roll-up by invoice with per-model label stamps. | Enables label generation and cohorting while retaining invoice granularity for audits.【F:README.md†L69-L73】 |
| `schema_snapshot.json` | `gosales/outputs/star/contracts/` | Captures column names, types, and constraints after curation. | Provides regression guardrails—diffs highlight unintended schema drift.【F:README.md†L11-L12】 |
| `row_counts.csv` | `gosales/outputs/star/contracts/` | Counts records for each curated table. | Lets operators confirm pipeline completeness and spot truncation early.【F:README.md†L11-L12】 |
| `contract_violations.csv` | `gosales/outputs/star/contracts/` | Lists records failing contract rules (PK, date bounds). | Surfaces data quality issues that can invalidate downstream modeling if ignored.【F:README.md†L11-L12】 |
| `checksums.json` | `gosales/outputs/star/contracts/` | Hashes of curated tables. | Ensures deterministic builds and supports reproducibility investigations.【F:README.md†L11-L12】 |

## Phase 1 — Labeling

| Artifact | Location Pattern | Purpose | Importance |
| --- | --- | --- | --- |
| `labels_<division>_<cutoff>.parquet` | `gosales/outputs/labels/` | Leakage-safe targets for each (customer, division, cutoff). | Training truth set; every model and validation task depends on accurate labels.【F:README.md†L14-L19】【F:README.md†L197-L198】 |
| `label_prevalence_<division>_<cutoff>.csv` | `gosales/outputs/labels/` | Positive rate summary for each cohort. | Highlights class imbalance and data sufficiency, guiding modeling choices.【F:README.md†L14-L19】【F:README.md†L191-L199】 |
| `cutoff_manifest_<division>.json` | `gosales/outputs/labels/` | Records window parameters and censoring decisions. | Documents assumptions per run, enabling reproducibility and audit trails.【F:README.md†L14-L19】 |

## Phase 2 — Feature Engineering

| Artifact | Location Pattern | Purpose | Importance |
| --- | --- | --- | --- |
| `features_<division>_<cutoff>.parquet` | `gosales/outputs/features/` | Model-ready matrix with RFM, lifecycle, seasonality, cross-division, and optional embeddings. | Primary input to model training; quality directly impacts performance.【F:README.md†L20-L26】 |
| `feature_catalog_<division>_<cutoff>.csv` | `gosales/outputs/` | Lists every engineered feature with coverage. | Supports explainability, data dictionary tasks, and downstream schema contracts.【F:README.md†L20-L26】【F:README.md†L137-L139】 |
| `feature_stats_<division>_<cutoff>.json` | `gosales/outputs/features/` | Captures winsorization thresholds, coverage, and checksums. | Detects drift and validates deterministic feature generation.【F:README.md†L20-L26】 |
| `feature_list.json` | `gosales/outputs/` | Canonical list of expected features. | Used at scoring time to reindex matrices and prevent LightGBM shape mismatches.【F:README.md†L231-L235】 |

## Phase 3 — Modeling

| Artifact | Location Pattern | Purpose | Importance |
| --- | --- | --- | --- |
| `metrics.json` | `gosales/outputs/models/<division>/<cutoff>/` | Aggregated evaluation metrics (AUC, PR-AUC, lift@K, Brier). | Core evidence for model acceptance and release decisions.【F:README.md†L27-L35】 |
| `gains.csv` | `gosales/outputs/models/<division>/<cutoff>/` | Gains/lift table for decile analysis. | Validates ranking quality and supports sales prioritization exercises.【F:README.md†L27-L35】 |
| `calibration.csv` | `gosales/outputs/models/<division>/<cutoff>/` | Reliability curve data for Platt/Isotonic calibration. | Confirms well-calibrated probabilities before capacity planning.【F:README.md†L27-L35】 |
| `thresholds.csv` | `gosales/outputs/models/<division>/<cutoff>/` | Suggested probability cutoffs for target business K values. | Guides SDR capacity planning and scenario modeling.【F:README.md†L27-L35】 |
| `model_card.json` | `gosales/outputs/models/<division>/<cutoff>/` | Model metadata (train window, hyperparameters, guardrail status). | Provides compliance documentation and eases downstream integration.【F:README.md†L27-L35】 |
| `shap_summary_<division>_<cutoff>.csv` | `gosales/outputs/models/<division>/<cutoff>/` | Optional SHAP feature importance dump. | Powers explainability in UI and stakeholder reporting when available.【F:README.md†L27-L35】【F:README.md†L200-L202】 |
| `train_scores_<division>_<cutoff>.parquet` | `gosales/outputs/models/<division>/<cutoff>/` | Training-set score distribution snapshots. | Feeds drift analysis alongside holdout monitoring.【F:README.md†L37-L41】 |
| `train_feature_sample_<division>_<cutoff>.parquet` | `gosales/outputs/models/<division>/<cutoff>/` | Sample of features used to train the model. | Baseline for feature drift comparisons in forward validation.【F:README.md†L37-L41】 |

## Phase 4 — Whitespace Ranking / Next-Best-Action

| Artifact | Location Pattern | Purpose | Importance |
| --- | --- | --- | --- |
| `whitespace_<cutoff>.csv` | `gosales/outputs/whitespace/` | Ranked whitespace opportunities per account. | Directly fuels sales motions and UI whitespace lists.【F:README.md†L42-L48】 |
| `whitespace_explanations_<cutoff>.csv` | `gosales/outputs/whitespace/` | Feature-level drivers for each whitespace recommendation. | Supplies interpretable rationale in UI and sales enablement decks.【F:README.md†L42-L48】 |
| `thresholds_whitespace_<cutoff>.csv` | `gosales/outputs/whitespace/` | Cutoffs for whitespace selection strategies. | Aligns opportunity volume with sales capacity constraints.【F:README.md†L42-L48】 |
| `whitespace_metrics_<cutoff>.json` | `gosales/outputs/whitespace/` | Summary KPIs (coverage, expected lift). | Measures whitespace plan health and drift over time.【F:README.md†L42-L48】 |
| `whitespace_log_<cutoff>.jsonl` | `gosales/outputs/whitespace/logs/` | JSONL trace of gating, normalization, and diversification decisions. | Essential for debugging eligibility and reproducibility issues.【F:README.md†L42-L48】 |
| `mb_rules_<division>_<cutoff>.csv` | `gosales/outputs/whitespace/` | Market-basket lift rules per division. | Documents affinity drivers and exposes thresholds used in whitespace ranking.【F:README.md†L42-L48】 |
| `capacity_summary_<cutoff>.csv` | `gosales/outputs/whitespace/` | Aggregated capacity allocation after diversification. | Gives leadership a quick read on workload distribution across reps.【F:README.md†L226-L230】 |

## Phase 5 — Forward Validation / Holdout Monitoring

| Artifact | Location Pattern | Purpose | Importance |
| --- | --- | --- | --- |
| `validation_frame.parquet` | `gosales/outputs/validation/<division>/<cutoff>/` | Holdout evaluation frame with labels and predictions. | Ground-truth evidence of out-of-sample performance.【F:README.md†L49-L52】 |
| `gains.csv` | `gosales/outputs/validation/<division>/<cutoff>/` | Holdout lift analysis (mirrors training). | Confirms business value before enabling whitespace pushes for a division.【F:README.md†L49-L52】 |
| `calibration.csv` | `gosales/outputs/validation/<division>/<cutoff>/` | Holdout calibration curve. | Demonstrates probability reliability on future cohorts.【F:README.md†L49-L52】 |
| `topk_scenarios*.csv` | `gosales/outputs/validation/<division>/<cutoff>/` | Sensitivity tables for different capacity scenarios. | Helps sales ops choose the right cutoffs for live deployment.【F:README.md†L49-L52】 |
| `segment_performance.csv` | `gosales/outputs/validation/<division>/<cutoff>/` | Performance sliced by segment/industry. | Surfaces disparate impact and areas needing recalibration.【F:README.md†L49-L52】 |
| `metrics.json` | `gosales/outputs/validation/<division>/<cutoff>/` | Holdout KPIs mirroring training metrics. | Enables apples-to-apples comparison of model generalization.【F:README.md†L49-L52】 |
| `drift.json` | `gosales/outputs/validation/<division>/<cutoff>/` | PSI and drift alerts for features/labels. | Early warning system for data drift before go-live regressions.【F:README.md†L49-L52】 |

## Phase 6 — Configuration, UX, and Observability

| Artifact | Location Pattern | Purpose | Importance |
| --- | --- | --- | --- |
| `config_resolved.yaml` | `gosales/outputs/runs/<run_id>/` | Fully-resolved configuration manifest captured per run. | Documents effective settings and supports reproducible reruns.【F:README.md†L55-L62】 |
| `alerts.json` | `gosales/outputs/validation/<division>/<cutoff>/` | Consolidated warnings (e.g., PSI highlights). | Powers UI alert badges and enables proactive remediation.【F:README.md†L55-L62】 |
| `calibration_<division>.csv` | `gosales/outputs/` | Cross-cutoff calibration curves per division. | Streamlit UI badge source and longitudinal calibration tracking.【F:README.md†L115-L118】 |

## Scoring & Delivery

| Artifact | Location Pattern | Purpose | Importance |
| --- | --- | --- | --- |
| `icp_scores.csv` | `gosales/outputs/` | Customer-by-division probability scores. | Primary feed for CRM activation and Streamlit scoring views.【F:README.md†L80-L87】 |
| `whitespace.csv` | `gosales/outputs/` | End-to-end whitespace roll-up used by the UI. | Quick refresh for whitespace dashboards and exports.【F:README.md†L80-L87】 |
| `icp_scores_<timestamp>.csv` | `gosales/outputs/` | Timestamped fallback when Windows locks the primary file. | Prevents data loss on contention and aids troubleshooting in shared environments.【F:README.md†L226-L230】 |

## Leakage Gauntlet & Quality Gates

| Artifact | Location Pattern | Purpose | Importance |
| --- | --- | --- | --- |
| `fold_customer_overlap_<division>_<cutoff>.csv` | `gosales/outputs/leakage/` | GroupKFold leakage audit results. | Ensures data splits remain leakage-free before training sign-off.【F:README.md†L238-L245】 |
| `feature_date_audit_<division>_<cutoff>.csv` | `gosales/outputs/leakage/` | Feature timestamp consistency audit. | Detects inadvertent future-looking features that would inflate performance.【F:README.md†L238-L245】 |
| `static_scan_<division>_<cutoff>.json` | `gosales/outputs/leakage/` | Static code scan for banned datetime calls. | Stops coding patterns that reintroduce leakage when new features are added.【F:README.md†L238-L245】 |
| `leakage_report_<division>_<cutoff>.json` | `gosales/outputs/leakage/` | Consolidated PASS/FAIL report from the gauntlet. | Single source of truth for release gating; CI fails if issues persist.【F:README.md†L238-L245】 |

## Cross-Run Aggregations & Dashboards

| Artifact | Location Pattern | Purpose | Importance |
| --- | --- | --- | --- |
| `metrics_summary.csv` | `gosales/outputs/` | Roll-up of metrics across divisions and cutoffs. | Enables executive dashboards and trend monitoring without manual aggregation.【F:README.md†L246-L247】 |

## Published Reports

| Artifact | Location Pattern | Purpose | Importance |
| --- | --- | --- | --- |
| `gosales_cross_sell_top100_accounts.csv` | `reports/` | Ranked list of top 100 cross-sell accounts. | Sales-ready export for outreach prioritization.【F:reports/gosales_cross_sell_top100_summary.md†L1-L10】 |
| `gosales_cross_sell_top100_accounts_icp.csv` | `reports/` | Adds ICP probability context to the top accounts list. | Blends score strength with whitespace opportunities for GTM planning.【F:reports/gosales_cross_sell_top100_summary.md†L1-L15】 |
| `gosales_cross_sell_top100_accounts_whitespace.csv` | `reports/` | Focuses on whitespace gaps within the top accounts. | Targets product expansion plays by exposing missing lines of business.【F:reports/gosales_cross_sell_top100_summary.md†L1-L15】 |
| `gosales_cross_sell_top100_summary.md` | `reports/` | Narrative summary of the cross-sell exports. | Gives stakeholders quick interpretation without opening raw data files.【F:reports/gosales_cross_sell_top100_summary.md†L1-L21】 |

## How to Use This Catalog

1. Identify the pipeline phase you are validating or debugging.
2. Locate the corresponding artifact in the tables above.
3. Use the Purpose and Importance columns to determine what to inspect first and which downstream teams rely on the file.
4. When creating new outputs, update this catalog to keep cross-functional documentation accurate.
