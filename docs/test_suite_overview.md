# Test Suite Overview and Importance

This document summarizes the intent and criticality of every automated test in the repository. Tests are grouped by domain so engineers can quickly understand the guardrails that protect each stage of the GoSales pipeline.

## ETL and Data Integrity Safeguards
- **`test_cleaners.py`**
  - `test_clean_currency_value` verifies that currency strings, negatives in parentheses, numeric inputs, and nulls are all normalized into reliable floats so downstream aggregates are accurate. Importance: prevents ingestion of malformed monetary values that would skew revenue metrics.【F:gosales/tests/test_cleaners.py†L6-L18】
  - `test_coerce_datetime` checks that mixed-quality date strings are coerced to timestamps with invalid entries nulled, ensuring temporal filters do not crash on bad source data. Importance: protects date-based joins and windows from corrupt timestamps.【F:gosales/tests/test_cleaners.py†L13-L18】
- **`test_parse_and_keys.py`**
  - `test_parse_currency_cases`, `test_parse_date_and_bool_and_clean`, and `test_keys_deterministic` stress the parsing helpers and surrogate key builders so that canonical formats and deterministic identifiers are produced regardless of source noise. Importance: guarantees referential integrity across staging tables and prevents duplicate keys.【F:gosales/tests/test_parse_and_keys.py†L7-L30】
- **`test_contracts.py`**
  - `test_contract_required_columns_and_pk` and `test_contract_date_bounds` confirm that contract ingestion flags missing columns, duplicate primary keys, and invalid/out-of-range dates. Importance: stops contract snapshots with schema drift or future-dated entries from corrupting fact tables.【F:gosales/tests/test_contracts.py†L11-L36】
- **`test_fact_transactions_exists.py`**
  - `test_fact_transactions_table_exists` ensures the pipeline seeds the core `fact_transactions` table. Importance: early failure if the staging database lacks the anchor fact table.【F:gosales/tests/test_fact_transactions_exists.py†L5-L20】
- **`test_sku_map.py`**
  - `test_sku_map_basic_contract` and `test_sku_map_extended_divisions_and_aliases` validate that SKU mappings cover critical divisions and metadata fields, catching regressions in product alignment. Importance: prevents misclassification of sales into the wrong product families, which would cascade into incorrect features and labels.【F:gosales/tests/test_sku_map.py†L4-L29】

## Feature Engineering and Labeling Quality
- **`test_als_embedding_window.py`**
  - `test_customer_als_embeddings_respects_lookback` checks collaborative filtering embeddings honor look-back windows, ensuring stale transactions do not leak into current features. Importance: maintains temporal integrity of affinity signals.【F:gosales/tests/test_als_embedding_window.py†L7-L22】
- **`test_feature_matrix_memory.py`**
  - `test_feature_matrix_memory_smoke` executes feature generation on a large synthetic dataset while bounding memory usage, protecting CI from regressions that would explode resource consumption. Importance: keeps nightly builds and orchestrations stable.【F:gosales/tests/test_feature_matrix_memory.py†L9-L45】
- **`test_features.py`**
  - `_seed` helper seeds the DB for deterministic fixtures.【F:gosales/tests/test_features.py†L15-L23】
  - `test_feature_window_and_target` inspects cohort windows and label assignment to guarantee only the prediction horizon influences positives. Importance: avoids label leakage when training models.【F:gosales/tests/test_features.py†L25-L33】
  - `test_feature_cli_checksum` runs the CLI end-to-end with a patched config to ensure artifacts are emitted for downstream scoring. Importance: keeps the orchestration entry-point healthy.【F:gosales/tests/test_features.py†L36-L55】
  - `test_cli_config_override_persist` verifies config overrides persist to exported statistics (e.g., winsorization caps). Importance: protects reproducibility of feature snapshots when hyperparameters change.【F:gosales/tests/test_features.py†L58-L95】
- **`test_labels.py`**
  - `_seed_curated` sets up representative transaction histories.【F:gosales/tests/test_labels.py†L15-L28】
  - `test_denylist_threshold` ensures deny-listed SKUs and gross profit thresholds suppress label positives. Importance: prevents bad SKUs from teaching the model false positives.【F:gosales/tests/test_labels.py†L31-L53】
  - `test_build_labels_modes` covers expansion vs. all-customer modes so segmentation logic doesn't regress. Importance: keeps training cohorts aligned to business definitions.【F:gosales/tests/test_labels.py†L55-L68】
  - `test_censoring_flag` guarantees censoring metadata exists for survival-style analyses. Importance: signals partial observation windows to downstream analytics.【F:gosales/tests/test_labels.py†L71-L78】
- **`test_deciles_constant.py`**
  - `test_gains_and_capture_decile_counts` asserts validation deciles collapse/expand correctly when scores lack variance. Importance: keeps reporting robust against edge scoring distributions.【F:gosales/tests/test_deciles_constant.py†L7-L30】
- **`test_phase2_golden.py`**
  - `test_golden_small_features` compares feature outputs to curated expectations under constrained windows. Importance: protects against unintentional changes in aggregation math.【F:gosales/tests/test_phase2_golden.py†L9-L43】
- **`test_phase2_winsor_determinism.py`**
  - `_seed_two_customers` fixture prepares deterministic comparisons.【F:gosales/tests/test_phase2_winsor_determinism.py†L9-L20】
  - `test_determinism_in_memory` reruns feature generation twice to confirm byte-for-byte stability. Importance: avoids flaky diffs in ML pipelines.【F:gosales/tests/test_phase2_winsor_determinism.py†L22-L30】
  - `test_winsorization_effect` checks winsorization clamps extreme gross profit values as configured. Importance: ensures outliers do not dominate model training.【F:gosales/tests/test_phase2_winsor_determinism.py†L33-L56】

## Modeling Metrics, Leakage Guards, and Explainability
- **`test_phase3_determinism_and_leakage.py`**
  - `test_drop_leaky_features_by_name_and_auc` and `test_drop_leaky_features_robust_to_constant` validate leakage detection by name heuristics and AUC thresholds. Importance: stops future-target columns from inflating model quality metrics.【F:gosales/tests/test_phase3_determinism_and_leakage.py†L7-L40】
- **`test_phase3_determinism_pipeline.py`**
  - `_train_lr_calibrated` helper sets up a reproducible calibration pipeline.【F:gosales/tests/test_phase3_determinism_pipeline.py†L11-L22】
  - `test_determinism_same_seed_same_probs` ensures training with fixed seeds yields identical probabilities. Importance: guards against nondeterministic training regressions.【F:gosales/tests/test_phase3_determinism_pipeline.py†L25-L40】
  - `test_leakage_probe_no_gain_after_guard` validates that dropping leaky features restores baseline AUC. Importance: ensures leakage defenses remain effective.【F:gosales/tests/test_phase3_determinism_pipeline.py†L43-L72】
- **`test_phase3_metrics.py`**
  - Suite of tests (`test_threshold_math_correctness`, `test_calibration_bins_and_mae`, `test_calibration_bins_constant_scores`, `test_lift_at_k_monotonic`, `test_lift_at_k_zero_base_nan_default`, `test_lift_at_k_zero_base_custom_default`, `test_lift_at_k_sanitizes_nan_scores`, `test_lift_at_k_invalid_k_percent`, `test_weighted_lift_handles_nan_and_zero_base`, `test_topk_threshold_partition_performance`, `test_lift_at_k_ties_consistent`) exercise ranking threshold math, calibration stability, NA handling, and performance optimizations. Importance: keeps evaluation dashboards mathematically correct even under degenerate inputs.【F:gosales/tests/test_phase3_metrics.py†L13-L143】
- **`test_shap_sampling.py`**
  - `test_shap_sampling_controls` confirms SHAP exports respect sampling controls and skip conditions while logging guidance. Importance: avoids runaway explainability jobs on massive datasets.【F:gosales/tests/test_shap_sampling.py†L9-L71】
- **`test_phase4_bias_and_explanations.py`**
  - `test_explain_short_and_tokens` enforces that customer-facing explanations remain concise and informative. Importance: preserves clarity for sales reps consuming the model outputs.【F:gosales/tests/test_phase4_bias_and_explanations.py†L7-L17】

## Whitespace Ranking and Opportunity Generation
- **`test_build_lift.py`**
  - `test_product_indicators_and_rules_non_empty` verifies market-basket lift generation yields binary indicators and association rules. Importance: ensures affinity modeling inputs exist for downstream ranking.【F:gosales/tests/test_build_lift.py†L6-L38】
- **`test_whitespace_lift.py`**
  - `test_basket_plus_binary_and_lift_finite` checks that transactional baskets are binarized and resulting lift scores remain finite. Importance: protects rule-mining artifacts from schema regressions.【F:gosales/tests/test_whitespace_lift.py†L9-L67】
- **`test_whitespace_als.py`**
  - `test_build_als_generates_top_n_recommendations` validates ALS outputs fixed numbers of recommendations per customer and uses a portable code path. Importance: keeps recommendation quality and CI compatibility intact.【F:gosales/tests/test_whitespace_als.py†L10-L33】
- **`test_whitespace_als_smoke.py`**
  - `test_build_als_outputs_readable_ids` ensures ALS exports use human-readable identifiers. Importance: guarantees ops teams can reconcile results with CRM data.【F:gosales/tests/test_whitespace_als_smoke.py†L6-L15】
- **`test_whitespace_missing_divisions.py`**
  - `test_generate_whitespace_opportunities_skips_missing_divisions` asserts whitespace generation filters blank divisions. Importance: prevents junk entries from polluting opportunity queues.【F:gosales/tests/test_whitespace_missing_divisions.py†L8-L28】
- **`test_whitespace_score.py`**
  - `test_whitespace_score_is_continuous` confirms whitespace scores remain normalized and varied. Importance: avoids flat scoring that would render prioritization meaningless.【F:gosales/tests/test_whitespace_score.py†L6-L23】
- **Ranking weight and normalization checks (`test_phase4_*.py`)**
  - `test_phase4_bias_diversity_warning.py::test_bias_warning_logic_share_calc` ensures diversity warnings trigger when one division dominates selections. Importance: upholds fairness guardrails.【F:gosales/tests/test_phase4_bias_diversity_warning.py†L7-L16】
  - `test_phase4_capacity_selection_ties.py::test_capacity_selection_with_ties_returns_exact_k` validates tie-breaking keeps capacity targets exact. Importance: prevents over-allocating accounts to reps.【F:gosales/tests/test_phase4_capacity_selection_ties.py†L4-L14】
  - `test_phase4_capture_at_k.py::test_capture_at_k_math` sanity-checks capture metrics under ideal ranking. Importance: keeps whitespace performance reports trustworthy.【F:gosales/tests/test_phase4_capture_at_k.py†L7-L18】
  - `test_phase4_challenger_feature_list.py::test_rank_whitespace_handles_extra_feature` ensures challenger models gracefully consume extended feature sets. Importance: allows experimentation without breaking prod scoring.【F:gosales/tests/test_phase4_challenger_feature_list.py†L8-L34】
  - `test_phase4_cooldown_resort.py::test_cooldown_resorts_order` checks cooldown logic reorders recently surfaced accounts. Importance: prevents reps from seeing the same accounts repeatedly.【F:gosales/tests/test_phase4_cooldown_resort.py†L6-L14】
  - `test_phase4_determinism_ranking.py::test_score_determinism_sort_stable` enforces stable ranking order under repeated calculations. Importance: guarantees reproducible account lists for audits.【F:gosales/tests/test_phase4_determinism_ranking.py†L7-L26】
  - `test_phase4_eligibility_counts.py::test_eligibility_counts_sum_to_dropped_rows` reconciles eligibility drop counts with filtered results. Importance: provides trustworthy audit stats for eligibility rules.【F:gosales/tests/test_phase4_eligibility_counts.py†L8-L38】
  - `test_phase4_ev_cap_and_degradation.py::test_ev_cap_applies` verifies expected-value normalization caps outliers. Importance: avoids single whales dominating scoring weights.【F:gosales/tests/test_phase4_ev_cap_and_degradation.py†L7-L19】
  - `test_phase4_pool_vs_per_div_normalization.py::test_pooled_vs_per_division_normalization_behaviors` confirms percentile normalization remains balanced across different distributions. Importance: sustains comparability between divisions.【F:gosales/tests/test_phase4_pool_vs_per_div_normalization.py†L7-L15】
  - `test_phase4_rank_normalization.py` ensures percentile normalization works for varied and constant inputs so ranks stay meaningful.【F:gosales/tests/test_phase4_rank_normalization.py†L7-L24】
  - `test_phase4_weight_scaling_and_als.py` validates coverage-based weight scaling, ALS centroids, lift normalization, and fallback scoring when signals are sparse. Importance: keeps composite scoring resilient when certain signals drop out.【F:gosales/tests/test_phase4_weight_scaling_and_als.py†L12-L53】
  - `test_phase4_bias_and_explanations.py` described above keeps explanation text concise.【F:gosales/tests/test_phase4_bias_and_explanations.py†L7-L17】

## Scoring Pipeline and Model Loading
- **`test_discover_available_models.py`**
  - `test_discover_available_models_preserves_casing` checks that saved model directories round-trip to display names without losing casing. Importance: prevents operator confusion and misloads when models contain spaces.【F:gosales/tests/test_discover_available_models.py†L32-L41】
- **`test_score_customers.py`**
  - `test_score_customers_dedupes_names` confirms deduplication of customer records and fallback to joblib models when MLflow is unavailable. Importance: ensures scoring jobs complete even with metadata quirks.【F:gosales/tests/test_score_customers.py†L13-L67】
- **`test_score_customers_sanitize.py`**
  - `test_score_handles_strings_and_nans` verifies feature matrices are sanitized to numeric values before inference. Importance: prevents runtime crashes when training features include mixed types.【F:gosales/tests/test_score_customers_sanitize.py†L11-L64】
- **`test_scoring_joblib.py`**
  - `test_scoring_with_joblib` exercises the end-to-end scoring path using a joblib model and metadata. Importance: guarantees the legacy deployment path stays functional.【F:gosales/tests/test_scoring_joblib.py†L19-L71】
- **`test_scoring_metadata.py`**
  - `test_missing_metadata_fields_raises` enforces required metadata keys for loaded models. Importance: prevents silently misconfigured models from scoring customers.【F:gosales/tests/test_scoring_metadata.py†L14-L27】
- **`test_scoring_probabilities.py`**
  - `test_score_customers_with_predict_proba` and `test_score_customers_with_decision_function` cover both probability and decision-function model interfaces, ensuring probabilities are computed consistently. Importance: keeps scoring robust across estimator types.【F:gosales/tests/test_scoring_probabilities.py†L40-L82】
- **`test_score_p_icp_fallback.py`**
  - `test_score_p_icp_ignores_label_and_extra_columns` confirms probability scoring ignores non-feature columns and respects fallback logic. Importance: stops accidental inclusion of target columns in inference.【F:gosales/tests/test_score_p_icp_fallback.py†L8-L21】
- **`test_score_p_icp_sanitizes.py`**
  - `test_score_p_icp_handles_nan_inf_and_non_numeric` sanitizes NaNs, infinities, and strings before scoring. Importance: keeps batch scoring resilient to dirty features.【F:gosales/tests/test_score_p_icp_sanitizes.py†L7-L25】
- **`test_validate_holdout.py`**
  - `test_validate_holdout_restores_fact_table` runs the holdout validation command end-to-end, ensuring temporary overrides are cleaned up and original facts restored. Importance: protects production data when validating models against external holdout files.【F:gosales/tests/test_validate_holdout.py†L9-L93】
- **`test_whitespace_score.py`** and **`test_whitespace_missing_divisions.py`**
  - These tests collectively ensure whitespace opportunity generation produces continuous scores and filters invalid divisions. Importance: keeps opportunity exports actionable for sales operations.【F:gosales/tests/test_whitespace_score.py†L6-L23】【F:gosales/tests/test_whitespace_missing_divisions.py†L8-L28】

## Monitoring, Validation, UI, and Operations
- **`test_deciles_constant.py`** (noted earlier) already covers validation output stability.
- **`test_phase5_drift_calibration.py`**
  - `test_drift_psi_smoke` and `test_calibration_sanity` simulate drift and calibration CLI runs to ensure metrics and alert thresholds are emitted. Importance: keeps monitoring dashboards accurate and actionable.【F:gosales/tests/test_phase5_drift_calibration.py†L19-L92】
- **`test_phase5_dry_run.py`**
  - `test_dry_run_creates_single_run` verifies dry-run mode still logs a single run with registry entries. Importance: allows safe rehearsal of promotions without cluttering history.【F:gosales/tests/test_phase5_dry_run.py†L9-L31】
- **`test_phase5_ks_snapshot.py`**
  - `test_ks_train_vs_holdout_computed` confirms Kolmogorov–Smirnov statistics are written for train vs. holdout comparisons. Importance: flags scoring drift before production pushes.【F:gosales/tests/test_phase5_ks_snapshot.py†L23-L71】
- **`test_phase5_per_feature_psi_highlight.py`**
  - `test_per_feature_psi_highlight` ensures per-feature PSI alerts surface the correct features above threshold. Importance: directs analysts to the exact drivers of drift.【F:gosales/tests/test_phase5_per_feature_psi_highlight.py†L19-L65】
- **`test_phase5_scenarios_and_segments.py`**
  - `test_scenarios_math_and_segment_csv` and `test_per_rep_and_hybrid_scenarios` validate scenario calculators, confidence intervals, and segment outputs. Importance: keeps capacity planning outputs reliable for go-to-market leadership.【F:gosales/tests/test_phase5_scenarios_and_segments.py†L21-L157】
- **`test_ui_smoke.py`**
  - `test_compute_validation_badges_ok`, `test_compute_validation_badges_alerts`, and `test_streamlit_app_import_smoke` make sure UI utilities render badges correctly, load alerts, and the Streamlit app imports given realistic artifacts. Importance: protects executive dashboards from breaking after backend changes.【F:gosales/tests/test_ui_smoke.py†L9-L90】
- **`test_phase6_config_and_registry.py`**
  - `test_config_unknown_keys_rejected`, `test_run_registry_and_manifest`, `test_whitespace_weights_normalized`, and `test_whitespace_weights_malformed_raise` enforce config validation, manifest writing, and weight normalization. Importance: ensures operational runs are auditable and configurations stay sane.【F:gosales/tests/test_phase6_config_and_registry.py†L10-L50】
- **`test_phase4_bias_and_explanations.py`** (mentioned earlier) also supports explainability compliance.
- **`test_phase5_...`** entries above cover ongoing monitoring.
- **`test_discover_available_models.py`** (noted earlier) protects deployment UX.
- **`test_build_lift.py`** and whitespace tests (noted above) ensure exploratory mining outputs function.

Collectively, these tests form a safety net over data ingestion, feature generation, model training, scoring, whitespace prioritization, validation, UI, and operational workflows. Keeping them green is essential for trustworthy recommendations and measurable business outcomes.
