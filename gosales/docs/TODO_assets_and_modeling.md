# GoSales: High-Impact TODOs (Assets + Modeling)

This list captures prioritized actions to improve model accuracy, stability, and
operational robustness. Items marked [HI] are high-impact. Focus first on code
changes that do not require ETL rebuilds or retraining.

## High-Impact (No ETL/Training Needed)

- [x] [HI] Scoring reindex: enforce exact training feature order/shape at
      inference to avoid LightGBM shape errors (use `DataFrame.reindex`).
- [x] [HI] Metrics roll-up: aggregate `metrics_*.json` into a single summary CSV
      (AUC, PR-AUC, lift@K, Brier) for quick comparison/monitoring.
- [x] [HI] Add score-time logging guards: emit counts of missing/added columns,
      top 20 missing features, and ensure numeric dtype before prediction.
- [x] [HI] Bias/capacity: make ranker thresholds explicit in outputs (already
      emitted) and add quick sanity alert when a single division > threshold.
- [x] [HI] File-lock-resistant output writes: write ICP scores to primary path,
      fallback to timestamped file on Windows lock; log warning.
- [x] [HI] Identifier allow-list for DB objects: sanitize/validate schema and
      view/table names sourced from config (regex + explicit allow-list) before
      building f-string SQL; keep WHERE parameters bound via SQLAlchemy.
- [x] [HI] Connection health check: add `validate_connection(engine)` and use in
      pipeline entrypoints (score/build/train). Add config `database.strict_db`
      to fail instead of silently falling back to SQLite when required.

## Leakage Gauntlet (from GPT-5 suggestions)

- [x] [HI] Entrypoint CLI: `gosales/pipeline/run_leakage_gauntlet.py` to run per
      division+cutoff; writes artifacts under `outputs/leakage/`.
- [x] [HI] Group-safe CV: add option to use time-aware split + GroupKFold by
      `customer_id` in training; emit `fold_customer_overlap_<div>_<cutoff>.csv`
      and fail if any overlap.
- [x] [HI] Feature date audit: record for each derived feature the latest
      source event date contributing to it; write
      `feature_date_audit_<div>_<cutoff>.csv`; fail if any date > cutoff.
- [x] [HI] Static scan: forbid calls like `datetime.now()`, `pd.Timestamp.now()`,
      `date.today()` in feature paths; write `static_scan_<div>_<cutoff>.json`.
- [x] [HI] 14-day shift test: re-build features/events with all dates shifted
      back 14 days and (optionally) retrain; metrics must not improve beyond a small epsilon.
      CLI: `python -m gosales.pipeline.run_leakage_gauntlet --division <Div> --cutoff <YYYY-MM-DD> --no-static-only --run-shift14-training`
      Artifacts: `shift14_metrics_<div>_<cutoff>.json`.
- [x] [HI] Top-K ablation: drop K most important features and ensure metrics do
      not improve beyond noise.
      CLI: `python -m gosales.pipeline.run_leakage_gauntlet --division <Div> --cutoff <YYYY-MM-DD> --no-static-only --run-topk-ablation --topk-list 10,20`
      Artifacts: `ablation_topk_<div>_<cutoff>.{csv,json}`.
- [x] [HI] Consolidated report: `leakage_report_<div>_<cutoff>.json` aggregating
      status of all checks with PASS/FAIL and actionable messages; non-zero exit
      code on failure when run via CLI.

## High-Impact (Code Ready Now; Run Later)

- [x] Renewal pressure at cutoff: expand asset features to
      `expiring_{30,60,90}d_<rollup>` + shares. (Requires ETL rebuild to use.)
- [x] Subscription signals: merge OnSubs/OffSubs by rollup when available and
      derive `subs_share` features. (Requires new mapping + ETL.)
- [x] Class-imbalance tuning: enable `class_weight='balanced'` (LR) and
      `scale_pos_weight` (LGBM) via config. (Takes effect next training.)
- [x] Ablation study: train 1–2 divisions with assets OFF to quantify lift deltas.
      Script: `scripts/ablation_assets_off.py`
      Usage: `python scripts/ablation_assets_off.py --division Solidworks --cutoff 2024-12-31`
      Artifact: `gosales/outputs/ablation_assets_off_<division>_<cutoff>.json`
- [x] Transaction boundaries in ETL: wrap destructive DDL/DML in
      `with engine.begin():` to ensure rollback on failure in star-build and
      future asset materializations.
- [x] Expand chunking: ensure all large reads use chunked `read_sql_query`, and
      large CSV/Parquet writes use streaming/sink patterns where available.

## Nice-to-Have

- [x] Tenure QA export: histogram and per-rollup bad-date reliance over time.
 - [x] Name-join QA: coverage of Moneyball names + `dim_customer.customer_id`.
       Script `scripts/name_join_qa.py` writes:
       - `name_join_qa_summary.json` with row/unique coverage and ambiguous norms
       - `unmapped_names_top_<N>.csv` (default N=50)
       - `coverage_by_department.csv`
       Usage: `python scripts/name_join_qa.py --top 50`
- [x] Drift snapshots: monthly scoring prevalence and calibration MAE trend.
- [x] Explanations: optional SHAP export for selected divisions (sample gated).
      Use `gosales/models/train.py --shap-sample 5000` (requires `shap` installed).
- [x] Move ad-hoc SQL into `sql/` with parameterized templates (helpers added; coverage expanded incrementally).
- [x] Cache small dims (e.g., `dim_customer`) in memory during scoring to
      reduce repeated DB calls. Implemented in `pipeline/score_customers.py` with an in-process cache.
- [x] CI checks: quick contract tests for assets (rollup mapping coverage
      threshold) and tenure imputation sanity (script `scripts/ci_assets_sanity.py`).
      Scorer feature-list alignment added (script `scripts/ci_featurelist_alignment.py`).
      Helper to build features for each model's cutoff: `scripts/build_features_for_models.py` (run before alignment CI).

## Complete / Done

- [x] Build `fact_assets` (Moneyball A- items rollup) and map customers by
      human-readable names.
- [x] Tenure imputation for pre-1996 purchase dates and `assets_bad_purchase_share`.
- [x] Add asset features at cutoff and join into feature matrices.
- [x] Train core divisions and generate ranked whitespace at 2024-12-31.



