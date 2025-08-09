### Phase 5 To-Do (Forward Validation / Holdout vs playbook)

- Frame & config
  - Add `validation` section to `gosales/config.yaml`: bootstrap_n, top_k_percents (reuse), capacity_grid, ev_cap_percentile, segment_columns (industry/size/region), ks_threshold, psi_threshold. DONE
  - Create `gosales/validation/` package with helpers and CLI. PARTIAL (package + forward CLI skeleton)

- Evaluation frame
  - Load frozen model + calibrator + feature list for a given cutoff. TODO
  - Build features ≤ cutoff for all customers; score `p_hat`. DONE (from features parquet)
  - Join holdout labels (Phase 1 logic) and EV proxy; apply Phase 4 eligibility. PARTIAL (EV proxy fallback; labels fallback)
  - Persist `validation/{division}/{cutoff}/validation_frame.parquet` (deterministic sort). TODO

- Metrics & artifacts
  - Ranking/business: AUC, PR-AUC, gains by decile, capture@{5,10,20}%, precision@K, revenue-weighted capture, expected GP @ capacity. PARTIAL (AUC/PR-AUC/gains/capture grid/expected GP norm)
  - Calibration: Brier, cal-MAE, reliability bins (10–20). PARTIAL (bins + Brier + cal-MAE)
  - Stability by segment (cohort/industry/size/region). PARTIAL (segment_performance.csv for first available segment)
  - Write: `metrics.json`, `gains.csv`, `calibration.csv`, `topk_scenarios.csv`. PARTIAL (also writes sorted scenarios)

- Confidence intervals
  - Block bootstrap by customer (seeded) producing 95% CIs for capture@K, revenue capture, Brier, cal-MAE, precision@K. TODO

- Drift diagnostics
  - Feature drift PSI between train (latest training frame) and holdout; flag PSI > cfg.threshold. PARTIAL (PSI utility; proxy wired; drift.json writes EV vs holdout GP PSI)
  - Score drift KS on `p_hat` (train vs holdout); flag KS > cfg.threshold. PARTIAL (KS(p_hat) pos vs neg as proxy)
  - Optional: SHAP drift if LGBM available and SHAP installed. TODO
  - Write `drift.json`. DONE (basic PSI/KS proxies)

- Scenarios (capacity & thresholds)
  - Grid over capacity modes (top-N% and per-rep) and percents; compute contacts, precision, recall (capture), expected_GP, realized_GP (historical), 95% CI. PARTIAL (top-% + per-rep + hybrid-segment; capture/precision/rev_capture/realized_GP with CIs)
  - Rank by expected GP if calibration is strong; otherwise by capture@K. TODO

- CLI
  - `gosales/validation/forward.py` with flags: `--division`, `--cutoff`, `--window-months`, `--capacity-grid`, `--bootstrap`, `--config`. PARTIAL (skeleton implemented: gains, calibration, scenarios, minimal metrics)

- Guardrails
  - Censoring: if holdout window incomplete, flag and exclude; log counts. TODO
  - Base-rate collapse warn if prevalence < 0.2% or > 50%. TODO
  - EV outliers cap at p95; log count capped. TODO
  - Segment failure: if any segment top-decile capture < baseline by > 5 pts, flag. TODO

- Tests
  - Window integrity: ensure no post-cutoff features leak. TODO
  - Censoring behavior on synthetic incomplete holdout. TODO
  - Bootstrap determinism: same seed → same CI bands. TODO
  - Drift smoke: injected shift triggers PSI flag. TODO
  - Calibration sanity: synthetic sigmoid recovers low Brier. TODO
  - Scenario math: counts and expected GP correct. TODO


