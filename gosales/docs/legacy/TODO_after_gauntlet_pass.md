# GoSales: Next Steps After Gauntlet PASS (Unified TODO)

This list merges GPT-5-Proâ€™s recommendations with our proposed upgrades. Format mirrors other TODO docs; mark [HI] for high-impact. Tackle top-down; prefer changes not requiring ETL rebuilds.

## High-Impact (Immediate; 48â€“72h)

- [x] [HI] Repair label permutation test (time-bucket, train-only shuffle, re-seed per perm, add p-value; write JSON + plot).
  - Printers: baseline_auc 0.7095, perm_mean 0.5272, degradation 0.1823, pâ‰ˆ0.0476
    gosales/outputs/leakage/Printers/2024-12-31/permutation_diag.json
  - Solidworks: baseline_auc 0.5774, perm_mean 0.4948, degradation 0.0826, pâ‰ˆ0.0476
    gosales/outputs/leakage/Solidworks/2024-12-31/permutation_diag.json
- [x] [HI] Make Gauntlet PASS reproducible (pin seeds; ensure 0 customer overlap; two runs within tiny deltas; emit overlap CSVs).
  - Printers: PASS (delta_auc=0.0000, delta_lift10=0.0000, overlap=0) â€” repro_check_Printers_2024-12-31.json
  - Solidworks: PASS (delta_auc=0.0000, delta_lift10=0.0000, overlap=0) â€” repro_check_Solidworks_2024-12-31.json
- [x] [HI] Add Shift-grid sanity to Gauntlet (evaluate {7,14,28,56}; ensure non-improving metrics as we shift earlier).
  - Implemented CLI + artifacts; initial run for {7,14} PASS
    gosales/outputs/leakage/Printers/2024-12-31/shift_grid_Printers_2024-12-31.json
  - Pending: execute {28,56} and add to summary
- [x] [HI] Attach diagnostics to Gauntlet report and UI (include permutation/stability artifacts; surface PASS/FAIL gates + links).
  - Gauntlet report now includes diagnostics_summary_* when artifacts exist.
  - UI panel renders diagnostics summary JSON and plots; adds Shift-Grid summary table.

## Prove Horizon-Robust Accuracy (1â€“2 weeks)

- [x] Prequential forward-month evaluation (freeze at 2024-06-30; score monthly through 2025; plots for AUC, lift@K, Brier vs horizon).
  - Implemented: `python -m gosales.pipeline.prequential_eval --division <Div> --train-cutoff 2024-06-30 --start 2025-01 --end 2025-12 --window-months 6`
  - Artifacts under `gosales/outputs/prequential/<division>/<train_cutoff>/`: JSON, CSV, and curves PNG.
- [x] Adjacency ablation triad (Full vs No-recency/short-windows vs SAFE under GroupCV+purge; require Fullâ‰¥SAFE on far-month holdouts).
  - Printers (train 2024-06-30 â†’ holdout 2025-03-31, 6m): Full AUC 0.8301, SAFE AUC 0.8251, Î”AUC +0.0050. Artifacts:
    gosales/outputs/ablation/adjacency/Printers/2024-06-30_2025-03-31/adjacency_ablation_Printers_2024-06-30_2025-03-31.{json,csv}
  - Solidworks (train 2024-06-30 â†’ holdout 2025-03-31, 6m): Full AUC 0.7380, SAFE AUC 0.7449, Î”AUC âˆ’0.0069 (SAFE > Full; flag for review). Artifacts:
    gosales/outputs/ablation/adjacency/Solidworks/2024-06-30_2025-03-31/adjacency_ablation_Solidworks_2024-06-30_2025-03-31.{json,csv}
  - UI: Added a results viewer under QA â†’ Ablation to browse triad runs and metrics.
 - [x] Calibration & business yield reporting (Platt/Isotonic metrics, top-K yield, coverage curves; include in model cards).
   - UI (Metrics tab): shows AUC/PR-AUC/Brier/Cal-MAE summary, renders calibration plot, top-K thresholds, and Business Yield (Top-K) table + coverage curve (Capture vs K with Pos Rate).
   - Model cards now include: calibration.method (platt/isotonic), calibration.mae_weighted, and topK summary (pos_rate, capture, threshold per K).
   - Files: model_card_<div>.json, calibration_<div>.csv, thresholds_<div>.csv, gains_<div>.csv.
 - [x] Adopt SAFE feature policy for Solidworks (per-division) and retrain at 2024-06-30. Rationale: SAFE > Full on far-month holdout; improves robustness while minimizing adjacency risk. Config: modeling.safe_divisions: ["Solidworks"].
   - Prequential (Solidworks @2024-06-30 SAFE): generated 2024-07..2025-02 curves (AUC, Lift@10, Brier). Artifacts under:
     gosales/outputs/prequential/Solidworks/2024-06-30/
   - SAFE-lite experiment (Solidworks): added 'safe_lite' ablation variant; SAFE-lite underperformed SAFE and Full on far-month holdout. Artifact:
     gosales/outputs/ablation/adjacency/Solidworks/2024-06-30_2025-03-31/adjacency_ablation_Solidworks_2024-06-30_2025-03-31.json
   - CI helper: auto-SAFE from ablation. Adds division to modeling.safe_divisions when SAFE â‰¥ Full by Î”AUC â‰¥ 0.005.
     python -m gosales.pipeline.auto_safe_from_ablation --threshold 0.005
   - Model cards now include top-K yield summary (pos_rate, capture, threshold per K). Re-train to populate updated cards.

## Accuracy Roadmap (Data & Modeling)

- [x] Cycle-aware features (hazard/log-recency, tenure buckets; reorder estimators).
  - Added log-recency and hazard/decay transforms with half-lives (30/90/180d) for all/div recency.
  - Added tenure months and tenure bucket dummies (<3m, 3â€“6m, 6â€“12m, 1â€“2y, â‰¥2y).
  - Reordered estimators to [lgbm, logreg] in config (selection still automatic by metrics).
  - Config: features.recency_decay_half_lives_days controls decay set.
- [x] Offset windows and deltas (e.g., 12m block ending cutoff-60d; 12m-vs-24m deltas) to decorrelate from boundary.
  - Added offset RFM aggregates for each configured window w ending at cutoffâˆ’offset_days (default 60d):
    rfm__all|div__{tx_n,gp_sum,gp_mean}__{w}m_off60d
  - Added window deltas comparing last 12m vs previous 12m from 24m totals (all and div):
    delta and ratio for gp_sum and tx_n (e.g., rfm__all__gp_sum__delta_12m_prev12m, ...ratio_12m_prev12m).
  - Config toggles: features.enable_offset_windows, features.offset_days, features.enable_window_deltas.
- [x] Affinity with lag (co-purchase lift features with ≥60d embargo; keep ALS in prod, SAFE-off in audits).
  - Implemented market-basket affinity features with lagged exposure using transactions up to `cutoff - affinity_lag_days` (default 60d):
    mb_lift_max_lag{N}d, mb_lift_mean_lag{N}d, affinity__div__lift_topk__12m_lag{N}d.
  - ALS embeddings remain enabled in production; SAFE policy drops `als_f*` during audits.
  - Config: features.affinity_lag_days controls the embargo window.
- [x] Sparse divisions (hierarchical/pooled encoders for niche industries).
  - Added pooled/hierarchical encoders for industry and industry_sub: pre-cutoff transaction rates and GP shares smoothed with priors (global and parent-industry). Configurable via features.pooled_encoders_*.
- [ ] Model class exploration (GBDT with monotonic constraints on known monotone features; keep LR baseline).
- [x] Calibration stability (per-division choice Platt vs Isotonic; persist curves).
  - Final calibration method now chosen per-division by volume: if positives >= modeling.sparse_isotonic_threshold_pos → Isotonic; else Platt (sigmoid). Falls back to available methods.
  - Training still evaluates both on validation (Brier); model card records method + cal_MAE.
  - Artifacts: calibration_<div>.csv (bins) and calibration_plot_<div>.png (curve) written to outputs.

## Whitespace Improvements

- [x] [HI] Enable challenger meta-learner for whitespace weights; compare vs current on last 3 cutoffs.
  - Implemented shadow challenger (logistic) over [p_icp_pct, lift_norm, als_norm, EV_norm] with no effect on champion ranking unless enabled. Outputs `score_challenger` alongside `score` in `whitespace.csv`.
  - Toggle via `whitespace.challenger_enabled: true` and `whitespace.challenger_model: lr` in `config.yaml`. Overlap vs champion can be assessed by diffing top‑N.
- [x] Segment-wise weights (industry/size); fallback to global when sparse.
  - Added optional `whitespace.segment_columns` (e.g., `['industry','size_bin']`) and `whitespace.segment_min_rows` to apply coverage‑aware blending per segment; falls back to global weights for small groups.
  - `size_bin` is derived from `total_gp_all_time` (or 12m GP/TX) into small/mid/large quantiles if requested.
- [x] ALS coverage enforcement + item2vec backfill to reduce cold-start.
  - If `features.use_item2vec: true` and ALS coverage < `whitespace.als_coverage_threshold`, ranker computes item2vec similarity and backfills rows with missing ALS.
  - Weight scaling also shrinks ALS contribution proportional to coverage (config threshold honored).
  - Division-specific ALS owner centroids are now persisted and used during ranking to prevent cross-division leakage. Files: `als_owner_centroid_<division>.npy` (and for assets-ALS: `assets_als_owner_centroid_<division>.npy`). Regression tests verify distinct centroids and behavior when a division has no owners.
- [x] Capacity-aware threshold optimization (hit capacity targets with stable capture@K across eras).
  - Selection rebalancer enforces `whitespace.bias_division_max_share_topN` at the top‑percent capacity by capping each division’s share and topping up from others by score.
  - Emits capacity summary and bias/diversity warnings as before; now the selected set respects the share cap when configured.

## Reporting & Ops

- [ ] Model cards per division (Gauntlet status, horizon curves, cal metrics, top-K tables, artifacts links).
- [ ] Gains/capture@K by segment exports; combined summary.
- [ ] Stability vs prior run (Jaccard overlap; account in/out diffs).
- [ ] Bias/diversity reporting and enforcement (division share caps; rep/region balance).
- [ ] CI gates (fail PR on Gauntlet FAIL, permutation p>0.01, customer overlap non-empty).

## Acceptance & Rollback

- [ ] Shift-14 PASS (Î”AUC â‰¤ 0.01; Î”Lift@10 â‰¤ 0.25); Shift-grid non-improving.
- [ ] Permutation: p â‰¤ 0.01 and meaningful degradation.
- [ ] Forward-months: gentle decline; no paradoxical early gains.
- [ ] Rollback toggles documented (disable SAFE/purge/permutation/shift-grid) for forensics only.




---

## Addendum — Post–Gauntlet PASS Hardening & Roadmap (2025‑09‑09)

The items below **do not remove or replace** anything above; they extend it with
the most valuable next steps to make GoSales world‑class and production‑ready. Each
item is small, testable, and references target files/CLIs.

### A) Gauntlet & CV Hardening (now)

- [ ] **Enforce GroupCV + Purge in all Gauntlet paths**
  - Patch `gosales/pipeline/run_leakage_gauntlet.py` to pass `--group-cv` and `--purge-days 45` into every internal training call.
  - Patch `gosales/models/train.py` to default to `group_cv=True` when `--safe-mode` is set and respect `--purge-days` for time‑adjacent splits.
  - **Artifact:** write `cv_manifest_{division}_{cutoff}.json` with: seed, folds, purge_days, per‑fold customer counts.
  - **Test:** add `tests/test_phase3_determinism_and_leakage.py::test_gauntlet_uses_groupcv_and_purge` to assert flags are plumbed through.

- [ ] **Shift‑Grid completion and guard**
  - Run {28, 56} days and append to `shift_grid_*.json`.
  - Add guard: earlier cutoffs **must not** improve by more than `epsilons` (AUC≤0.01; Lift@10≤0.25). Fail Gauntlet on breach; include offending rows in `leakage_report_*`.

- [ ] **Reproducibility receipts**
  - Persist `fold_assignments_{division}_{cutoff}.csv` (customer_id → fold) and `random_state` for each estimator.
  - Add CI check that two Gauntlet runs with same seed hash to identical `shift_grid_*.json`.

---

## Completed This Session (2025-09-09)

- [x] Add Post_Processing as a model-backed target and wire into scoring/whitespace.
- [x] Align per-division score frames before concatenation to fix schema width mismatch in scoring.
- [x] Prefer curated DB connection in scoring (fallback to primary).
- [x] Ensure trainer writes metadata.json even on degenerate runs.
- [x] Sanitize features in prequential evaluator to avoid dtype errors (LightGBM).

Artifacts:
- `gosales/models/post_processing_model/{model.pkl,feature_list.json,metadata.json}`
- `gosales/outputs/metrics_post_processing.json`, `gosales/outputs/thresholds_post_processing.csv`, `gosales/outputs/gains_post_processing.csv`
- `gosales/outputs/prequential/Post_Processing/2024-12-31/{prequential_*.json,csv,png}`

## Next Steps (Recommended)

- [ ] Run Leakage Gauntlet for Post_Processing (Shift‑14 + Permutation) and attach PASS/FAIL to QA.
- [ ] UI: Badge divisions as model‑backed vs heuristic in Metrics/QA views.
- [ ] Review whitespace top‑N capacity and division share; adjust `whitespace.weights` or `bias_division_max_share_topN` if Post_Processing over‑dominates.
- [ ] Expand Post_Processing training with future cutoffs and evaluate horizon curves quarterly.

### B) SAFE‑Mode Expansion (time‑adjacent risk reduction)

- [ ] **Feature family minimum window**
  - New config: `features.safe_windows_min_months: 6` (applies when `safe_mode=True`). Drop or swap `{1–3m}` windows for `{6–12m}` in: `rfm__*`, `sku_nunique__*`, `division_share__*`.
  - Files: `gosales/features/engine.py` (feature registry), `gosales/utils/config.py` (defaults), tests in `tests/test_features.py`.

- [ ] **Recency & expiry lags**
  - When `safe_mode=True`, drop: `days_since_last_*`, `assets_*_expiring_*`, and any `*_recency_days_*` that look within 30 days of cutoff.
  - Add lagged proxies instead: 30‑day offset versions of short‑window rates (compute on `[cutoff-90d, cutoff-30d]`).

- [ ] **Adjacency audit**
  - New CLI: `python -m gosales.pipeline.leakage_diagnostics --division <Div> --cutoff <Date> --mode time_adjacency`
  - Emits `adjacency_report_{division}_{cutoff}.json` with: top features within 60 days of cutoff, correlation to `order_date`, and SAFE‑drop recommendations.

### C) Forward Validation (Phase‑5) upgrades

- [ ] **Block bootstrap CIs (1,000 reps, by customer)**
  - Implement in `gosales/validation/forward.py` → `bootstrap_ci` with fixed seed; write CI columns for capture@K, precision@K, rev_capture, realized_GP.
  - Tests: `tests/test_phase5_scenarios_and_segments.py` expands to check CI determinism across two identical runs.

- [ ] **Scenario ranking policy**
  - If `cal_mae ≤ threshold`: rank scenarios by **expected GP**; else by **capture@K**. Add to `topk_scenarios_sorted.csv` and `metrics.json`.

- [ ] **Censoring & base‑rate guards**
  - Flag incomplete holdouts (`censored_flag`); warn on prevalence outside [0.2%, 50%]. Fail “harsh” CI gate when both present.

### D) Business‑grade Whitespace

- [ ] **Bias & diversification**
  - Post‑ranking diversification when one division exceeds `whitespace.division_share_cap` (config). Emit `diversification_report_{cutoff}.json` and log which candidates were swapped.
  - Segment fairness table: capture@K by `industry/size/region`. File: `whitespace_metrics_{cutoff}.json`.

- [ ] **ALS/affinity coverage‑aware weights**
  - Already implemented dynamic weight scaling; add explicit coverage thresholds to the metrics JSON and the UI badges.

- [ ] **Rep capacity modes & hybrid**
  - Ensure `rank_whitespace.py` supports `per_rep` and `hybrid_segment` (done in tests). Expose config examples in `gosales/docs/OPERATIONS.md`.

### E) Data & Mapping correctness (high leverage)

- [ ] **Adopt CPE & Post_Processing mapping**
  - Implement SKU/division updates per `docs/Repo review feedback.txt` (CPE, Post_Processing, SW_Plastics, DraftSight rules). Update `etl/sku_map.py` and `etl/build_star.py`.
  - Tests: extend `tests/test_sku_map.py` and add ETL smoke tests to ensure division counts/GP match expectations.

- [ ] **Contracts & schema gates**
  - Add Pandera‑style checks for `fact_transactions` and `dim_customer` to catch type drifts and date bounds before features.

### F) Model Cards & Governance

- [ ] **Complete model cards**
  - Include: Gauntlet status (shift‑14/grid), permutation p‑value, SAFE status, CV settings (folds, purge_days), calibration metrics, top‑K tables, links to artifacts.
  - File: `gosales/models/train.py` (writer), rendered in UI.

- [ ] **Run registry / manifest**
  - Ensure every pipeline invocation writes `outputs/runs/<run_id>/{manifest.json, config_resolved.yaml, logs.jsonl}`
  - Add `whitespace_metrics_{cutoff}.json` checksum & run_id for determinism tracking.

### G) Reporting & UI polish

- [ ] **Leakage dashboard**
  - New UI section to render: Gauntlet results (shift‑grid table & plots), permutation histogram, feature stability chart. Link back to artifacts.

- [ ] **Validation badges**
  - Surface Cal‑MAE, PSI(EV vs GP), KS(train vs holdout) with thresholds in config. Already partially implemented; wire to `validation/forward` outputs.

### H) Hygiene & Docs

- [ ] **Encoding cleanup**
  - Normalize smart quotes/dashes in this file and docs (UTF‑8). Add a pre‑commit hook to fix mojibake automatically.

- [ ] **End‑to‑end smoke on sample DB**
  - Add `scripts/smoke_run.ps1` to train → score → rank → validate on a small slice and verify artifacts presence + schema with one command.

---

### Acceptance Add‑Ons

- [ ] **Shift‑Grid:** For each division, the metric curves **monotonically degrade** as the shift increases; any violation < eps becomes a **warning**, ≥ eps a **failure**.
- [ ] **Forward Validation:** All metrics reported with 95% CIs; CI widths shrink with increased positives vs prior run.
- [ ] **Governance:** Model card present for each division with Gauntlet & CV details; UI badges OK.

### Rollback Safety

- [ ] One‑flag revert for SAFE mode (`features.safe_mode=false`) and for purge CV (`--purge-days 0`), documented in `docs/OPERATIONS.md`.
- [ ] Snapshot copy of the last known-good artifacts (run registry) with quick restore instructions.

### Follow-up Issues

- [ ] Centralize missingness mask + fill strategy for consistent `_missing` flags across all engineered columns.
  - Tracking doc: `gosales/docs/issues/0001-centralize-missingness-flags.md`
  - Summary: Compute features first, capture a comprehensive NaN mask, generate `_missing` flags, then apply a single centralized fill pass (configurable). Provide a feature flag `features.centralized_fill_enable` for safe rollout.
