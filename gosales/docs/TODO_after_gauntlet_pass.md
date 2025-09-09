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
- [ ] Calibration stability (per-division choice Platt vs Isotonic; persist curves).

## Whitespace Improvements

- [ ] [HI] Enable challenger meta-learner for whitespace weights; compare vs current on last 3 cutoffs.
- [ ] Segment-wise weights (industry/size); fallback to global when sparse.
- [ ] ALS coverage enforcement + item2vec backfill to reduce cold-start.
- [ ] Capacity-aware threshold optimization (hit capacity targets with stable capture@K across eras).

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


