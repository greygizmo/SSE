# GoSales: Next Steps After Gauntlet PASS (Unified TODO)

This list merges GPT-5-Pro’s recommendations with our proposed upgrades. Format mirrors other TODO docs; mark [HI] for high-impact. Tackle top-down; prefer changes not requiring ETL rebuilds.

## High-Impact (Immediate; 48–72h)

- [x] [HI] Repair label permutation test (time-bucket, train-only shuffle, re-seed per perm, add p-value; write JSON + plot).
  - Printers: baseline_auc 0.7095, perm_mean 0.5272, degradation 0.1823, p≈0.0476
    gosales/outputs/leakage/Printers/2024-12-31/permutation_diag.json
  - Solidworks: baseline_auc 0.5774, perm_mean 0.4948, degradation 0.0826, p≈0.0476
    gosales/outputs/leakage/Solidworks/2024-12-31/permutation_diag.json
- [x] [HI] Make Gauntlet PASS reproducible (pin seeds; ensure 0 customer overlap; two runs within tiny deltas; emit overlap CSVs).
  - Printers: PASS (delta_auc=0.0000, delta_lift10=0.0000, overlap=0) — repro_check_Printers_2024-12-31.json
  - Solidworks: PASS (delta_auc=0.0000, delta_lift10=0.0000, overlap=0) — repro_check_Solidworks_2024-12-31.json
- [x] [HI] Add Shift-grid sanity to Gauntlet (evaluate {7,14,28,56}; ensure non-improving metrics as we shift earlier).
  - Implemented CLI + artifacts; initial run for {7,14} PASS
    gosales/outputs/leakage/Printers/2024-12-31/shift_grid_Printers_2024-12-31.json
  - Pending: execute {28,56} and add to summary
- [ ] [HI] Attach diagnostics to Gauntlet report and UI (include permutation/stability artifacts; surface PASS/FAIL gates + links).

## Prove Horizon-Robust Accuracy (1–2 weeks)

- [ ] Prequential forward-month evaluation (freeze at 2024-06-30; score monthly through 2025; plots for AUC, lift@K, Brier vs horizon).
- [ ] Adjacency ablation triad (Full vs No-recency/short-windows vs SAFE under GroupCV+purge; require Full≈SAFE on far-month holdouts).
- [ ] Calibration & business yield reporting (Platt/Isotonic metrics, top-K yield, coverage curves; include in model cards).

## Accuracy Roadmap (Data & Modeling)

- [ ] Cycle-aware features (hazard/log-recency, tenure buckets; reorder estimators).
- [ ] Offset windows and deltas (e.g., 12m block ending cutoff-60d; 12m-vs-24m deltas) to decorrelate from boundary.
- [ ] Affinity with lag (co-purchase lift features with ≥60d embargo; keep ALS in prod, SAFE-off in audits).
- [ ] Sparse divisions (hierarchical/pooled encoders for niche industries).
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

- [ ] Shift-14 PASS (ΔAUC ≤ 0.01; ΔLift@10 ≤ 0.25); Shift-grid non-improving.
- [ ] Permutation: p ≤ 0.01 and meaningful degradation.
- [ ] Forward-months: gentle decline; no paradoxical early gains.
- [ ] Rollback toggles documented (disable SAFE/purge/permutation/shift-grid) for forensics only.
