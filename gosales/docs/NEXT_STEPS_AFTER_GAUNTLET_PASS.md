# GoSales — Post Gauntlet **PASS** Roadmap
**Audience:** OpenAI Codex CLI Agent + GoSales team  
**Date:** generated from latest artifacts (cutoff=2024‑12‑31)

---

## 0) Executive summary (where we are)

- **Leakage Gauntlet: PASS** for Printers and Solidworks at the 2024‑12‑31 cutoff (per your attached `summary_2024-12-31.md`).  
- The Gauntlet now runs with **GroupKFold**, **purged/embargoed splits**, **SAFE feature policy**, and **tail masking**.
- We should treat PASS as **necessary** (models aren’t cheating with near‑cutoff momentum) but not **sufficient** (we still need to prove horizon‑robust accuracy and improve signal).

> Two auxiliary diagnostics were also produced:
> - **Label permutation** plots (one per division). Current JSON shows `baseline_auc == permuted_auc_mean` to numerical precision for both divisions, which indicates a **bug** in the permutation routine (we’re likely refitting but not actually destroying label–feature alignment in train). Fixing this is part of the checklist below.
> - **Importance stability (bootstrapped LR coefficients)** shows moderate stability. Useful as an integrity check, not a pass/fail gate.

---

## 1) The principle going forward

We optimize *honest* ranking power for sales reps. That means:
- **Measurement is sacred**: purged, group‑safe CV + forward months + Shift‑N should never flatter us.
- **Adjacency is contained**: SAFE mode remains *audit‑only*; production models can continue to use legitimate recent signals once audits prove they don’t overfit to the boundary.
- **Reporting is decision‑ready**: every run emits artifacts that a non‑ML stakeholder can read (model card, horizon curves, lift tables).

---

## 2) Immediate fixes (48–72h)

1) **Repair the label permutation test**
   - **Goal:** When labels are permuted within time buckets, AUC should collapse toward `≈0.5`. Our current output shows no degradation, which is physically implausible and signals an implementation bug.
   - **Changes:** Re‑seed per permutation; guarantee that only **train labels** are shuffled; keep **splits fixed**; evaluate on original validation labels.
   - **Artifacts:** `outputs/leakage/<division>/<cutoff>/permutation_diag.json` with keys: `baseline_auc`, `permuted_auc_mean`, `permuted_auc_std`, `p_value`, and a `perm_auc_hist.png` plot.
   - **PASS:** `baseline_auc - permuted_auc_mean ≥ 0.05` **or** `p_value ≤ 0.01` (one‑sided).

2) **Make PASS reproducible**
   - Pin `random_state` and fold assignment; ensure **no customer overlap** CSVs are empty.
   - Re‑run Gauntlet twice; deltas of `ΔAUC`, `ΔLift@10` between runs should be ≤ `0.002` and `0.05` respectively.

3) **Add Shift‑grid sanity**
   - Extend audit to **Shift‑{7, 14, 28, 56}**. Metrics should **not** *improve* as we move earlier; light, monotonic soft‑degradation is healthy.

---

## 3) Prove horizon‑robust accuracy (1–2 weeks)

1) **Prequential forward‑month evaluation**
   - Freeze at **2024‑06‑30**, then score month‑by‑month through **2025** (and now 2026 if data available).
   - Emit plots of **AUC**, **lift@K (K in {10, 20, 50})**, and **Brier** vs horizon (+0d, +30d, +60d, …).
   - **Acceptance:** gentle decline with horizon; improvements at earlier horizons should *not* exceed Gauntlet eps.

2) **Adjacency ablation triad**
   - Train & evaluate three variants on purged, group‑safe CV and far‑month holdout: **Full**, **No‑recency/short‑windows**, **SAFE**.
   - **Acceptance:** `AUC(Full) − AUC(SAFE) ≤ 0.01` on CV **and** far‑month; if Full ≫ SAFE only on CV, evaluation still adjacency‑biased.

3) **Calibration & business yield**
   - Persist **Platt** and **Isotonic** calibration metrics per division (Cal‑MAE, ECE).  
   - Report **top‑K yield** (actual buys / contacted) and **coverage curves** so sales can pick a K that matches capacity.

---

## 4) Accuracy roadmap (data & modeling)

- **Cycle‑aware features:** time‑to‑reorder estimators and hazard‑style recency transforms (e.g., log‑recency, bucketized tenure × frequency).  
- **Trajectory without adjacency:** use **offset windows** (e.g., 12‑month block ending `cutoff−60d`) and **delta features** (12m vs 24m prior blocks).  
- **Affinity features:** clean ALS embeddings for audit but allow in production; add **co‑purchase lift** features with *lagged exposure* (last seen SKU family ≥ 60d before cutoff).  
- **Sparse divisions:** hierarchical shrinkage / pooled encoders for niche industries to stabilize estimates.  
- **Model class exploration:** continue with LR/LGBM; optionally trial **GBDT with monotonic constraints** on known monotone features (e.g., negative monotone with `days_since_last_order`).  
- **Calibration stability:** per‑division choice between Platt vs Isotonic based on volume; maintain calibration curves in artifacts.

---

## 5) Reporting & ops (make it easy to trust)

- **Model cards** per division with: dataset size, CV scheme, Gauntlet status, horizon curves, top‑K tables, calibration, and major features.  
- **Run registry**: `runs/manifest.json` summarizes params → artifacts for reproducibility.  
- **CI gates:** block PRs unless Gauntlet PASS, permutation p‑value ≤ 0.01, and no overlap CSV empty.  
- **Streamlit dashboards:** single page for a division to view recent leakage reports, horizon plots, and download CSVs for sales ops.

---

## 6) Concrete commands

```powershell
# Re-run Gauntlet (Printers)
$env:PYTHONPATH="$PWD"
python -m gosales.pipeline.run_leakage_gauntlet `
  --division Printers `
  --cutoff 2024-12-31 `
  --window-months 6 `
  --group-cv `
  --safe-mode `
  --purge-days 60 `
  --run-shift14-training `
  --shift14-eps-auc 0.01 `
  --shift14-eps-lift10 0.25

# Re-run Gauntlet (Solidworks)
python -m gosales.pipeline.run_leakage_gauntlet `
  --division Solidworks `
  --cutoff 2024-12-31 `
  --window-months 6 `
  --group-cv `
  --safe-mode `
  --purge-days 60 `
  --run-shift14-training `
  --shift14-eps-auc 0.01 `
  --shift14-eps-lift10 0.25
```

**Inspect:**  
- `outputs/leakage/<Division>/<Cutoff>/shift14_metrics_*.json` → `status: PASS`, deltas within eps.  
- `outputs/leakage/fold_customer_overlap_*.csv` → empty.  
- `outputs/leakage/leakage_report_*.json` → overall PASS.  
- `outputs/leakage/permutation_diag.json` → `p_value ≤ 0.01`, `baseline_auc` ≫ permutation mean (~0.5).

---

## 7) Minimal code diffs the agent can apply

> These diffs are deliberately surgical so they apply cleanly even if the agent’s previous run stopped mid‑stream.

**A) Fix label permutation (refit with shuffled train labels; compute p‑value)**

```diff
diff --git a/gosales/pipeline/run_leakage_gauntlet.py b/gosales/pipeline/run_leakage_gauntlet.py
@@
-    rng = np.random.RandomState(seed)
-    auc_perm: list[float] = []
-    for _ in range(int(n_perm)):
-        y_perm = y.copy()
-        try:
-            for g in np.unique(groups):
-                mask = (groups == g)
-                rng.shuffle(y_perm[mask])
-        except Exception:
-            rng.shuffle(y_perm)
+    auc_perm: list[float] = []
+    for i in range(int(n_perm)):
+        # independent seed per permutation to avoid accidental state reuse
+        rng = np.random.RandomState(seed + i)
+        y_perm = y.copy()
+        # shuffle **only within train** and within time buckets to preserve base rates
+        try:
+            for g in np.unique(groups[it]):
+                mask = (groups[it] == g)
+                rng.shuffle(y_perm[it][mask])
+        except Exception:
+            rng.shuffle(y_perm[it])
@@
-        try:
-            # Evaluate against true labels to measure degradation
-            a = float(roc_auc_score(y[iv], pp))
-        except Exception:
-            a = float('nan')
+        # Evaluate against true (unshuffled) validation labels
+        a = float(roc_auc_score(y[iv], pp))
         auc_perm.append(a)
@@
-    stats = {
+    # one-sided p-value: P(AUC_perm >= AUC_baseline)
+    perm = np.array(auc_perm, dtype=float)
+    p_value = float(((perm >= auc_baseline).sum() + 1) / (len(perm) + 1))
+    stats = {
         'baseline_auc': auc_baseline,
         'permuted_auc_mean': float(np.mean(auc_perm)) if auc_perm else None,
         'permuted_auc_std': float(np.std(auc_perm)) if auc_perm else None,
         'n_permutations': int(len(auc_perm)),
-        'auc_degradation': (auc_baseline - float(np.mean(auc_perm))) if auc_perm else None,
+        'auc_degradation': (auc_baseline - float(np.mean(auc_perm))) if auc_perm else None,
+        'p_value': p_value,
     }
```

**B) Add Shift‑grid (optional but recommended)**

```diff
diff --git a/gosales/pipeline/run_leakage_gauntlet.py b/gosales/pipeline/run_leakage_gauntlet.py
@@
-    parser.add_argument('--run-shift14-training', action='store_true')
+    parser.add_argument('--run-shift14-training', action='store_true')
+    parser.add_argument('--shift-grid', type=str, default='7,14,28,56',
+                        help='comma list of day offsets to evaluate (earlier is positive).')
@@
-    if args.run_shift14_training:
-        # existing Shift‑14 block...
+    if args.run_shift14_training:
+        shifts = [int(x) for x in args.shift_grid.split(',') if x.strip()]
+        for d in shifts:
+            # call training/eval with cutoff_minus_d and write metrics as shift{d}_metrics_*.json
```

---

## 8) Acceptance & rollback

- **Gauntlet**: `ΔAUC ≤ 0.01` and `Δlift@10 ≤ 0.25` for Shift‑14; Shift‑grid non‑improving; permutation `p ≤ 0.01`.
- **Forward months**: gentle decline; no horizon where earlier cutoffs look better than later ones.
- **Rollback knobs**: `--no-safe-mode`, `--purge-days 0`, and disable permutation/shift‑grid in Gauntlet if you need to reproduce pre‑hardening behavior (forensics only).

---

## 9) Hand‑off checklist (for the agent)

- [ ] Validate PASS deterministically (two Gauntlet runs; same results within small noise).  
- [ ] Apply permutation‑test patch; re‑run; confirm `p_value ≤ 0.01`.  
- [ ] Add Shift‑grid; confirm non‑improving trend across {7,14,28,56}.  
- [ ] Produce prequential horizon plots for 2025; attach to `outputs/reports/`.  
- [ ] Generate model cards per division with Gauntlet status, horizon curves, calibration, and top‑K tables.  
- [ ] Wire CI gates: fail PR if Gauntlet FAIL, permutation `p>0.01`, or customer overlap non‑empty.  
- [ ] Summarize in a single `NEXT_STEPS_REPORT.md` and link artifacts.

---

**Done.** This plan keeps the honesty gates locked while giving us a clean runway to add data and iterate on modeling without re‑introducing time adjacency.

---

## Update (2025-09-08)

- Diagnostics: permutation test fixed (train-only shuffle within time buckets, p-value added). Importance stability emitted. Gauntlet now attaches diagnostics summaries and UI renders summaries + plots.
- Reproducibility: PASS for both divisions (?AUC=0.0000, ?Lift@10=0.0000, overlap=0). Artifacts in outputs/leakage/<division>/<cutoff>/repro_check_*.json.
- Shift-grid: Implemented and run at {7,14,28,56}. Both divisions show non-improving metrics as we shift earlier. See shift_grid_<division>_<cutoff>.json.
- Prequential: Added gosales/pipeline/prequential_eval.py and produced initial horizon curves (2025-01..03) for both divisions. Artifacts in outputs/prequential/<division>/<train_cutoff>/ (JSON/CSV/curves PNG).
