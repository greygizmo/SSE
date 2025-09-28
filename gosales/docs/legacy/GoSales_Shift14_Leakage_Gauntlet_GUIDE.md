
# GoSales – Shift‑14 Leakage Gauntlet Failures & “Suspiciously Accurate” Models
**Audience:** OpenAI Codex CLI agent (and human reviewers)  
**Goal:** Provide a complete, code‑forward plan to (1) diagnose why models look “too good,” (2) fix evaluation contamination, (3) harden the Leakage Gauntlet, and (4) validate horizon‑robust performance.

---

## 0) Executive Summary

**Symptom:** In the Leakage Gauntlet’s **Shift‑14** check (training at `cutoff - 14 days`), both **Printers** and **Solidworks** improve **AUC** and **lift@10** beyond epsilons instead of degrading. That is a classic **time‑adjacency** leak pattern (not “future data,” but features overly coupled to the boundary).

**Why we care:** You started this Gauntlet because real‑world predictions felt **suspiciously accurate**. Shift‑14 getting **better** is a smoking gun that your evaluation was **over‑optimistic** due to:
- **CV mode mismatch** (Gauntlet path training without GroupKFold, enabling cross‑customer leakage).
- **Time adjacency** (short windows/recency/expiring/subs‑shares/ALS embeddings harvesting momentum right before the target window).

**Plan:** Fix **measurement** (GroupKFold + purge/embargo gap) and introduce **SAFE mode** (lag + drop high‑risk families) specifically for Gauntlet audits. Use this as a **binding gate**, plus add **forward‑month** and **ablation** checks to verify the model’s accuracy is horizon‑robust.

**Acceptance:** For each division/cutoff under audit: `ΔAUC ≤ 0.01` and `Δlift@10 ≤ 0.25` for Shift‑14, **PASS** in Gauntlet metrics, **no customer overlap** across folds; plus forward‑month degradation behaves sanely (gentle decline, not edge‑boost).

---

## 1) Repo Context (quick orientation)

- **Purpose:** Division‑level ICP + Whitespace engine. Ingest sales logs, build curated star (`fact_transactions`, `dim_customer`), engineer features as of a **cutoff date**, train calibrated models per division, output ICP scores + whitespace rankings.
- **Key cutoffs:** Train at **2024‑06‑30** for 2H‑2024 internal test and **2025** forward validation. Gauntlet stresses a **2024‑12‑31** cutoff and a **Shift‑14** variant (`cutoff - 14 days`).

**Pipelines & relevant code (by area):**  
- **Config & plumbing:** `gosales/config.yaml`, `gosales/utils/config.py`, `gosales/utils/paths.py`, `gosales/utils/db.py`  
- **SQL safety & ETL assets:** `gosales/utils/sql.py`, `gosales/sql/queries.py`, `gosales/etl/assets.py`, `gosales/etl/build_star.py`, `gosales/etl/sku_map.py`  
- **Feature engineering:** `gosales/features/engine.py`, `gosales/features/als_embed.py` (if present), `gosales/features/utils.py` (if present)  
- **Trainer & pipelines:** `gosales/models/train.py`, `gosales/pipeline/run_leakage_gauntlet.py`, `gosales/pipeline/score_customers.py`, `gosales/pipeline/score_all.py`  
- **QA / Scripts:** `scripts/ci_featurelist_alignment.py`, `scripts/ci_assets_sanity.py`, `scripts/ablation_assets_off.py`, `scripts/build_features_for_models.py`, `scripts/feature_importance_report.py`, `scripts/leakage_summary.py`

**Artifacts of record (by division/cutoff):**  
`gosales/outputs/leakage/<Division>/<Cutoff>/`  
- `leakage_report_<Division>_<Cutoff>.json`  
- `shift14_metrics_<Division>_<Cutoff>.json`  
- `fold_customer_overlap_<Division>_<Cutoff>.csv` (if GroupKFold on)  
- Summaries: `gosales/outputs/metrics_summary.csv`, `gosales/outputs/drift_snapshots.csv`, `gosales/outputs/feature_importance_*.csv`

---

## 2) Observed Metrics (why we flagged this)

### Printers @ 2024‑12‑31
- **Baseline vs Shift‑14 (pre‑guard):**  
  - AUC: `0.9340 → 0.9600` (**+0.0260**)  
  - lift@10: `≈ +0.91` improvement (later computed)  
- **After adding lift compare:**  
  - `auc_base: 0.9340`, `auc_shift: 0.9600`  
  - `lift10_base: 7.5963`, `lift10_shift: 8.5105`  
  - `brier_base: 0.00577`, `brier_shift: 0.00603`
- **After masks/guards (still FAIL):**  
  - `auc_base: 0.9326`, `auc_shift: 0.9599`  
  - `lift10_base: 7.8637`, `lift10_shift: 8.5602`  
  - `brier_base: 0.00408`, `brier_shift: 0.00606`  
  - LR masked (sanity): `auc_lr_masked_base: 0.6433` → `0.6316`; `lift10_lr_masked_base: 2.6133` → `3.1971`  
  - LR masked‑dropped (sanity): `auc: 0.6795` → `0.7536`; `lift10: 4.1463` → `4.7727`

### Solidworks @ 2024‑12‑31
- **Baseline vs Shift‑14 (pre‑guard):**  
  - AUC: `0.8304 → 0.9404` (**+0.1101**)  
  - lift@10: `4.4904 → 6.8655` (**+2.3751**)

**Interpretation:** Moving the cutoff **earlier** should reduce information; instead it **improves** metrics. That screams **adjacency leakage** and/or **cross‑customer leakage** in CV, and implies your production‑era “accuracy” was **over‑estimated**.

---

## 3) Likely Causes (ranked)

1) **Gauntlet Shift‑14 trains without GroupKFold** (default `--group-cv` is False) ⇒ **cross‑customer leakage** across CV folds inflates metrics precisely where adjacency is strongest.  
2) **Time adjacency** in features: Recency family, short RFM windows (≤3m), division dynamics, expiring windows, subscription shares/compositions, ALS‑style embeddings. Even with 14–30 day tail masks, 14 days is inside the “momentum band.”  
3) **Apples‑to‑apples alignment:** Ensure Shift‑14 comparisons use same pipeline, grids, calibration, seeds, CV mode.  
4) **Label construction edge‑cases:** Not a post‑cutoff leak per se, but if labels cluster heavily in the first 14–30 days after cutoff, adjacency pressure is very high; evaluation must embargo.

---

## 4) Remediation – Two Complementary Tracks

### Track A — **Gauntlet‑SAFE** training and features (audit‑only)

**Intent:** Make the **Leakage Gauntlet** adversarial to adjacency. Enforce:  
- **GroupKFold** by `customer_id` (no cross‑customer leakage),  
- a **purge/embargo gap** (30–60d) between train/valid cohorts based on **recency‑days**,  
- and a **SAFE feature policy** that **lags windows** and **drops high‑risk feature families** during Gauntlet training.

#### A1. Patch: Force GroupKFold + SAFE + purge in Gauntlet’s Shift‑14 subprocess

_File: `gosales/pipeline/run_leakage_gauntlet.py`_

```diff
*** a/gosales/pipeline/run_leakage_gauntlet.py
--- b/gosales/pipeline/run_leakage_gauntlet.py
@@
-        cmd = [sys.executable, "-m", "gosales.models.train",
-               "--division", division, "--cutoffs", cutoff,
-               "--window-months", str(window_months)]
+        cmd = [sys.executable, "-m", "gosales.models.train",
+               "--division", division, "--cutoffs", cutoff,
+               "--window-months", str(window_months),
+               "--group-cv",
+               "--safe-mode"]
+        # read purge/safe lag defaults from config.validation (fallbacks)
+        from gosales.utils.config import load_config
+        _cfg = load_config()
+        _purge = int(getattr(getattr(_cfg, "validation", object()), "purge_days", 45) or 45)
+        _safe_lag = int(getattr(getattr(_cfg, "validation", object()), "safe_lag_days", 45) or 45)
+        if _purge > 0:
+            cmd += ["--purge-days", str(_purge)]
+        if _safe_lag > 0:
+            cmd += ["--safe-lag-days", str(_safe_lag)]
@@
-        cmd2 = [sys.executable, "-m", "gosales.models.train",
-                "--division", division, "--cutoffs", cut_shift,
-                "--window-months", str(window_months)]
+        cmd2 = [sys.executable, "-m", "gosales.models.train",
+                "--division", division, "--cutoffs", cut_shift,
+                "--window-months", str(window_months),
+                "--group-cv",
+                "--safe-mode"]
+        if _purge > 0:
+            cmd2 += ["--purge-days", str(_purge)]
+        if _safe_lag > 0:
+            cmd2 += ["--safe-lag-days", str(_safe_lag)]
```

> **Why:** The Gauntlet is the instrument. Enforce the no‑leak CV mode and lag/drops here to test for adjacency dependence.


#### A2. Patch: Add SAFE + purge knobs to trainer CLI and apply **recency‑based embargo**

_File: `gosales/models/train.py`_

```diff
*** a/gosales/models/train.py
--- b/gosales/models/train.py
@@
-@click.option("--group-cv/--no-group-cv", default=False, help="Use GroupKFold by customer_id for train/valid split (leakage guard)")
+@click.option("--group-cv/--no-group-cv", default=False, help="Use GroupKFold by customer_id for train/valid split (leakage guard)")
+@click.option("--purge-days", type=int, default=0, help="Recency embargo gap between train and valid cohorts (days).")
+@click.option("--safe-mode/--no-safe-mode", default=False, help="Use SAFE feature policy for Gauntlet audits (drop/lag adjacency-heavy families).")
+@click.option("--safe-lag-days", type=int, default=0, help="Lag amount (days) applied to rolling windows when --safe-mode.")
-def main(..., group_cv: bool, dry_run: bool) -> None:
+def main(..., group_cv: bool, dry_run: bool, purge_days: int, safe_mode: bool, safe_lag_days: int) -> None:
@@
-        fm = create_feature_matrix(engine, division, cutoff, window_months)
+        # SAFE: lag windows by safe_lag_days via mask_tail_days
+        _mask_tail = safe_lag_days if (safe_mode and safe_lag_days and safe_lag_days > 0) else None
+        fm = create_feature_matrix(engine, division, cutoff, window_months, mask_tail_days=_mask_tail)
         df = fm.to_pandas()
         y = df['bought_in_division'].astype(int).values
-        X = df.drop(columns=['customer_id','bought_in_division'])
+        X = df.drop(columns=['customer_id','bought_in_division']).copy()
+        if safe_mode:
+            from gosales.features.engine import safe_filter_columns
+            X = safe_filter_columns(X, keep=['rfm__all__recency_days__life'])
@@
-        if group_cv:
+        if group_cv:
             from sklearn.model_selection import GroupKFold
             gkf = GroupKFold(n_splits=cfg.modeling.folds)
             groups = df['customer_id'].astype(str).values
             splits = list(gkf.split(X, y, groups))
             train_idx, valid_idx = splits[-1]
+            # Embargo gap by recency
+            rec_col = 'rfm__all__recency_days__life'
+            if purge_days and rec_col in df.columns:
+                import numpy as _np
+                rec = _np.nan_to_num(df[rec_col].astype(float).values, nan=1e9)
+                r_valid_max = float(_np.max(rec[valid_idx]))
+                keep = [i for i in train_idx if rec[i] >= (r_valid_max + purge_days)]
+                train_idx = _np.array(keep, dtype=int)
         else:
             X_train, X_valid, y_train, y_valid = _train_test_split_time_aware(X, y, cfg.modeling.seed)
+            if purge_days:
+                import numpy as _np
+                rec_col = 'rfm__all__recency_days__life'
+                if rec_col in df.columns:
+                    rec_all = _np.nan_to_num(df[rec_col].astype(float).values, nan=1e9)
+                    r_valid_max = float(_np.max(rec_all[X_valid.index]))
+                    keep = [i for i in X_train.index if rec_all[i] >= (r_valid_max + purge_days)]
+                    X_train = X_train.loc[keep]; y_train = y_train[X_train.index]
```

> **Why:** GroupKFold kills cross‑customer leakage; the recency‑based **purge** removes adjacency coupling between train/valid. SAFE drops/lag remove inherently adjacency‑heavy features during the audit.


#### A3. Patch: Implement **SAFE feature filter**

_File: `gosales/features/engine.py`_

```diff
*** a/gosales/features/engine.py
--- b/gosales/features/engine.py
@@
-from typing import Optional
+from typing import Optional, Iterable
+import pandas as pd
@@
 def create_feature_matrix(...):
     ...
     return FeatureMatrix(df)
+
+def safe_filter_columns(df: pd.DataFrame, keep: Iterable[str] = ()) -> pd.DataFrame:
+    """
+    Drop adjacency-heavy families for Gauntlet SAFE mode.
+    Kept by default: 'rfm__all__recency_days__life' so trainer can use it for splitting/embargo.
+    """
+    keep = set(str(k) for k in (keep or []))
+    drop = []
+    for c in list(df.columns):
+        cl = str(c).lower()
+        if c in keep:
+            continue
+        # High-risk families near the cutoff:
+        if cl.startswith("assets_expiring_"):      drop.append(c); continue
+        if "recency" in cl or "days_since_last" in cl:  drop.append(c); continue
+        if cl.startswith("assets_subs_") or cl.endswith("_share") or "_share_" in cl:  drop.append(c); continue
+        if cl.endswith("_30d") or cl.endswith("_60d") or cl.endswith("_90d") or "__3m" in cl or "_3m_" in cl:  drop.append(c); continue
+        if cl.startswith("als__") or cl.startswith("als_f"):  drop.append(c); continue
+    return df.drop(columns=drop, errors="ignore")
```

> **Targets removed in SAFE:** `assets_expiring_*`, recency & `days_since_last_*`, subs/share compositions, any ≤90d/≤3m windows, and ALS embeddings. Keeps `rfm__all__recency_days__life` for split/embargo computations.


#### A4. Config defaults (optional, Gauntlet‑oriented)

_File: `gosales/config.yaml`_

```yaml
validation:
  shift14_epsilon_auc: 0.01
  shift14_epsilon_lift10: 0.25
  gauntlet_mask_tail_days: 14
  purge_days: 45
  safe_lag_days: 45
```

> **Note:** You can pass flags instead of pinning defaults here. Use 45–60d near fiscal boundaries; 30d mid‑year.


#### A5. Optional: Mirror **purge** in Gauntlet’s inline quick‑eval (if used)

Inside `run_leakage_gauntlet.py` LR quick evaluator (where you sort by `recency_days` and split 80/20), apply the same **embargo gap** between train/valid by filtering training indices whose recency is within `purge_days` of validation’s max‑recency.


---

### Track B — **Purged / embargoed CV** (general trainer support)

- Keep SAFE mostly **Gauntlet‑only**, but add `--purge-days` as a **standard** option to trainer and use it for **model selection**.  
- Recommended **purge_days**: **45–60** near Nov‑Jan; **30** otherwise.  
- This slightly reduces train size but yields **time‑robust** CV metrics and prevents “edge‑boost” from dominating during selection.


---

## 5) Does this fix the “suspiciously accurate” core issue? Yes—if we enforce it as a gate

- **Diagnosis:** Shift‑14 getting better means your evaluation setup is flattering you—either from cross‑customer leakage (no GroupKFold) or adjacency features.  
- **Repair:** Gauntlet with **GroupKFold + purge + SAFE** becomes your **circuit breaker**. Any model that passes cannot be relying on the near‑boundary momentum trick.  
- **Incentives:** Make Gauntlet **binding**—no champion without a PASS. Promote **purge + group‑cv** to your **model selection** path as well.


---

## 6) Verification & Acceptance Criteria

### Gauntlet PASS criteria (per division/cutoff)

- `ΔAUC = auc_shift − auc_base ≤ 0.01`  
- `Δlift@10 = lift10_shift − lift10_base ≤ 0.25`  
- `status: PASS` in `shift14_metrics_<Division>_<Cutoff>.json`  
- **No customer overlap** across folds with GroupKFold (empty `fold_customer_overlap_*.csv`)  
- Consolidated `leakage_report_<Division>_<Cutoff>.json` shows **overall: PASS**

### Commands (PowerShell; from repo root)

```powershell
# Ensure module path
$env:PYTHONPATH = "$PWD"

# Re-run Gauntlet: Printers
python -m gosales.pipeline.run_leakage_gauntlet `
  --division Printers `
  --cutoff 2024-12-31 `
  --window-months 6 `
  --no-static-only `
  --run-shift14-training `
  --shift14-eps-auc 0.01 `
  --shift14-eps-lift10 0.25

# Re-run Gauntlet: Solidworks
python -m gosales.pipeline.run_leakage_gauntlet `
  --division Solidworks `
  --cutoff 2024-12-31 `
  --window-months 6 `
  --no-static-only `
  --run-shift14-training `
  --shift14-eps-auc 0.01 `
  --shift14-eps-lift10 0.25
```

### Inspect these artifacts

- `gosales/outputs/leakage/<Division>/<Cutoff>/shift14_metrics_<Division>_<Cutoff>.json`  
  Expect keys: `auc_base`, `auc_shift`, `lift10_base`, `lift10_shift`, `status` (→ **PASS**), and deltas within eps.  
- `gosales/outputs/leakage/fold_customer_overlap_<Division>_<Cutoff>.csv`  
  **Empty** when `--group-cv` is active.  
- `gosales/outputs/leakage/leakage_report_<Division>_<Cutoff>.json`  
  Consolidated **overall: PASS**.


---

## 7) Portfolio‑level checks (go beyond Gauntlet; prove horizon‑robustness)

1) **Prequential (rolling) forward‑month evaluation**  
   - Freeze model at **2024‑06‑30**; score monthly slices in **2H‑2024** and **2025** sequentially.  
   - Plot AUC, lift@10, Brier vs **distance from training cutoff** (0, +30d, +60d, …).  
   - **Pass behavior:** gentle decline; **Fail behavior:** improvement or sharp knee near the embargo horizon.

2) **Adjacency ablation triad (same cutoff)**  
   - **Full** (today’s features), **No‑recency/short‑windows**, **SAFE**.  
   - Evaluate on **purged group‑CV** and a **far‑month holdout**.  
   - **Red flag:** Full ≫ No‑recency on CV but ≈ on far‑month holdout ⇒ CV still adjacency‑biased.

3) **Time‑bucket permutation placebo**  
   - Within months immediately before cutoff, **permute labels** and re‑fit; AUC should crash toward 0.5.  
   - If AUC stays elevated, you still have structural leakage (e.g., identity bleed‑through).

**Forward‑robustness gates (suggested):**  
- Average AUC drop from +0d→+60d ≥ **0.005** (or non‑improving within 95% CI triggers fail).  
- `AUC_gap(Full − SAFE)` on purged CV ≤ **0.01** **and** on far‑month holdout ≤ **0.01**.  
- Brier at +60–90d not worse than +0–30d by > **0.002**.


---

## 8) Label sanity (belt‑and‑suspenders) SQL

Use this to audit label timing pressure around the cutoff:

```sql
SELECT
  cutoff_date,
  SUM(CASE WHEN label_date <= cutoff_date THEN 1 ELSE 0 END) AS labels_on_or_before_cutoff,
  SUM(CASE WHEN label_date > cutoff_date AND label_date <= DATEADD(day,14,cutoff_date) THEN 1 ELSE 0 END) AS labels_in_first_14d,
  SUM(CASE WHEN label_date > cutoff_date AND label_date <= DATEADD(day,30,cutoff_date) THEN 1 ELSE 0 END) AS labels_in_first_30d
FROM labels_table -- replace with your label source/view
GROUP BY cutoff_date;
```

If the first‑14d density is high, justify a **larger Gauntlet purge (60d)**.


---

## 9) Risks, Tradeoffs, Rollback

- **Metric dip:** Absolute AUC/lift may soften under purge/SAFE. That’s expected; it’s **honest**.  
- **Train size:** Purge shrinks train rows; tune 30–60d by seasonality.  
- **Rollback:** All changes are **flag‑gated**. Disable with `--no-safe-mode` and `--purge-days 0`; remove `--group-cv` in Gauntlet if you must reproduce legacy results (not recommended).


---

## 10) Agent TODO – crisp checklist

### A) Gauntlet & Trainer code
- [ ] Modify `gosales/pipeline/run_leakage_gauntlet.py` to call trainer with `--group-cv --safe-mode`, and forward `--purge-days` and `--safe-lag-days` from `config.validation` (fallback 45).
- [ ] Extend `gosales/models/train.py` CLI with `--purge-days`, `--safe-mode/--no-safe-mode`, `--safe-lag-days`.
- [ ] In trainer, implement **recency‑based purge** in both GroupKFold and time‑aware split branches using `rfm__all__recency_days__life`.
- [ ] Keep/verify **fold overlap audit** (empty `fold_customer_overlap_*.csv` when group CV enabled).

### B) Feature engine (SAFE)
- [ ] Add `safe_filter_columns(df, keep=[...])` to drop adjacency families: `assets_expiring_*`, any `*recency*` or `*days_since_last*`, `assets_subs_*` and any `*_share*`, any ≤90d/≤3m window (`*_30d`, `*_60d`, `*_90d`, `*__3m*`), and `als__*`/`als_f*` embeddings.
- [ ] Ensure `create_feature_matrix(..., mask_tail_days=)` supports lagging windows when SAFE is on (`mask_tail_days = safe_lag_days`).

### C) Config
- [ ] In `gosales/config.yaml`, under `validation`, add (or confirm):  
  `shift14_epsilon_auc: 0.01`, `shift14_epsilon_lift10: 0.25`, `gauntlet_mask_tail_days: 14`, `purge_days: 45`, `safe_lag_days: 45`.

### D) Run & verify
- [ ] Re‑run Gauntlet for **Printers** and **Solidworks** at `2024‑12‑31` using the PowerShell commands in §6.
- [ ] Confirm `shift14_metrics_*.json` → **PASS** with deltas within eps.
- [ ] Confirm consolidated `leakage_report_*.json` → **overall: PASS**.
- [ ] Confirm `fold_customer_overlap_*.csv` is empty.

### E) Beyond Gauntlet (prove horizon‑robustness)
- [ ] Implement **prequential forward‑month** scoring: model trained at `2024‑06‑30` scored across monthly slices through **2025**; plot AUC/lift/Brier vs horizon.
- [ ] Run **ablation triad** (Full vs No‑recency/short‑windows vs SAFE) on purged CV and on far‑month holdout; inspect gaps.
- [ ] Run **time‑bucket permutation placebo** near cutoff; verify AUC collapses toward 0.5.
- [ ] Add these checks (or a lighter version) to CI or a scheduled QA job.

### F) Deployment hygiene
- [ ] Promote **GroupKFold + purge** to the **model selection** stage for champion pick.
- [ ] Consider a **T‑30 freeze** (or feature‑family lag) for production if business latency allows.


---

## 11) Single‑page “Why this works”

- **GroupKFold** removes **cross‑customer memorization** that flatters CV.  
- **Purge/embargo** introduces a **temporal gap**, denying the model the easy momentum right at the boundary.  
- **SAFE** removes/defers inherently adjacency‑heavy families for the audit, so passing the Gauntlet means the model relies on **structural behavior**, not **edge effects**.  
- The **portfolio checks** prove that accuracy persists **beyond** the momentum band, making your “suspiciously accurate” results either **validated** or **exposed**—both are wins.

---

## 12) Known numbers to anchor expectations

- **Printers (pre‑guard)**: AUC `0.9340 → 0.9600 (+0.0260)`; lift@10 `7.5963 → 8.5105 (+0.9142)`; Brier worsens slightly.  
- **Printers (after guards; still FAIL)**: AUC `0.9326 → 0.9599 (+0.0273)`; lift@10 `7.8637 → 8.5602 (+0.6965)`; Brier `0.00408 → 0.00606`.  
- **Solidworks (pre‑guard)**: AUC `0.8304 → 0.9404 (+0.1101)`; lift@10 `4.4904 → 6.8655 (+2.3751)`.

These magnitudes are too large to attribute to real foresight; they are consistent with evaluation contamination and time adjacency.

---

**End of file.**
