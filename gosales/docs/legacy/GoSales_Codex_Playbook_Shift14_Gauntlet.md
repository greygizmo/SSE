
# GoSales Codex Playbook — Shift‑14 Leakage Gauntlet, Time‑Adjacency, and Honest Accuracy

**Audience:** OpenAI Codex CLI agent (and human reviewers)  
**Objective:** Provide every detail needed to (1) understand why our models looked “too good,” (2) harden evaluation against time‑adjacency and cross‑customer leakage, (3) patch the code with unified diffs, and (4) validate that models remain accurate in a horizon‑robust way that truly helps sales reps.

---

## 0) Executive Summary (Read Me First)

**Symptom:** In the Leakage Gauntlet’s **Shift‑14** check (training at `cutoff − 14 days`), both **Printers** and **Solidworks** show **better** AUC and lift@10 than at the true cutoff—breaching our epsilon thresholds (`ΔAUC ≤ 0.01`, `Δlift@10 ≤ 0.25`).

**Diagnosis:** That pattern is classic **time‑adjacency leakage** amplified by a **CV mode mismatch** (Gauntlet subprocess did not use GroupKFold) and proximity features (short/mid windows, recency, expiring, subs‑shares, ALS). Evaluation has been flattering our models—hence “suspiciously accurate.”

**Plan:** Fix the **measurement** and remove **shortcuts** during audit:  
- Enforce **GroupKFold** and a **purge/embargo** gap (30–60d) between train and validation cohorts.  
- Introduce a **SAFE** mode for **Gauntlet only** to **lag** windows and **drop** adjacency‑heavy families.  
- Optionally upgrade CV to **Blocked + Purged GroupKFold** (deterministic, time‑ordered).  
- Verify with **prequential forward‑month** scoring and **ablations**.

**Acceptance:** For each division/cutoff under audit: **Shift‑14 PASS** with `ΔAUC ≤ 0.01` and `Δlift@10 ≤ 0.25`, no customer overlap across folds, sensible forward‑month behavior.

---

## 1) Context: What this repo does

- **Purpose:** Division‑level ICP + Whitespace engine. Ingest sales logs, build curated star (`fact_transactions`, `dim_customer`), engineer features **as of cutoff**, train calibrated per‑division models, output ICP scores + whitespace rankings.  
- **Key cutoffs:** Train at **2024‑06‑30** (leave 2H‑2024 for internal testing, **2025** for forward validation). Gauntlet stresses **2024‑12‑31** and **Shift‑14** (`cutoff − 14 days`).

**Relevant code areas**  
- **Config & plumbing:** `gosales/config.yaml`, `gosales/utils/config.py`, `gosales/utils/paths.py`, `gosales/utils/db.py`  
- **SQL/ETL & assets:** `gosales/utils/sql.py`, `gosales/sql/queries.py`, `gosales/etl/assets.py`, `gosales/etl/build_star.py`, `gosales/etl/sku_map.py`  
- **Features:** `gosales/features/engine.py`, `gosales/features/als_embed.py` (if present), `gosales/features/utils.py` (if present)  
- **Trainer & pipelines:** `gosales/models/train.py`, `gosales/pipeline/run_leakage_gauntlet.py`, `gosales/pipeline/score_customers.py`, `gosales/pipeline/score_all.py`  
- **QA/Scripts:** `scripts/ci_featurelist_alignment.py`, `scripts/ci_assets_sanity.py`, `scripts/ablation_assets_off.py`, `scripts/build_features_for_models.py`, `scripts/feature_importance_report.py`, `scripts/leakage_summary.py`

**Artifacts of record** (per division/cutoff under `gosales/outputs/leakage/<Division>/<Cutoff>/`)  
- `shift14_metrics_<Division>_<Cutoff>.json` (AUC/lift@10 deltas + PASS/FAIL)  
- `leakage_report_<Division>_<Cutoff>.json` (consolidated)  
- `fold_customer_overlap_<Division>_<Cutoff>.csv` (should be empty under GroupKFold)

Other summaries: `gosales/outputs/metrics_summary.csv`, `gosales/outputs/drift_snapshots.csv`, `gosales/outputs/feature_importance_*.csv`

---

## 2) Observed metrics that triggered this effort

### Printers @ 2024‑12‑31
- **Pre‑guard baseline vs Shift‑14:** AUC `0.9340 → 0.9600` (**+0.0260**), lift@10 `+~0.91` (later computed).  
- **After adding lift compare:** AUC `0.9340 → 0.9600`, lift@10 `7.5963 → 8.5105`, Brier `0.00577 → 0.00603`.  
- **After guards/masks (still FAIL):** AUC `0.9326 → 0.9599` (**+0.0273**), lift@10 `7.8637 → 8.5602` (**+0.6965**), Brier `0.00408 → 0.00606`.  
  LR sanity probes show residual adjacency even after initial drops.

### Solidworks @ 2024‑12‑31 (pre‑guard)
- AUC `0.8304 → 0.9404` (**+0.1101**), lift@10 `4.4904 → 6.8655` (**+2.3751**).

**Interpretation:** Earlier cutoff should weaken information. Improvement = evaluation contamination (cross‑customer CV + near‑boundary momentum).

---

## 3) Why this is the same as the “suspiciously accurate” problem

- If Shift‑14 improves, our evaluation is sampling **time‑adjacent** structure that doesn’t transport into the future.  
- **Fixing the Gauntlet is not a distraction**: it is the **instrument** that forces models to prove they’re not relying on edge effects. When Gauntlet passes under purged GroupKFold + SAFE, the accuracy you see is **honest**.

---

## 4) The Fix — Two Tracks (and then some)

### Track A — Gauntlet‑SAFE (audit‑only hardened training)

**Enforce in Gauntlet training subprocess:**  
1) **GroupKFold by `customer_id`** (no cross‑customer leakage).  
2) **Purge/embargo gap** of **45–60 days** between train/valid cohorts (by recency).  
3) **SAFE features**: **lag** windows and **drop** adjacency‑heavy families for the audit.

#### A1. Patch Gauntlet subprocess to pass the right flags

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
+        vcfg = getattr(_cfg, "validation", object())
+        _purge    = int(getattr(vcfg, "purge_days", 45) or 45)
+        _safe_lag = int(getattr(vcfg, "safe_lag_days", 45) or 45)
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

> This makes the Gauntlet actually audit with group splits + embargo + SAFE.


#### A2. Add SAFE+purge knobs to trainer and implement recency‑based embargo

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

> GroupKFold blocks cross‑customer leakage; **purge** removes near‑boundary coupling in evaluation. SAFE prunes adjacency‑heavy families for the audit.


#### A3. Implement SAFE feature filter

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

> Drops: `assets_expiring_*`, any `*recency*` or `*days_since_last*`, `assets_subs_*` and any `*_share*`, ≤90d/≤3m windows, and ALS embeddings.


#### A4. (Optional) Config defaults for Gauntlet audits

_File: `gosales/config.yaml`_

```yaml
validation:
  shift14_epsilon_auc: 0.01
  shift14_epsilon_lift10: 0.25
  gauntlet_mask_tail_days: 14
  purge_days: 45
  safe_lag_days: 45
```

> You may also pass flags via CLI instead of editing config.


### Track B — Purged / embargoed CV (general support)

- Make `--purge-days` a **first‑class** trainer option (already added above).  
- Use it for **model selection** so offline CV is time‑robust instead of adjacency‑boosted.  
- Typical settings: **45–60d** near Nov–Jan; **30d** mid‑year.

---

## 5) Phase 2 Enhancements (if stubborn FAIL remains)

If Shift‑14 still exceeds eps after Track A, apply **two** upgrades. These are still **Gauntlet‑only** unless you explicitly opt in for selection.

### (i) Aggressive SAFE — drop ≤12m windows + momentum transforms; lag 60d

**Rationale:** Residual bump can come from **mid‑range momentum** (≤6–12m) and trend/velocity/slope transforms. Make SAFE parametric.

_Additions to earlier patches:_

_File: `gosales/features/engine.py` (replace the SAFE function with parametric version)_
```diff
*** a/gosales/features/engine.py
--- b/gosales/features/engine.py
@@
-from typing import Optional, Iterable
+from typing import Optional, Iterable
 import pandas as pd
+import re
@@
-def safe_filter_columns(df: pd.DataFrame, keep: Iterable[str] = ()) -> pd.DataFrame:
+def safe_filter_columns(
+    df: pd.DataFrame,
+    keep: Iterable[str] = (),
+    safe_max_window_months: int = 6,
+) -> pd.DataFrame:
@@
-    for c in list(df.columns):
+    win_m = int(safe_max_window_months or 0)
+    rx_months = re.compile(r"__(\d{1,2})m\b|_(\d{1,3})d\b", re.IGNORECASE)
+    momentum_markers = ("trend", "velocity", "slope", "momentum", "delta", "chg")
+    for c in list(df.columns):
         cl = str(c).lower()
@@
-        if cl.startswith("als__") or cl.startswith("als_f"):  drop.append(c); continue
+        if cl.startswith("als__") or cl.startswith("als_f"):  drop.append(c); continue
+        if any(m in cl for m in momentum_markers): drop.append(c); continue
+        m = rx_months.search(cl)
+        if m:
+            months = None
+            if m.group(1): months = int(m.group(1))
+            elif m.group(2):
+                d = int(m.group(2)); months = max(1, round(d/30.0))
+            if months is not None and months <= max(1, win_m):
+                drop.append(c); continue
     return df.drop(columns=drop, errors="ignore")
```

_File: `gosales/models/train.py` (plumb the knob)_
```diff
*** a/gosales/models/train.py
--- b/gosales/models/train.py
@@
 @click.option("--safe-mode/--no-safe-mode", default=False, help="Use SAFE feature policy for Gauntlet audits (drop/lag adjacency-heavy families).")
 @click.option("--safe-lag-days", type=int, default=0, help="Lag amount (days) applied to rolling windows when --safe-mode.")
+@click.option("--safe-max-window-months", type=int, default=6, help="When --safe-mode, drop windows <= this many months.")
-def main(..., group_cv: bool, dry_run: bool, purge_days: int, safe_mode: bool, safe_lag_days: int) -> None:
+def main(..., group_cv: bool, dry_run: bool, purge_days: int, safe_mode: bool, safe_lag_days: int, safe_max_window_months: int) -> None:
@@
-        if safe_mode:
+        if safe_mode:
             from gosales.features.engine import safe_filter_columns
-            X = safe_filter_columns(X, keep=['rfm__all__recency_days__life'])
+            X = safe_filter_columns(
+                X,
+                keep=['rfm__all__recency_days__life'],
+                safe_max_window_months=safe_max_window_months,
+            )
```

_File: `gosales/pipeline/run_leakage_gauntlet.py` (set aggressive defaults for Gauntlet)_
```diff
*** a/gosales/pipeline/run_leakage_gauntlet.py
--- b/gosales/pipeline/run_leakage_gauntlet.py
@@
-        _purge    = int(getattr(vcfg, "purge_days", 45) or 45)
-        _safe_lag = int(getattr(vcfg, "safe_lag_days", 45) or 45)
+        _purge    = int(getattr(vcfg, "purge_days", 45) or 45)
+        _safe_lag = int(getattr(vcfg, "safe_lag_days", 60) or 60)
+        _safe_win = int(getattr(vcfg, "safe_max_window_months_gauntlet", 12) or 12)
@@
         if _purge > 0:
             cmd += ["--purge-days", str(_purge)]
         if _safe_lag > 0:
             cmd += ["--safe-lag-days", str(_safe_lag)]
+        cmd += ["--safe-max-window-months", str(_safe_win)]
@@
         if _purge > 0:
             cmd2 += ["--purge-days", str(_purge)]
         if _safe_lag > 0:
             cmd2 += ["--safe-lag-days", str(_safe_lag)]
+        cmd2 += ["--safe-max-window-months", str(_safe_win)]
```

_Config (optional)_
```yaml
validation:
  safe_lag_days: 60
  safe_max_window_months_gauntlet: 12
```

### (ii) Deterministic **Blocked + Purged GroupKFold**

**Goal:** Lower evaluation variance and deny accidental adjacency by **time‑ordering** group folds and **embargoing** neighbors.

_Add new module:_

_File: `gosales/models/cv.py`_
```python
from typing import Iterator, Tuple
import numpy as np
import pandas as pd

class BlockedPurgedGroupCV:
    """
    Deterministic blocked GroupKFold with a temporal ordering and a purge gap.
    Groups = customer_ids. Time anchor = last-activity date per row.
    Steps:
      1) Compute per-group anchor_time (e.g., cutoff - recency_days).
      2) Order groups by anchor_time ascending.
      3) Slice into K contiguous blocks (deterministic).
      4) For fold k as validation, training = all other blocks EXCEPT
         any groups whose anchor_time lies within `purge_days` of the
         min/max anchor_time of the validation block (embargo).
    """
    def __init__(self, n_splits: int = 5, purge_days: int = 45, seed: int = 42):
        self.n_splits = int(n_splits)
        self.purge_days = int(purge_days)
        self.seed = int(seed)

    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        anchor_days_from_cutoff: np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        df = pd.DataFrame({"group": groups.astype(str), "anchor": anchor_days_from_cutoff.astype(float)})
        g = df.groupby("group", as_index=False)["anchor"].min()
        # order groups: older (larger anchor) -> newer (smaller anchor)
        g = g.sort_values("anchor", ascending=False).reset_index(drop=True)

        n = len(g)
        block_sizes = [n // self.n_splits] * self.n_splits
        for i in range(n % self.n_splits):
            block_sizes[i] += 1
        blocks, start = [], 0
        for bsz in block_sizes:
            blocks.append(g.iloc[start:start+bsz])
            start += bsz

        # group -> row indices
        idx_by_group = {}
        for i, grp in enumerate(groups.astype(str)):
            idx_by_group.setdefault(grp, []).append(i)

        for k in range(self.n_splits):
            val_groups = set(blocks[k]["group"].tolist())
            val_anchor_min = blocks[k]["anchor"].min()
            val_anchor_max = blocks[k]["anchor"].max()

            train_groups = []
            for j, block in enumerate(blocks):
                if j == k:
                    continue
                mask = (block["anchor"] >= (val_anchor_max + self.purge_days)) | \
                       (block["anchor"] <= (val_anchor_min - self.purge_days))
                safe_block = block[mask]
                train_groups.extend(safe_block["group"].tolist())

            train_idx, val_idx = [], []
            for grp in train_groups:
                train_idx.extend(idx_by_group.get(grp, []))
            for grp in val_groups:
                val_idx.extend(idx_by_group.get(grp, []))
            yield np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)
```

_Integrate when `--group-cv` and `--purge-days` are set:_

_File: `gosales/models/train.py`_
```diff
*** a/gosales/models/train.py
--- b/gosales/models/train.py
@@
-        if group_cv:
-            from sklearn.model_selection import GroupKFold
-            gkf = GroupKFold(n_splits=cfg.modeling.folds)
-            groups = df['customer_id'].astype(str).values
-            splits = list(gkf.split(X, y, groups))
-            train_idx, valid_idx = splits[-1]
-            # Embargo gap by recency
-            rec_col = 'rfm__all__recency_days__life'
-            if purge_days and rec_col in df.columns:
-                import numpy as _np
-                rec = _np.nan_to_num(df[rec_col].astype(float).values, nan=1e9)
-                r_valid_max = float(_np.max(rec[valid_idx]))
-                keep = [i for i in train_idx if rec[i] >= (r_valid_max + purge_days)]
-                train_idx = _np.array(keep, dtype=int)
+        if group_cv:
+            rec_col = 'rfm__all__recency_days__life'
+            groups = df['customer_id'].astype(str).values
+            if purge_days and rec_col in df.columns:
+                from gosales.models.cv import BlockedPurgedGroupCV
+                import numpy as _np
+                rec = _np.nan_to_num(df[rec_col].astype(float).values, nan=1e9)
+                cv = BlockedPurgedGroupCV(n_splits=cfg.modeling.folds, purge_days=purge_days, seed=cfg.modeling.seed)
+                splits = list(cv.split(X, y, groups, anchor_days_from_cutoff=rec))
+                train_idx, valid_idx = splits[-1]
+            else:
+                from sklearn.model_selection import GroupKFold
+                gkf = GroupKFold(n_splits=cfg.modeling.folds)
+                splits = list(gkf.split(X, y, groups))
+                train_idx, valid_idx = splits[-1]
```

---

## 6) Verification & PASS Criteria

### PASS rules (per division/cutoff)
- `ΔAUC = auc_shift − auc_base ≤ 0.01`  
- `Δlift@10 = lift10_shift − lift10_base ≤ 0.25`  
- `status: PASS` in `shift14_metrics_<Division>_<Cutoff>.json`  
- **No customer overlap** in `fold_customer_overlap_<Division>_<Cutoff>.csv`  
- Consolidated `leakage_report_<Division>_<Cutoff>.json → overall: PASS`

### Commands (PowerShell)

```powershell
# Ensure module path
$env:PYTHONPATH = "$PWD"

# Printers
python -m gosales.pipeline.run_leakage_gauntlet `
  --division Printers `
  --cutoff 2024-12-31 `
  --window-months 6 `
  --no-static-only `
  --run-shift14-training `
  --shift14-eps-auc 0.01 `
  --shift14-eps-lift10 0.25

# Solidworks
python -m gosales.pipeline.run_leakage_gauntlet `
  --division Solidworks `
  --cutoff 2024-12-31 `
  --window-months 6 `
  --no-static-only `
  --run-shift14-training `
  --shift14-eps-auc 0.01 `
  --shift14-eps-lift10 0.25
```

### Inspect
- `outputs/leakage/<Division>/<Cutoff>/shift14_metrics_*.json` → PASS & deltas within eps.  
- `outputs/leakage/fold_customer_overlap_*.csv` → empty.  
- `outputs/leakage/leakage_report_*.json` → overall PASS.

---

## 7) Prove horizon‑robust accuracy (beyond Gauntlet)

1) **Prequential forward‑month evaluation**  
   - Freeze at **2024‑06‑30**; score monthly slices through **2025**.  
   - Plot AUC/lift@10/Brier vs horizon (+0d, +30d, +60d, …).  
   - **Healthy pattern:** gentle decline, not improvement near earlier cutoffs.

2) **Adjacency ablation triad**  
   - Train: **Full**, **No‑recency/short windows**, **SAFE** (audit).  
   - Evaluate on **purged group‑CV** and **far‑month holdout**.  
   - If Full ≫ No‑recency only on CV but ≈ on far‑month, CV still adjacency‑biased.

3) **Time‑bucket permutation placebo**  
   - Permute labels in months immediately before cutoff; refit. AUC should drop to ~0.5.  
   - If not, structural leakage remains (identity bleed, etc.).

**Forward gates (suggested):**  
- Avg AUC drop from +0d→+60d ≥ **0.005** (or non‑improving → fail).  
- `AUC(Full) − AUC(SAFE)` ≤ **0.01** on purged CV **and** far‑month holdout.  
- Brier at +60–90d within **0.002** of +0–30d.

---

## 8) Risks, trade‑offs, rollback

- **Metrics may dip modestly** under purge/SAFE—this is the honest baseline.  
- **Purge reduces train rows**: tune 30–60d seasonally.  
- **Rollback:** Everything is **flag‑gated**. Disable with `--no-safe-mode`, `--purge-days 0`; remove `--group-cv` in Gauntlet to reproduce legacy behavior (not recommended).

---

## 9) Practical guardrails going forward

- Use **purged GroupKFold** for model selection (not only Gauntlet).  
- Keep **SAFE** for audits; production models can use full features.  
- Monitor **drift** on slow‑moving features (6–24m RFM, tenure, mix).  
- Keep **prequential monitoring** in production to spot horizon decay early.  
- For trust/adoption, expose **interpretable slices** (e.g., “Top decile: stable 12‑month spend + service tickets down”).

---

## 10) Codex Agent — Comprehensive Checklist

### A) Gauntlet & Trainer
- [ ] Modify `gosales/pipeline/run_leakage_gauntlet.py` to pass `--group-cv --safe-mode` and forward `--purge-days`, `--safe-lag-days` (defaults from `config.validation`).
- [ ] Extend `gosales/models/train.py` CLI with `--purge-days`, `--safe-mode/--no-safe-mode`, `--safe-lag-days`.
- [ ] Implement **recency‑based purge** in both GroupKFold and time‑aware split branches using `rfm__all__recency_days__life`.
- [ ] Keep/verify **fold overlap audit** (empty `fold_customer_overlap_*.csv` with GroupKFold).

### B) Feature Engine (SAFE)
- [ ] Add `safe_filter_columns(df, keep=[...])` to drop: `assets_expiring_*`, any `*recency*` or `*days_since_last*`, `assets_subs_*` and `*_share*`, `*_30d`, `*_60d`, `*_90d`, `*__3m*`, ALS embeddings.
- [ ] Ensure `create_feature_matrix(..., mask_tail_days=)` supports lagging windows when SAFE is on (`mask_tail_days = safe_lag_days`).

### C) Phase 2 (if needed)
- [ ] Upgrade SAFE to **parametric** (`safe_max_window_months`) and drop **momentum** transforms; set Gauntlet defaults to **≤12m** windows and **60d lag**.
- [ ] Add `gosales/models/cv.py` with **BlockedPurgedGroupCV** and integrate when `--group-cv` and `--purge-days` are set.

### D) Config
- [ ] In `gosales/config.yaml` under `validation`, add/confirm:  
  `shift14_epsilon_auc: 0.01`, `shift14_epsilon_lift10: 0.25`, `gauntlet_mask_tail_days: 14`, `purge_days: 45–60`, `safe_lag_days: 45–60`, and (Phase 2) `safe_max_window_months_gauntlet: 12`.

### E) Run & Verify
- [ ] Re‑run Gauntlet for **Printers** and **Solidworks** at `2024‑12‑31` with the commands above.  
- [ ] Confirm `shift14_metrics_*.json` → **PASS**; check deltas within eps.  
- [ ] Confirm `leakage_report_*.json` → **overall: PASS**.  
- [ ] Confirm `fold_customer_overlap_*.csv` is empty.

### F) Beyond Gauntlet
- [ ] Implement **prequential forward‑month** scoring (model frozen at `2024‑06‑30`) through **2025**; chart AUC/lift/Brier vs horizon.  
- [ ] Run **ablation triad** (Full / No‑recency / SAFE) on purged CV **and** far‑month holdout; inspect gaps.  
- [ ] Run **time‑bucket permutation placebo** near the cutoff and verify AUC → ~0.5.  
- [ ] Add a minimal CI job to smoke‑test Gauntlet + one prequential point on each PR touching features/trainer.

---

## 11) Why this works (one‑pager)

- **GroupKFold** stops the model from memorizing customers across folds.  
- **Purge/embargo** adds a **temporal gap** around validation so “edge momentum” can’t bleed into train.  
- **SAFE** strips inherently adjacency‑heavy families during audits; a model that passes is relying on **durable structure**, not near‑boundary noise.  
- **Blocked + Purged GroupKFold** makes CV deterministic, time‑ordered, and resistant to fold luck.  
- **Prequential + ablations** prove the accuracy carries **months forward**, not just at the edge.

---

## 12) Known numbers to anchor expectations

- **Printers (pre‑guard):** AUC `0.9340 → 0.9600 (+0.0260)`; lift@10 `+~0.91`.  
- **Printers (after guards; still FAIL):** AUC `0.9326 → 0.9599 (+0.0273)`; lift@10 `7.8637 → 8.5602 (+0.6965)`; Brier `0.00408 → 0.00606`.  
- **Solidworks (pre‑guard):** AUC `0.8304 → 0.9404 (+0.1101)`; lift@10 `4.4904 → 6.8655 (+2.3751)`.

These jumps are far too large to be legitimate foresight; they’re exactly what evaluation contamination looks like. With the patches above, Shift‑14 should **degrade or stay flat**, not improve.

---

**End of file.**
