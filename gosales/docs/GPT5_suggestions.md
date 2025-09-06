# GPT‑5 Instructions — Leakage Gauntlet for GoSales

**Goal:** implement a rigorous, automated battery of leakage checks for every division model and cutoff.  
**Outcome:** a repeatable suite that *proves* features use **only data ≤ cutoff**, splits are **time & group safe**, and top‑line metrics **do not improve** when we remove 14 days of information.

> If any check fails, the suite must **fail fast** and emit a clear, actionable reason.

---

## 0) Scope, Definitions, and Deliverables

**Leakage types we must detect**
- **Temporal leakage:** any feature uses events **after** `cutoff_date`.  
- **Group leakage:** the same **customer_id** is in **both** train and validation for a given cutoff/fold.  
- **Label leakage:** a feature directly/indirectly encodes the *target in the prediction window*.

**Divisions covered**  
All modelled divisions.

**Deliverables (new artifacts written per run)**
- `outputs/leakage/leakage_report_{division}_{cutoff}.json` — **PASS/FAIL** per check + deltas.  
- `outputs/leakage/feature_date_audit_{division}_{cutoff}.csv` — **max_date_used** per feature and source.  
- `outputs/leakage/fold_customer_overlap_{division}_{cutoff}.csv` — any customer_ids appearing in both train & val.  
- `outputs/leakage/shift14_metrics_{division}_{cutoff}.json` — metrics for baseline vs −14d data shift.  
- `outputs/leakage/ablation_topk_{division}_{cutoff}.csv` — metrics after dropping top‑K features.  
- `outputs/leakage/static_scan_{division}_{cutoff}.json` — static code scan results.

**Acceptance thresholds**
- **Max date used** per feature/source ≤ cutoff_date.  
- **Customer overlap** between train and val = 0.  
- **−14d shift**: metrics must not improve beyond tiny noise.  
- **Static scan**: no `datetime.now()`, `pd.Timestamp.now()`, `date.today()`.  
- **Ablation sanity**: removing top‑K features should not increase metrics by more than noise.

---

## 1) Wiring: entrypoint

Create `gosales/pipeline/run_leakage_gauntlet.py` with CLI to run for division+cutoff.

---

## 2) Guardrail: time‑aware CV + group safety

Implement rolling origin CV with GroupKFold by customer.

Write overlaps to `fold_customer_overlap_{division}_{cutoff}.csv`; must be empty.

---

## 3) Feature Date Audit

Add provenance recorder in `features/engine.py`.  
Emit `feature_date_audit_{division}_{cutoff}.csv`.  
All max dates must be ≤ cutoff.  
Add static scan to forbid banned time calls.

---

## 4) −14‑day Shift Test

Shift all event dates by −14d, recompute features, retrain.  
Metrics must not improve.  
Write `shift14_metrics_{division}_{cutoff}.json`.

---

## 5) Top‑K Feature Ablation Test

Remove top‑K most important features, retrain.  
Metrics should drop or stay same, not improve.  
Write `ablation_topk_{division}_{cutoff}.csv`.

---

## 6) Consolidated Report

Collect results of all checks into `leakage_report_{division}_{cutoff}.json`.

---

## 7) Unit Tests

Add tests for each component: feature date audit, static scan, group overlap, shift14, ablation.

---

**Done criteria:** all checks pass, artifacts written, non‑zero exit code on failure.

---
