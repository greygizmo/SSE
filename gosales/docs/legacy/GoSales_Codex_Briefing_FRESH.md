
# GoSales — Fresh Briefing for Codex Agent  
**Topic:** Shift‑14 Leakage Gauntlet, time‑adjacent signal, and how to stabilize honest accuracy  
**Owner:** GoSales Engine / Division ICP & Whitespace  
**Audience:** Codex Agent (fresh session), MLOps/FE/Trainer maintainers  
**Last updated:** 2025‑09‑08

---

## TL;DR

- The **Leakage Gauntlet** is catching **time‑adjacent signal**, not literal “future data”. When we move the cutoff **earlier by 14 days**, metrics **should** get a little worse. For two divisions (Printers, Solidworks), they get **better**, which is a red flag.
- Primary causes we’ve isolated so far:  
  1) **CV mismatch** (Gauntlet Shift‑14 trained without GroupKFold by `customer_id`), and  
  2) **near‑cutoff momentum features** (short‑horizon recency/RFM/expiring signals remain too coupled to labels).  
- We hardened the Gauntlet: enforce **GroupKFold + purge/embargo**, add a **SAFE feature policy** for audits, and **tail‑mask** near‑cutoff windows. This reduced the issue but didn’t fully eliminate it for Printers; Solidworks needs a re‑run post‑patch.
- **Production ≠ Audit:** We keep **recent features in production** (they’re predictive!). The Gauntlet temporarily **lags/drops** risky families only to **prove** the model’s strength isn’t “too‑near‑future momentum.”
- Next steps for the Agent: **verify patches landed**, **re‑run Gauntlet with SAFE+GroupCV+Purge**, **emit reports**, and if still failing, **expand SAFE policy one notch** and **lift purge to 60d**. In parallel, run a **small statistical sanity pack** (label permutation, feature‑importance stability) to quantify residual adjacency.

---

## Why this exists

Models that look “suspiciously accurate” often exploit **time adjacency**: they memorize “how hot the last weeks were” and echo that back as “future intent.” That’s great for **monitoring** but can inflate **forward‑looking validation**. The Gauntlet is a **safety harness** to ensure a model trained at _T_ doesn’t get **better** when trained at _T‑14d_.

**Key observed signals (ground truth examples):**  
- **Printers** at 2024‑12‑31: AUC ~0.934 → 0.960 (+0.026), Lift@10 ~7.60 → 8.51 when shifting cutoff by ‑14d.  
- **Solidworks** at 2024‑12‑31: AUC ~0.830 → 0.940 (+0.110), Lift@10 4.49 → 6.87 on Shift‑14.  

Even after guards and tail masking, Printers still shows a **residual improvement**—smaller, but non‑trivial. The correct outcome is **flat or worse** performance when we move the cutoff earlier.

---

## Production vs Audit modes (important mindset)

- **Production mode:** Use **all honest signal** (including recent windows) to maximize lift for reps. Recent activity is valuable and allowed.
- **Audit mode (Gauntlet SAFE):** We **temporarily embargo** or **lag** adjacency‑heavy families so that Shift‑14 **must** degrade or stay flat. This is a **stress test**, not the everyday feature policy. Passing this test gives confidence that production performance isn’t riding on an accidental “peek” at near‑future dynamics.

> Bottom line: You **can** have recent features in production. You **must** prove that performance doesn’t **depend** on a 0‑to‑14‑day halo when we audit.

---

## Current repo status (what likely changed)

The prior Agent updated: `gosales/config.yaml`, `gosales/features/engine.py`, `gosales/models/train.py`, `gosales/pipeline/run_leakage_gauntlet.py`, `gosales/utils/config.py`.

**Intended net effects of those edits:**
- Gauntlet’s Shift‑14 path calls the trainer with `--group-cv --safe-mode --purge-days <N>` and passes `validation.gauntlet_mask_tail_days` to feature creation.
- Trainer recognizes `--safe-mode` and **drops/lag‑masks** adjacency‑heavy families for audits (recency/days_since_last, expiring assets, subscription composition, ≤3–6m windows, ALS embeddings).
- Trainer supports **purged/embargo GroupKFold** (drop rows with label dates within `purge_days` of validation boundary). Typical values: **45–60 days**.
- Config adds:  
  - `validation.gauntlet_mask_tail_days` (14 → tried 45),  
  - `validation.gauntlet_purge_days` (default 30 → tried 45).

---

## What the Agent must do next (resume from “ran out of context”)

1) **Sanity‑check the edits actually landed**
```powershell
git status
git diff --name-only
git diff gosales/pipeline/run_leakage_gauntlet.py
git diff gosales/models/train.py
git diff gosales/features/engine.py
git diff gosales/utils/config.py
git diff gosales/config.yaml
```
Confirm the following are present:
- Gauntlet invoking trainer with `--group-cv --safe-mode --purge-days <cfg>` in Shift‑14 flow.
- Feature engine accepts `mask_tail_days` and applies it to **all windowed** RFM/recency aggregations in audit calls.
- SAFE policy in trainer/feature builder prunes: `assets_expiring_*`, `*_subs_share_*`, `recency*`/`days_since_last_*`, windows `<= 6m`, and ALS embeddings.
- `validation.gauntlet_mask_tail_days` and `validation.gauntlet_purge_days` exist in config and are read by pipeline.

2) **Pin deterministic CV**
- Ensure a fixed `random_state` or deterministic fold assignment for GroupKFold.
- Confirm purge logic is applied **per split** and removes any training rows whose **label date** falls within `purge_days` of that split’s validation start.

3) **Re‑run the Leakage Gauntlet (Printers first)**
```powershell
$env:PYTHONPATH="$PWD"
python -m gosales.pipeline.run_leakage_gauntlet `
  --division Printers `
  --cutoff 2024-12-31 `
  --window-months 6 `
  --group-cv `
  --safe-mode `
  --purge-days 60 `
  --no-static-only `
  --run-shift14-training `
  --shift14-eps-auc 0.01 `
  --shift14-eps-lift10 0.25
```
Artifacts to inspect:
- `gosales/outputs/leakage/Printers/2024-12-31/leakage_report_Printers_2024-12-31.json`
- `gosales/outputs/leakage/Printers/2024-12-31/shift14_metrics_Printers_2024-12-31.json`

**PASS criteria:**  
`auc_shift - auc_base ≤ 0.01` **and** `lift10_shift - lift10_base ≤ 0.25`.  
Side indicators: Brier **should not** improve when we shift earlier; small worsenings are fine.

4) **If FAIL persists**
- Increase **purge** to **60d** (if not already) and **expand SAFE** one notch:
  - Drop/lag **≤ 12m** windows for adjacency‑sensitive families (division share, SKU momentum).
  - Add **division‑level recency floor** for audit: e.g., set any per‑division recency < 45d → clamp to 45 in SAFE mode only.
- Re‑run Printers. Then **repeat the same for Solidworks**.

5) **Emit a “statistical sanity pack” (quick diagnostics)**
Run these lightweight checks to quantify adjacency:
- **Label permutation within monthly buckets**: AUC should **collapse toward 0.5–0.6**. If it doesn’t, adjacency leakage remains.
- **Feature importance stability across eras** (e.g., split 2023H2, 2024H1, 2024H2): large rank churn indicates time‑specific momentum.
- **Prediction drift vs. cutoff**: compare calibration/Brier pre‑ vs post‑Shift‑14; honest models generally **worsen** slightly on earlier cutoffs.

> These diagnostics are for **evidence**. Keep the hardened Gauntlet as the **gate**; use the stats pack to aim your next SAFE expansion precisely.

---

## Minimal diffs the Agent should expect (or apply if missing)

> Use these as **shape** guides if the code doesn’t already match. Names may differ slightly.

**`gosales/pipeline/run_leakage_gauntlet.py` (Shift‑14 call path)**
```diff
- cmd = ["python", "-m", "gosales.models.train", "--division", div, "--cutoff", shift_cutoff]
+ cmd = ["python", "-m", "gosales.models.train",
+        "--division", div,
+        "--cutoff", shift_cutoff,
+        "--group-cv",
+        "--safe-mode",
+        "--purge-days", str(cfg.validation.gauntlet_purge_days)]
```

**`gosales/models/train.py` (CLI and SAFE policy)**
```diff
 parser.add_argument("--group-cv", action="store_true")
+parser.add_argument("--safe-mode", action="store_true", help="Apply SAFE feature pruning for audits")
+parser.add_argument("--purge-days", type=int, default=0, help="Days to embargo near validation")
 ...
 if args.safe_mode:
     X = drop_cols_like(X, [
         r"(^|_)recency(_|$)",
         r"(^|_)days_since_last(_|$)",
         r"^assets_expiring_",
         r"_subs_share_",
         r"(__3m$|__6m$)",
         r"(^|_)als_"
     ])
 ...
 if args.group_cv:
     splitter = GroupKFold(n_splits=k)
     for train_idx, val_idx in splitter.split(X, y, groups=customer_id):
         if args.purge_days > 0:
             val_start = cutoff_for_split(val_idx)  # derive from y/metadata
             train_idx = drop_within_purge(X, y, train_idx, val_start, args.purge_days)
         ...
```

**`gosales/features/engine.py` (tail masking on audit builds)**
```diff
 def build_windows(df, cutoff, ... , mask_tail_days=0):
     # compute window [cutoff - W, cutoff)
     win = compute_window(...)
+    if mask_tail_days and mask_tail_days > 0:
+        win = win.excluding(cutoff - timedelta(days=mask_tail_days), cutoff)
     return aggregate(win)
```

**`gosales/config.yaml`**
```yaml
validation:
  gauntlet_mask_tail_days: 45   # was 14
  gauntlet_purge_days: 60       # was 30/45; raise to 60 for December cutoffs
features:
  expiring_guard_days: 30
  recency_floor_days: 30
```

---

## Acceptance criteria and reporting

- **Shift‑14 PASS** for each division: `ΔAUC ≤ 0.01` and `ΔLift@10 ≤ 0.25` (shift − base).
- **No paradoxical gains** in Brier/logloss when shifting earlier.
- **Gauntlet report** stored under `gosales/outputs/leakage/<division>/<cutoff>/` with the following JSON keys populated:  
  - `auc_base`, `auc_shift`, `lift10_base`, `lift10_shift`, `brier_base`, `brier_shift`, plus masked LR diagnostics.
- **Audit reproducibility:** Re‑running with the same seed yields the same decision.

**Command recap (Printers):**
```powershell
$env:PYTHONPATH="$PWD"
python -m gosales.pipeline.run_leakage_gauntlet --division Printers --cutoff 2024-12-31 --window-months 6 --group-cv --safe-mode --purge-days 60 --no-static-only --run-shift14-training --shift14-eps-auc 0.01 --shift14-eps-lift10 0.25
```

**Then (Solidworks):**
```powershell
$env:PYTHONPATH="$PWD"
python -m gosales.pipeline.run_leakage_gauntlet --division Solidworks --cutoff 2024-12-31 --window-months 6 --group-cv --safe-mode --purge-days 60 --no-static-only --run-shift14-training --shift14-eps-auc 0.01 --shift14-eps-lift10 0.25
```

---

## Notes on the “Alternative Temporal Leakage Strategy”

We will **borrow its best diagnostics** (e.g., label permutation within time buckets, importance stability across eras) to **measure** residual adjacency. We do **not** replace the hardened Gauntlet or SAFE/embargo in audits because:
- The Gauntlet is a **simple, falsifiable** gate: earlier cutoff should not improve metrics.  
- The diagnostics help **aim** SAFE policy, not replace the audit gate.

If desired later, we can add a small **“Leakage Score”** dashboard (temporal correlation, permutation degradation) alongside Gauntlet results.

---

## Rollback and risk

- All SAFE behavior is **flag‑guarded** (`--safe-mode`) and **only** used in audits. Production models can keep **recent features**.
- If a division’s forward metrics degrade materially after SAFE hardening, we can **tune**: lower purge to 45d, permit 6–12m windows back into SAFE for that division, or adjust epsilons after evidence review.
- Keep commits small, one file at a time. Always re‑run the Gauntlet before moving to the next knob.

---

## Checklist for this session

- [ ] Confirm `--group-cv`, `--safe-mode`, `--purge-days` are wired from Gauntlet → Trainer.
- [ ] Confirm `mask_tail_days` plumbs from config → feature engine during audits.
- [ ] Re‑run **Printers** Gauntlet with `purge_days=60`, SAFE expanded to drop **≤12m** where adjacency‑sensitive.
- [ ] Inspect `shift14_metrics_*` JSON; record `ΔAUC`, `ΔLift@10`, `ΔBrier`.
- [ ] If FAIL, expand SAFE one more notch (division share momentum, SKU short‑term) and re‑run.
- [ ] Re‑run **Solidworks** with identical knobs.
- [ ] Generate quick **permutation** and **importance‑stability** diagnostics; attach plots/JSON to `outputs/leakage/<division>/<cutoff>/`.
- [ ] Write a 1‑page summary: PASS/FAIL per division, which SAFE knobs mattered, any trade‑offs observed.
- [ ] Hand back artifacts + summary to the team.

---

## Appendix — Quick references

**Key folders / files**
- `gosales/pipeline/run_leakage_gauntlet.py`
- `gosales/models/train.py`
- `gosales/features/engine.py`
- `gosales/utils/config.py`, `gosales/config.yaml`
- Outputs: `gosales/outputs/leakage/<division>/<cutoff>/`

**Core flags**
- `--group-cv`, `--safe-mode`, `--purge-days <int>`  
- `--run-shift14-training`, `--shift14-eps-auc 0.01`, `--shift14-eps-lift10 0.25`

**Common values**
- `validation.gauntlet_mask_tail_days`: 45–60 for December cutoffs
- `validation.gauntlet_purge_days`: 45–60 depending on label proximity to holidays or year‑end dynamics

---

*End of brief.*
