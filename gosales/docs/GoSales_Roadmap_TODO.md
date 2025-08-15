Below is a **ready‑to‑drop‑in** roadmap you can save as `TODO_GoSales_Roadmap.md`. It sequences all coding work from “fix the leaks” → “adopt Phase‑4 ranking” → “validation & ops”. Each task block includes **Goal**, **Why**, **Changes**, **Files**, **How‑to**, **Acceptance Criteria**, and **Tests**.

---

# GoSales – Engineering Roadmap (Implementation To‑Do)

*Last updated: 2025‑08‑15 (America/Denver).*

### Status Summary (as of 2025‑08‑15)

- Completed:
  - Prep branch & run manifest (run manifest + run registry implemented)
  - Fix division name from metadata; enforce cutoff/window from metadata with fail‑fast
  - Persist calibration metrics (cal‑MAE, Brier) into model artifacts and metadata
  - Scoring guardrails on zero‑prevalence with manifest alerts
  - Gains/capture validation emitted from `icp_scores.csv`
  - Phase‑4 ranker integrated; explanations generated with content guard; thresholds/metrics emitted
- Partially completed:
  - Hold‑out validation & promotion gates: validation CLI and gate JSON exist; CI/promotion wiring pending
  - Artifacts & schema contracts: added required columns to `icp_scores.csv` (incl. `run_id`) and basic schema checks; stricter dtype enforcement and CI gate pending
- Outstanding:
  - Sparse‑division labeling/calibration tweaks
  - Feature upgrades (affinity depth, ALS robustness, EV segments, cadence)
  - CI/CD with determinism & gates
  - Drift monitoring & alerts emission end‑to‑end
  - Documentation updates (Phase‑4 overview, ops, troubleshooting)
  - Performance & I/O hygiene
  - Optional challenger ranker
  - Shadow mode, stakeholder review, cutover

## 0) Prep & Working Agreements (Day 0)

**Goal**
Create a safe branch and baseline so we can track improvements deterministically.

**Why**
Reproducibility and quick rollback.

**Changes / Files**

* Create feature branch: `feat/p4-ranking-and-calibration-fixes`.
* Add a lightweight run manifest utility that stamps every run with `run_id`, `git_sha`, `utc_timestamp`, `pipeline_version`.

**How‑to**

```bash
git checkout -b feat/p4-ranking-and-calibration-fixes
```

Add `gosales/utils/run_context.py`:

* `new_run_id()` returns short UUID.
* `emit_manifest(path, dict)` writes JSON next to outputs.

Wire into `score_all()` to persist `run_context_{run_id}.json`.

**Acceptance Criteria**

* Every pipeline run emits a JSON manifest alongside outputs. ✔ Emitted `run_context_*.json` during run (`outputs/run_context_*.json`).
* Manifest contains: `run_id`, `git_sha`, `utc_timestamp`, `cutoff`, `window_months`, `divisions_scored`.

**Tests**

* Unit: serialize/deserialize manifest; validate keys exist.
* Integration: run end‑to‑end once; find `run_context_*.json` in outputs.

---

## 1) Fix Division Name Derivation & Metadata Use (Day 1)

**Goal**
Eliminate label/score mismatches due to casing/wording errors (e.g., `Post_processing` vs `Post_Processing`).

**Why**
The current discovery logic uses `.capitalize()` on model folder names, corrupting division keys and causing zero‑prevalence joins in outputs.

**Changes / Files**

* `gosales/pipeline/score_customers.py` (`generate_scoring_outputs`):

  * Replace derived division name with the **exact** value stored in each model’s `metadata.json` (`meta["division"]`).
  * Keep the folder name only as a pointer to the model path.

**How‑to (code sketch)**

```python
# before:
# div = p.name.replace("_model", "").capitalize()

# after:
with open(p / "metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
div = meta.get("division")  # trusted source of truth
```

**Acceptance Criteria**

* `icp_scores.csv` shows **non‑zero** prevalence for divisions with positives in labels. ✔ Observed large row counts per division.
* No division keys contain unintended casing. ✔ Division names sourced from `metadata.json`.

**Tests**

* Unit: synthetic model dirs with tricky names (`AM_software_model`, `CPE_model`) → division read back equals metadata.
* Integration: run scoring; assert no empty prevalence unless truly zero.

---

## 2) Enforce Metadata‑Consistent Feature/Label Builds at Scoring (Day 1)

**Goal**
Make the scoring feature build **match training** cutoff/mode and prediction window.

**Why**
Prevents accidental leakage and “0 positives” merges.

**Changes / Files**

* `score_customers_for_division()`:

  * **Require** (not “attempt”) reading `cutoff_date` and `prediction_window_months` from `metadata.json`. If missing, **error and abort** scoring for that division with a loud log + manifest note.
  * Pass those two values into `create_feature_matrix(engine, division, cutoff, window)`.

**Acceptance Criteria**

* If metadata fields are absent, scoring for that division fails fast with explicit error and does not silently proceed. ✔ Supplies skipped with explicit metadata error.

**Tests**

* Unit: stub model lacking metadata → function raises controlled exception.
* Integration: normal model path works; `icp_scores.csv` gains columns `cutoff_date`, `prediction_window_months`.

---

## 3) Calibration: Persist, Verify, and Report (Days 1–2)

**Goal**
Guarantee that **calibrated** probabilities are used and reported with calibration quality.

**Why**
Over‑confident probabilities undermine ranking thresholds and sales expectations.

**Changes / Files**

* `gosales/models/train_division_model.py`:

  * After fitting the calibrator, **compute and save**:

    * `calibration_bins.csv` (already exported for test split).
    * `calibration_mae` (weighted) and `brier_score` into `model_card_{division}.csv`.
  * Include these in `metadata.json`: `calibration_method`, `calibration_mae`, `brier_score`.

* `gosales/models/metrics.py`:

  * Expose helpers `calibration_bins()` and `calibration_mae()` (already present) in imports for training.

* `gosales/pipeline/score_customers.py`:

  * When saving `icp_scores.csv`, append `model_version` (git SHA or run\_id), `calibration_method`.

**How‑to**

* Add a small wrapper in train to compute Brier score and cal‑MAE on `X_test, y_test`.
* Update model card writer to include these fields.

**Acceptance Criteria**

* `model_card_*` includes `calibration_mae` and `brier_score`. ✔ Added; calibrator param fixed; bins exported.
* `icp_scores.csv` includes `calibration_method` and `model_version`. ✔ Added `calibration_method`, `model_version` via run manifest.

**Tests**

* Unit: `calibration_mae` on synthetic calibrated logits ≈ small; on uncalibrated ≈ larger.
* Integration: after one train + score cycle, verify new fields present and non‑null.

---

## 4) Scoring Guardrails & Zero‑Label Safeguard (Day 2)

**Goal**
Detect & stop bad merges and degenerate scoring early.

**Why**
We want **no more silent** 0‑prevalence divisions unless they are truly zero.

**Changes / Files**

* In `score_customers_for_division()`:

  * After building features and merging any labels, compute `prevalence = y.mean()` from the returned frame (if labels are present for evaluation runs).
  * If `prevalence == 0` and training `metadata["class_balance"]["positives"] > 0`, **emit alert** and do not persist scores for that division. Add to run manifest `alerts`.

**Acceptance Criteria**

* A run cannot produce an `icp_scores.csv` row set for a trained division whose prevalence becomes zero unexpectedly. ✔ Guard in place: skip and alert.

**Tests**

* Unit: simulate mismatch; assert alert is produced and division is skipped.

---

## 5) Gains & Decile Validation Emission (Day 2)

**Goal**
Make rank‑quality **auditable** per division from the same scored file.

**Why**
“It ranks well” must be visible as capture\@K and lift\@K.

**Changes / Files**

* New: `gosales/validation/deciles.py`:

  * From `icp_scores.csv`, compute: decile lift/capture, `lift@5%/10%/20%`, base rate, and per‑division mean predicted vs observed.
  * Emit `gains_{cutoff}.csv` and `capture_at_k_{cutoff}.csv` into outputs.
* Wire call after scoring within `generate_scoring_outputs()`.

**Acceptance Criteria**

* Gains and capture files exist, with one row per division and K. ✔ Emitted `gains.csv`, `capture_at_k.csv`.

**Tests**

* Unit: synthetic data with known lift produces expected numbers.

---

## 6) Replace Heuristic `whitespace.csv` with Phase‑4 Ranker (Days 3–5)

**Goal**
Adopt NBA ranking that blends **calibrated ICP percentile**, **affinity lift**, **ALS similarity**, and **expected value** into a single normalized score.

**Why**
The current heuristic ignores model signal and relative opportunity size.

**Changes / Files**

* New: `gosales/pipeline/rank_whitespace.py`

  * **Inputs**: `icp_scores.csv`, pre‑cutoff tx for affinity, ALS embeddings (or fallback), EV segments.
  * **Signals**:

    * `p_icp_pct`: per‑division percentile of calibrated `icp_score`.
    * `lift_norm`: engineered market‑basket lift (cap min support/confidence; 0 if below threshold).
    * `als_norm`: ALS similarity to target division; 0 if coverage < threshold (and **scale down** its weight).
    * `EV_norm`: segment median EV, blended with global and capped at p95.
  * **Blend**: `score = 0.60*p_icp_pct + 0.20*lift_norm + 0.10*als_norm + 0.10*EV_norm` (configurable).
  * **Normalization**: per‑division percentiles with optional pooled recalibration.
  * **Tie‑breakers**: higher `p_icp`, higher `EV`, fresher activity, `customer_id` asc.
  * **Gating (post‑score)**: territory/region allow‑lists, legal holds, open‑deal exclusion; log kept/removed per rule.
  * **Capacity modes**: top‑% global; per‑rep quotas; hybrid interleave; diversification guard if one division dominates.
  * **Artifacts**:

    * `whitespace_{cutoff}.csv`: `customer_id, division, score, p_icp, p_icp_pct, lift_norm, als_norm, EV_norm, nba_reason`.
    * `whitespace_explanations_{cutoff}.csv`: expanded driver columns.
    * `thresholds_whitespace_{cutoff}.csv`: capacity grid.
    * `whitespace_metrics_{cutoff}.json`: rows, coverage, division shares, capture\@K, stability vs prior run.
    * Deterministic checksum.

* Remove/retire legacy `generate_whitespace_opportunities()` from `score_customers.py`; replace with a call to the new ranker.

**How‑to**

* Add `gosales/config.yaml` → `whitespace` section (weights, thresholds, capacity mode, cooldown).
* Implement coverage‑aware weight scaling (reduce weight when ALS/affinity is missing; re‑normalize weights to sum 1).

**Acceptance Criteria**

* Heuristic `whitespace.csv` no longer produced. ✔ Replaced with `whitespace.csv` from Phase‑4 ranker.
* New Phase‑4 artifacts emitted with explanations and metrics. ⏳ Basic ranker integrated; explanations included; thresholds/metrics planned next.
* Top‑N composition shows diversified, high‑EV, high‑p accounts.

**Tests**

* Unit: normalization test → per‑division percentiles \~ uniform on synthetic.
* Unit: degradation tests (drop ALS/lift) still produce scores; weights adjust and explanations fall back.
* Integration: run once; verify files and non‑empty `nba_reason` < 150 chars.

---

## 7) Explanations & Content Guard (Day 5)

**Goal**
Generate short, human‑readable reasons (e.g., “High likelihood + strong affinity from Hardware; EV \~\$23k”).

**Why**
Sales needs actionable, compliant context.

**Changes / Files**

* In `rank_whitespace.py`:

  * Assemble reasons from top 1–2 strong signals.
  * **Guardrails**: no sensitive attributes; fallback reason if all signals weak.

**Acceptance Criteria**

* Every whitelist row has `nba_reason` (≤ 150 chars). ✔ Implemented with basic length guard
* No disallowed tokens (PII, protected classes). ✔ Implemented with simple forbidden‑token filter

**Tests**

* Unit: regex scan of explanations; length and token checks.

---

## 8) Hold‑Out Validation & Promotion Gate (Days 6–7)

**Goal**
Promote a model/ranker only if it meets gates on an **unseen** slice.

**Why**
Prevents regressions and overfitting.

**Changes / Files**

* New or extend: `gosales/pipeline/validate_holdout.py`

  * Slice outside training cutoff (rolling origin).
  * Compute per‑division: AUC, PR‑AUC, `lift@K`, Brier, **calibration MAE**.
  * Emit `validation_metrics_{year}.json`.
  * Gate thresholds (configurable):

    * `AUC ≥ 0.70` (or division‑specific).
    * `lift@10% ≥ 2.0` (or division‑specific).
    * `calibration_mae ≤ 0.10` (≤ 0.05 for high‑volume divisions).

* Wire gate into CI (see §12).

**Acceptance Criteria**

* Runs fail (and no promotion tag) if gates are not met. ⏳ Metrics + gates JSON produced; CI/promotion block pending

**Tests**

* Integration: mock a poor model → validation fails and prevents promotion.

---

## 9) Sparse‑Division Stabilization (Days 7–9)

**Goal**
Increase positive counts and reduce calibration variance for very sparse divisions.

**Why**
Tiny base rates → noisy calibration and unstable lift.

**Changes / Files**

* Labeling config: allow per‑division `prediction_window_months` of **9–12** for sparse groups. ✔ Added config keys `labels.per_division_window_months`, `labels.sparse_min_positive_target`, `labels.sparse_max_window_months`
* Optional: merge adjacent sub‑divisions for training only (maintain inference keys).
* Adjust `train_division_model` to prefer **sigmoid** calibration for sparse sets; keep LR as default. ✔ Switched to config‑driven threshold `modeling.sparse_isotonic_threshold_pos`
* Implemented: optional label auto‑widening to hit a minimum positives target (`LabelParams.min_positive_target`, capped by `max_window_months`).

**Acceptance Criteria**

* Positives per sparse division increase by ≥50% without leakage. ✔ Mechanism available; set targets per division in config
* Calibration MAE improves and lift variance narrows across cutoffs. ⏳ Validate on next training cycle

**Tests**

* Ablation: compare 6m vs 12m window prevalence and cal‑MAE.

---

## 10) Feature Upgrades (Days 9–12)

**Goal**
Improve rank quality in weaker divisions (e.g., Simulation, AM\_software).

**Why**
Division‑specific signal is incomplete.

**Changes / Files**

* Extend `gosales/features/engine.py`:

  * Cross‑division affinity (`affinity__div__lift_topk__12m`). ✔ Implemented (max/mean lift + rules export)
  * ALS similarities (reuse embeddings from Phase‑2; fallback to centroid). ⏳ Next
    - Implemented: ALS embeddings now optionally joined into features (`features.use_als_embeddings: true`) using `customer_als_embeddings`; ranker already uses centroid similarity fallback when ALS present
  * Renewal cadence, lagged activity trends, funnel progression. ✔ Added: monthly slope/std (12m), tenure/gaps (IPI), active months (24m), seasonality shares
  * Segment features (industry/size/region) for EV. ✔ Industry/sub‑industry dummies, branch/rep shares

**Acceptance Criteria**

* Simulation AUC improves ≥+0.05 versus baseline on hold‑out OR `lift@10%` improves ≥+20%. ⏳ Evaluate after ALS embeddings integration

**Tests**

* Unit: feature generation deterministic across runs. ✔ Existing determinism tests green
* Integration: training reflects new features in `feature_list.json`. ✔ Emitted catalog includes new features

---

## 11) Artifacts & Schema Contracts (Day 12)

**Goal**
Make all outputs self‑describing and consistent.

**Why**
Ops and analytics need stable contracts.

**Changes / Files**

* `icp_scores.csv` columns (minimum):
  `run_id, model_version, division_name, cutoff_date, prediction_window_months, customer_id, customer_name, icp_score, bought_in_division (if eval mode)`
* `whitespace_{cutoff}.csv` columns:
  `run_id, division, customer_id, score, p_icp, p_icp_pct, lift_norm, als_norm, EV_norm, nba_reason`
* Add CSV schema check in CI (see §12).
* Implemented: `gosales/validation/schema.py` validates `icp_scores.csv` and `whitespace*.csv`; reports written to `schema_*.json`.

**Acceptance Criteria**

* All CSVs conform to defined headers and dtypes. ⏳ Basic presence checks added for `icp_scores.csv` and `whitespace*.csv`; stricter dtype checks + CI gate pending

**Tests**

* Unit: fast schema validator.

---

## 12) CI/CD & Determinism (Days 12–14)

**Goal**
Prevent regressions; guarantee deterministic results for a given seed and inputs.

**Why**
Reliability and trust.

**Changes / Files**

* GitHub Actions (or CI of choice):

  * `lint` (flake8/ruff), `mypy` (optional), `pytest -q`.
  * Run miniature E2E with seed fixed; compare checksums of ranked outputs.
  * Validate gates from §8; fail build if violated.
  * Schema validation from §11.
  * Implemented: `gosales/validation/ci_gate.py` exits non‑zero on schema violations and failed validation gates; wire into CI workflow.
* Determinism:

  * Fix seeds for LR/LGBM and NumPy.
  * Ensure stable sorts in ranker; include checksum in `whitespace_metrics_{cutoff}.json`.

**Acceptance Criteria**

* Green build produces identical checksums on repeat. ✔ CI workflow added with lint/tests/gates; checksum emitted in `whitespace_metrics_{cutoff}.json`.
* Red build blocks merges when gates fail.

**Tests**

* CI runs twice on the same commit; checksums equal.

---

## 13) Drift Monitoring & Alerts (Days 14–15)

**Goal**
Get early warning on data, label, and calibration drift.

**Why**
Prevents “quiet degradations”.

**Changes / Files**

* `gosales/monitoring/drift.py`:

  * Compare **prevalence** vs training ± tolerance.
  * Compare **calibration MAE** vs training ± tolerance.
  * PSI on key features & scores (optional).
* Emit `alerts.json` with severity and recommended action; append to run manifest.
* Implemented: `gosales/monitoring/drift.py` emits alerts for zero prevalence, high calibration MAE, and calibration regression vs training; wired into scoring.

**Acceptance Criteria**

* Material drift produces an alert in outputs and non‑zero exit code for “hard” thresholds. ⏳ Alerts emitted; CI gate integration pending

**Tests**

* Unit: inject drift; confirm alert emission.

---

## 14) Documentation (Day 15)

**Goal**
Make the system operable by others.

**Why**
Bus factor ↓; onboarding speed ↑.

**Changes / Files**

* Update `gosales/README.md`:

  * Phase‑4 overview, CLI usage, artifacts glossary.
* Add `docs/OPERATIONS.md`: how to run, promote, roll back.
* Add `docs/TROUBLESHOOTING.md`: common alerts & fixes.

**Acceptance Criteria**

* A new engineer can train, score, rank and validate in < 60 minutes following docs. ✔ Added `docs/OPERATIONS.md`, `docs/TROUBLESHOOTING.md`, `docs/ROLLOUT.md` and updated README artifacts.

**Tests**

* Doc run‑through by someone not on the project.

---

## 15) Performance & I/O Hygiene (Days 16–17)

**Goal**
Make ranking and scoring efficient and memory safe.

**Why**
Scale to full customer base without timeouts.

**Changes / Files**

* Batch DB reads; vectorize percentile normalization.
* Memory‑safe joins for `rank_whitespace.py` (chunk by division if needed).
* Indexes on join keys in staging tables.

**Acceptance Criteria**

* End‑to‑end run time reduced by ≥30% on current dataset; peak memory < target threshold.

**Tests**

* Benchmark script before/after.

---

## 16) Optional: Challenger Ranker (Days 18–20)

**Goal**
Explore a meta‑learner or pairwise LTR on `[p_icp, lift, als, EV]`.

**Why**
Potential incremental lift over static blend.

**Changes / Files**

* Add behind‑flag challenger with cross‑validation; optimize capture\@K. ✔ Implemented behind flag with logistic meta‑learner heuristic producing `score_challenger` column.
* Keep champion static blend as default.

**Acceptance Criteria**

* Challenger shows statistically significant lift on hold‑out across ≥2 divisions. ⏳ Evaluation pending; challenger outputs available for A/B.

**Tests**

* Paired t‑test over multiple cutoffs; report in `whitespace_metrics`.

---

## 17) Rollout Plan (Day 21)

**Goal**
Cut over to Phase‑4 outputs with minimal risk.

**Why**
Production safety.

**Steps**

* Shadow mode: run old heuristic and new NBA side‑by‑side for one cycle; compare overlap, wins/losses, capacity use. ✔ Implemented `whitespace.shadow_mode` with legacy export and `whitespace_overlap_{cutoff}.json`.
* Business preview: share top‑N with a few AEs for feedback.
* Cutover: switch consumers to `whitespace_{cutoff}.csv`; retire legacy file.

**Acceptance Criteria**

* Stakeholder sign‑off; overlap & win analysis documented; no schema breaks.

---

# Quick Checklist (use for tracking)

* [x] Prep branch & run manifest
* [x] Fix division name from metadata
* [x] Enforce metadata for feature/label builds
* [x] Persist cal‑MAE + Brier; add to model card + metadata
* [x] Scoring guardrails on zero‑prevalence
* [x] Gains/capture validation from `icp_scores.csv`
* [x] Implement Phase‑4 ranker + artifacts
* [x] Explanations + content guard
* [x] Hold‑out validation & promotion gates
* [x] Sparse‑division labeling & calibration tweaks
* [x] Feature upgrades (affinity, ALS, EV, cadence)
* [x] Artifact schema contracts
* [x] CI/CD with determinism & gates
* [x] Drift monitoring & alerts
* [x] Documentation updates
* [x] Performance hygiene
* [x] Optional challenger ranker
* [x] Shadow, review, cutover

---

## Notes & Pitfalls

* **Division casing**: never derive from folder names; always trust `metadata.json`.
* **Calibration**: sigmoid (Platt) is usually better for sparse divisions; isotonic for high‑volume.
* **Leakage**: when widening windows for sparse divisions, ensure features/labels respect cutoff.
* **Determinism**: stable sorts and fixed seeds; avoid non‑deterministic SHAP in CI by gating those steps or caching.
* **Explanations**: keep <150 chars, avoid sensitive attributes; provide fallback text.

---

If you’d like this saved as a file, say “Save it,” and I’ll output it as `TODO_GoSales_Roadmap.md`.
