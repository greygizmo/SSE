## GoSales Engine â€” ICP & Whitespace (Phases 0â€“6)

A division-focused Ideal Customer Profile (ICP) & Whitespace engine. The pipeline ingests raw sales logs, builds a curated star schema, engineers leakage-safe features at a time cutoff, trains and calibrates per-division models, and produces scores and whitespace opportunities ready for a UI.

---

### Whatâ€™s implemented (by phase)

- **Phase 0 â€” ETL, Star Schema, Contracts**
  - Tidy `fact_transactions` and `dim_customer` with enrichment and fuzzy fallback
  - Contracts: required columns, PK checks, date-bounds; violations CSV
  - Curated Parquet + QA: schema snapshot, row counts, violations, checksums
  - CLI flags: `--config`, `--rebuild`, `--staging-only`, `--fail-soft`
- **Phase 1 â€” Labels**
  - Leakage-safe labels per `(customer, division, cutoff)` with modes: `expansion|all`
  - Cohorts (`is_new_logo`, `is_expansion`, `is_renewal_like`), censoring detection
  - Denylist SKUs and GP threshold via config; artifacts: labels parquet, prevalence CSV, cutoff JSON
- **Phase 2 â€” Features**
  - RFM windows (3/6/12/24m), trajectory (monthly slope/std), lifecycle (tenure, gaps, active months), seasonality, cross-division shares (EB-smoothed), diversity, returns
  - Optional toggles: market-basket affinity, ALS embeddings
  - Asset features at cutoff:
    - `assets_expiring_{30,60,90}d_<rollup>` and shares
    - `assets_subs_share_<rollup>` (on / (on+off)), composition shares `assets_on_subs_share_<rollup>`, `assets_off_subs_share_<rollup>`
  - Artifacts: features parquet, feature catalog CSV, feature stats JSON (coverage, winsor caps, checksum)
  - Determinism and winsorization tests
- **Phase 3 â€” Modeling**
  - Config-driven modeling grids and seeds
  - Training CLI for LR (elastic-net) and LGBM across multiple cutoffs, with calibration (Platt/Isotonic) and selection by mean lift@10 (tie-breaker Brier)
  - Metrics: AUC, PR-AUC, Brier, lift@{5,10,20}%, revenue-weighted lift@K, calibration MAE
  - Artifacts: `metrics.json`, `gains.csv`, `calibration.csv`, `thresholds.csv`, `model_card.json`, SHAP summaries (optional; guarded if SHAP not installed)
  - Guardrails: degenerate classifier check, deterministic LGBM, early stopping, overfit-gap guard, capped `scale_pos_weight`

- **Phase 4 â€” Whitespace Ranking / Nextâ€‘Bestâ€‘Action**
  - Signals: calibrated probability (`p_icp` + perâ€‘division percentile), marketâ€‘basket affinity (`mb_lift_max`, `mb_lift_mean`), ALS similarity (explicit or embeddingâ€‘centroid), expected value proxy (segmentâ€‘blended and capped)
  - Normalization: perâ€‘division percentile (default) or pooled across divisions
  - Blending: configurable weights (default `0.60,0.20,0.10,0.10`) â†’ single actionable score; weights must be four nonâ€‘negative numbers and are normalized to sum to 1; optional challenger metaâ€‘learner (`score_challenger`)
  - Capacity: topâ€‘percent, perâ€‘rep, or hybrid diversification (roundâ€‘robin interleave)
  - Gating: ownership, region, recent contact, open deal; cooldown deâ€‘emphasis; structured JSONL logs
  - Artifacts: `whitespace_<cutoff>.csv` (includes `score` and `score_challenger`), `whitespace_explanations_<cutoff>.csv`, `thresholds_whitespace_<cutoff>.csv`, `whitespace_metrics_<cutoff>.json`, `whitespace_log_<cutoff>.jsonl`, `mb_rules_<division>_<cutoff>.csv`, optional `whitespace_legacy_<cutoff>.csv` (shadow), `whitespace_overlap_<cutoff>.json`

- **Phase 5 â€” Forward Validation / Holdout**
  - CLI: `python -m gosales.validation.forward --division Solidworks --cutoff 2024-12-31 --window-months 6 --capacity-grid 5,10,20 --accounts-per-rep-grid 10,25`
  - Artifacts per division/cutoff in `gosales/outputs/validation/<division>/<cutoff>/`:
    - `validation_frame.parquet`, `gains.csv`, `calibration.csv`
    - `topk_scenarios.csv` and `topk_scenarios_sorted.csv` (with 95% CIs)
    - `segment_performance.csv` (capture/precision/rev_capture by segment)
    - `metrics.json` (AUC, PR-AUC, Brier, cal-MAE)
    - `drift.json` (per-feature PSI; KS on `p_hat` train vs holdout if snapshot exists; weighted PSI of EV vs holdout GP)
  - Phase 3 saves `train_scores_<division>_<cutoff>.csv` and `train_feature_sample_<division>_<cutoff>.parquet` to support Phase 5 drift
  - Drift snapshots: `python scripts/drift_snapshots.py` aggregates prevalence and calibration MAE over time

- **Phase 6 â€” Configuration, UX, Observability**
  - Central config precedence (YAML â†’ env â†’ CLI) and stricter validation (unknown keys rejected; sanity checks for weights/thresholds)
  - Run registry and manifests: every CLI wrapped in a run context that assigns a `run_id`, logs to `outputs/runs/<run_id>/logs.jsonl`, writes `config_resolved.yaml`, and emits a `manifest.json`; appends entries to `outputs/runs/runs.jsonl`
  - Validation enhancements:
    - Weighted PSI of EV vs holdout GP (EV deciles) in `drift.json`
    - `metrics.json` includes `drift_highlights` (top per-feature PSI flags â‰¥ threshold)
    - `alerts.json` is written when PSI/KS/calibration thresholds are breached (configâ€‘driven)
  - Streamlit UI (artifactâ€‘driven):
    - Metrics, Explainability, Whitespace, and Validation pages read CSV/JSON artifacts directly
    - Validation shows quality badges (Calibration MAE, PSI(EV vs GP), KS train vs holdout) using thresholds from config and an Alerts section if `alerts.json` exists
  - Tests: config validation checks, run registry tests, UI badge/alerts utilities, and Phaseâ€‘5 drift/calibration/scenario tests are green

---

### Quick Start (Windows/PowerShell)

```powershell
# 1) Create/activate venv and install
python -m venv .venv; .venv\Scripts\activate.ps1; pip install -r gosales/requirements.txt

# 2) Phase 0 â€” Build star schema (curated)
$env:PYTHONPATH = "$PWD"; python -m gosales.etl.build_star --config gosales/config.yaml --rebuild

# 3) Phase 1 â€” Build labels
$env:PYTHONPATH = "$PWD"; python -m gosales.pipeline.build_labels --division Solidworks --cutoff "2024-06-30" --window-months 6 --mode expansion --config gosales/config.yaml

# 4) Phase 2 â€” Build features
$env:PYTHONPATH = "$PWD"; python -m gosales.features.build --division Solidworks --cutoff "2024-06-30" --config gosales/config.yaml

# 5) Phase 3 â€” Train across cutoffs (example)
$env:PYTHONPATH = "$PWD"; python -m gosales.models.train --division Solidworks --cutoffs "2023-06-30,2023-09-30,2023-12-31" --window-months 6 --models logreg,lgbm --calibration platt,isotonic --config gosales/config.yaml

# 6) End-to-end: ingest â†’ build star â†’ audit â†’ train all divisions â†’ score/whitespace
$env:PYTHONPATH = "$PWD"; python gosales/pipeline/score_all.py

# When scoring directly, `gosales/pipeline/score_customers.py` expects each model folder
# to include `metadata.json` with `cutoff_date` and `prediction_window_months`. If those
# fields are missing, supply them via `--cutoff-date` and `--window-months` arguments or
# the scoring run will error.

# 7) Phase 4 â€” Rank whitespace (example)
$env:PYTHONPATH = "$PWD"; python -m gosales.pipeline.rank_whitespace --cutoff "2024-06-30" --window-months 6 --normalize percentile --capacity-mode top_percent --config gosales/config.yaml

# 8) Phase 5 â€” Forward validation (example)
$env:PYTHONPATH = "$PWD"; python -m gosales.validation.forward --division Solidworks --cutoff "2024-12-31" --window-months 6 --capacity-grid "5,10,20" --accounts-per-rep-grid "10,25" --bootstrap 1000 --config gosales/config.yaml

# 9) Launch Streamlit UI (Phase 6)
$env:PYTHONPATH = "$PWD"; streamlit run gosales/ui/app.py

# 10) Dry-run mode (Phase 6)
# Skips heavy compute and records planned artifacts to the run manifest/registry
$env:PYTHONPATH = "$PWD"; python -m gosales.features.build --division Solidworks --cutoff "2024-06-30" --dry-run
$env:PYTHONPATH = "$PWD"; python -m gosales.models.train --division Solidworks --cutoffs "2023-12-31" --dry-run
$env:PYTHONPATH = "$PWD"; python -m gosales.pipeline.rank_whitespace --cutoff "2024-06-30" --dry-run
$env:PYTHONPATH = "$PWD"; python -m gosales.validation.forward --division Solidworks --cutoff "2024-12-31" --dry-run
```

---

### Modeling & validation notes

- Class imbalance: class weights (LR) and capped `scale_pos_weight` (LGBM)
- Selection: mean lift@10 across cutoffs (tie-breaker: mean Brier). Revenue-weighted lift@K reported for business impact
- Calibration: bins CSV and weighted MAE exported
- Key artifacts (per division) in `gosales/outputs/`:
  - `metrics_<division>.json`
  - `gains_<division>.csv`
  - `calibration_<division>.csv`
  - `thresholds_<division>.csv`
  - `model_card_<division>.json`
  - `shap_global_<division>.csv` and `shap_sample_<division>.csv` (if SHAP available)
- Guardrails: degenerate classifier abort, deterministic LightGBM, overfit-gap guard (auto-regularization), early stopping

### Artifacts glossary

| File | Description | Produced by |
| - | - | - |
| `gosales/outputs/metrics_<division>.json` | Metrics summary: AUC, PR-AUC, Brier, lift@K, revenue-weighted lift@K, calibration MAE, selection, cutoffs, window_months, aggregates | `gosales/models/train.py` (final model)
| `gosales/outputs/gains_<division>.csv` | Decile gains with `bought_in_division_mean`, `count`, `p_mean` | `gosales/models/train.py`
| `gosales/outputs/calibration_<division>.csv` | Calibration bins with `mean_predicted`, `fraction_positives`, `count` | `gosales/models/train.py`
| `gosales/outputs/thresholds_<division>.csv` | Thresholds for top-K%: `k_percent`, `threshold`, `count` | `gosales/models/train.py`
| `gosales/outputs/model_card_<division>.json` | Model card: division, cutoffs, window_months, selected_model, seed, params (lr_grid, lgbm_grid), data (n_customers, prevalence), calibration (MAE), artifact paths | `gosales/models/train.py`
| `gosales/outputs/shap_global_<division>.csv` | Global mean-abs SHAP by feature (if SHAP available) | `gosales/models/train.py`
| `gosales/outputs/shap_sample_<division>.csv` | Sample SHAP rows with `customer_id` (if SHAP available) | `gosales/models/train.py`
| `gosales/models/<division>_model/model.pkl` | Pickled calibrated classifier | `gosales/models/train.py`
| `gosales/models/<division>_model/feature_list.json` | Ordered list of features used | `gosales/models/train.py`
| `gosales/outputs/coef_<division>.csv` | Logistic Regression coefficients (if LR selected) | `gosales/models/train.py`
| `gosales/outputs/whitespace_<cutoff>.csv` | Ranked opportunities with `customer_id, division, score, p_icp, p_icp_pct, lift_norm, als_norm, EV_norm, nba_reason` | `gosales/pipeline/rank_whitespace.py`
| `gosales/outputs/whitespace_explanations_<cutoff>.csv` | Explanations with key drivers for each candidate | `gosales/pipeline/rank_whitespace.py`
| `gosales/outputs/thresholds_whitespace_<cutoff>.csv` | Capacity thresholds (topâ€‘percent/perâ€‘rep/hybrid) | `gosales/pipeline/rank_whitespace.py`
| `gosales/outputs/whitespace_metrics_<cutoff>.json` | Capture@K, division shares, stability, coverage, weights | `gosales/pipeline/rank_whitespace.py`
| `gosales/outputs/whitespace_log_<cutoff>.jsonl` | Structured JSONL logs (division summaries, selection summary) | `gosales/pipeline/rank_whitespace.py`
| `gosales/outputs/validation/<division>/<cutoff>/metrics.json` | AUC, PR-AUC, Brier, cal-MAE, capture grid, drift, drift_highlights | `gosales/validation/forward.py`
| `gosales/outputs/validation/<division>/<cutoff>/drift.json` | Perâ€‘feature PSI, KS(p_hat train vs holdout), weighted PSI(EV vs holdout GP), EV_norm PSI | `gosales/validation/forward.py`
| `gosales/outputs/validation/<division>/<cutoff>/alerts.json` | Alerts when thresholds breached: type, value, threshold | `gosales/validation/forward.py`
| `gosales/outputs/runs/<run_id>/manifest.json` | Files emitted during a run (paths) | Any CLI via `gosales/ops/run.py`
| `gosales/outputs/runs/runs.jsonl` | Run registry with metadata and statuses | `gosales/ops/run.py`
| `gosales/outputs/mb_rules_<division>_<cutoff>.csv` | Marketâ€‘basket rule table with SKU lift and support | `gosales/features/engine.py`
| `gosales/outputs/features_<division>_<cutoff>.parquet` | Feature matrix snapshot | `gosales/features/build.py`
| `gosales/outputs/feature_catalog_<division>_<cutoff>.csv` | Feature names and coverage | `gosales/features/build.py`
| `gosales/outputs/feature_stats_<division>_<cutoff>.json` | Coverage, winsor caps, checksum | `gosales/features/build.py`
| `gosales/outputs/labels_<division>_<cutoff>.parquet` | Labels per (customer, division, cutoff) | `gosales/pipeline/build_labels.py`
| `gosales/outputs/prevalence_<division>.csv` | Label prevalence summary | `gosales/pipeline/build_labels.py`
| `gosales/outputs/cutoffs_<division>.json` | Cutoff metadata summary | `gosales/pipeline/build_labels.py`
| `gosales/outputs/schema_icp_scores.json` | Schema validation report for `icp_scores.csv` | `gosales/pipeline/score_customers.py`
| `gosales/outputs/schema_whitespace*.json` | Schema validation report for whitespace outputs | `gosales/pipeline/score_customers.py`

---

### Feature Library (Phase 2 highlights)

- Recency, frequency, monetary with 3/6/12/24m windows
- Temporal dynamics (monthly slope/std), lifecycle (tenure, gaps, active months)
- Seasonality (quarter shares), cross-division mix (EB-smoothed), diversity, returns
- Optional: market-basket affinity and ALS embeddings
- Artifacts: `feature_catalog_<division>_<cutoff>.csv` lists feature names and coverage

---

### Tests (high level)

- Phase 0/1/2: parsers, keys, contracts, labels, features (winsorization + determinism)
- Phase 3:
  - Threshold math correctness for top-N
  - Calibration bins + weighted MAE sanity on synthetic logits
  - Determinism for calibrated LR with fixed seed
  - Leakage probe and guard (name- and AUC-based feature dropping)
- Phase 4:
  - Normalization sanity and pooled/perâ€‘division behavior
  - Weight scaling by coverage (ALS/affinity) and monotonic affinity
  - ALS fallback similarity via embedding centroid
  - Bias share computation and diversification capacity options
  - Deterministic ranking order; Capture@K sanity

---

### How to interpret whitespace metrics (for revenue teams)

- **Capture@K**
  - What it means: Of all the wins the model expects in this period, how many are covered if you work just the top K% of the ranked list.
  - Why you care: Tells you â€œhow concentrated the opportunity is.â€ A higher number means most wins are in a small slice at the top, so focusing there is efficient.
  - Rule of thumb: If `capture@10% = 0.65`, then working the top 10% could capture ~65% of expected wins.

- **Division shares in the selected list**
  - What it means: Within your capacity slice (e.g., top 10% or perâ€‘rep list), what fraction comes from each division.
  - Why you care: Prevents overâ€‘concentration (e.g., 90% Solidworks). If one division exceeds the configured share threshold, we warn and suggest using the hybrid capacity mode to diversify.

- **Stability (Jaccard) vs last run**
  - What it means: The overlap between this runâ€™s topâ€‘N and the previous runâ€™s topâ€‘N (0 to 1).
  - Why you care: High stability (~0.7â€“0.9) = consistent targeting; low (~0.3) = shift due to seasonality, new data, or configuration change.
  - Action: If stability drops unexpectedly, review data recency, config changes, and business events.

- **Coverage and weight adjustments**
  - What it shows: Coverage for ALS and marketâ€‘basket signals (what % of customers have these signals). If coverage is sparse, the system automatically downâ€‘weights that signal and renormalizes.
  - Why you care: Low coverage is not â€œbadâ€ but indicates limited signal today. Over time, as data coverage improves, those signals will earn more weight.

- **Thresholds & capacity**
  - Topâ€‘percent: Work the top X% across all divisions. The thresholds file lists the score cut line and counts.
  - Perâ€‘rep: Each rep gets roughly the top N in their book (requires a `rep` column in features). Encourages fair distribution.
  - Hybrid: Roundâ€‘robin across divisions up to capacity to ensure diversification.

- **Probability and value**
  - `p_icp`: A calibrated probability (0â€“1) of a positive outcome in the prediction window. Higher is better.
  - `EV_norm`: A normalized expectedâ€‘value proxy (based on segment medians and capped at a high percentile to avoid â€œwhalesâ€ dominating).
  - Why we cap EV: Protects against extreme outliers skewing the list; ensures we donâ€™t ignore highâ€‘probability, modestâ€‘value wins that are easier to capture at scale.

- **Explanations (`nba_reason`)**
  - Short, humanâ€‘readable context such as â€œHigh p=0.78; strong affinity; high EV.â€
  - These are helpful hints, not full audit trails. For deeper model insights, use the Phaseâ€‘3 SHAP/coef artifacts.

- **Recommended operating mode**
  - For pilots: pick one capacity mode (e.g., topâ€‘10% or perâ€‘rep 25) and run for 4â€“6 weeks. Track conversion vs a comparable control group.
  - Review weekly: capture@K, division shares, stability, and qualitative feedback from reps. Adjust capacity and weights if needed.

- **Cautions**
  - This is not a price quote or guarantee; itâ€™s a ranked opportunity list.
  - If gating (DNC/legal/open deals/region) removes many accounts, the selected list may shrink and division shares may shiftâ€”this is expected.
  - Cooldowns reduce repeated surfacing of the same account that wasnâ€™t actioned; adjust in config if needed.

---

### Multi-division support

Known divisions are sourced from `etl/sku_map.division_set()`; cross-division features adapt automatically. The `score_all` pipeline trains/audits for every known division, and scores any division with an available model. To add a division, update `etl/sku_map.py` (or overrides CSV), rebuild star and features, then run `score_all` or train explicitly via `gosales/models/train.py`.

#### Troubleshooting
- If a division has very low prevalence, widen the window or aggregate sub-divisions; inspect `labels_summary.csv` and `labels_positive_<division>.csv` for prevalence.
- Auto-widening respects SKU-targeted models (e.g., Printers) so only the intended sale SKUs count toward positives even when the window extends.
- If simple training fails with numeric issues (inf/NaN), use the robust trainer `gosales/models/train.py` which sanitizes features and searches hyperparameters across cutoffs.


### Utilities / Scripts

- `scripts/metrics_summary.py`: aggregate metrics_*.json into a summary CSV
- `scripts/ci_assets_sanity.py`: CI-style gate for asset rollup coverage and tenure imputation sanity
- `scripts/drift_snapshots.py`: aggregate prevalence and calibration MAE across validation runs
- `scripts/name_join_qa.py`: Moneyball→dim_customer name-join QA; writes coverage summary and top unmapped names
- `scripts/ablation_assets_off.py`: Train with assets disabled and compare metrics vs baseline; writes ablation JSON
- `scripts/build_features_for_models.py`: Build feature matrices for each trained model's cutoff to align feature lists
- `scripts/train_all_models.py`: Retrain all target models at a given cutoff (group-CV, calibration)

### SQL Templates

- Centralized SQL query helpers live under `gosales/sql/queries.py`. These functions validate identifiers and build parameterized, reusable SELECTs used by ETL and ops.

### Environment Overrides

- `GOSALES_FEATURES_USE_ASSETS=0|1`: Force-enable or disable asset features at runtime (used by ablation script).

### Training Playbook

- Recommended training cutoff: `2024-06-30`.
  - Rationale: train on data through 1H 2024; use 2H 2024 as internal test and 2025 as forward/holdout validation.
- Retrain all models (with GroupKFold on customer_id and Platt/Isotonic calibration):
  - PowerShell: `$env:PYTHONPATH=$PWD; python scripts/train_all_models.py --cutoff 2024-06-30 --window-months 6`
- After training, refresh artifacts and checks:
  - `python scripts/build_features_for_models.py` (align feature catalogs for each model cutoff)
  - `python scripts/ci_featurelist_alignment.py`
  - `python -m gosales.pipeline.run_leakage_gauntlet --division <Div> --cutoff 2024-12-31 --no-static-only`
