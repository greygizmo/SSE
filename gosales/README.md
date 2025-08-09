## GoSales Engine — ICP & Whitespace (Phases 0–4)

A division-focused Ideal Customer Profile (ICP) & Whitespace engine. The pipeline ingests raw sales logs, builds a curated star schema, engineers leakage-safe features at a time cutoff, trains and calibrates per-division models, and produces scores and whitespace opportunities ready for a UI.

---

### What’s implemented (by phase)

- **Phase 0 — ETL, Star Schema, Contracts**
  - Tidy `fact_transactions` and `dim_customer` with enrichment and fuzzy fallback
  - Contracts: required columns, PK checks, date-bounds; violations CSV
  - Curated Parquet + QA: schema snapshot, row counts, violations, checksums
  - CLI flags: `--config`, `--rebuild`, `--staging-only`, `--fail-soft`
- **Phase 1 — Labels**
  - Leakage-safe labels per `(customer, division, cutoff)` with modes: `expansion|all`
  - Cohorts (`is_new_logo`, `is_expansion`, `is_renewal_like`), censoring detection
  - Denylist SKUs and GP threshold via config; artifacts: labels parquet, prevalence CSV, cutoff JSON
- **Phase 2 — Features**
  - RFM windows (3/6/12/24m), trajectory (monthly slope/std), lifecycle (tenure, gaps, active months), seasonality, cross-division shares (EB-smoothed), diversity, returns
  - Optional toggles: market-basket affinity, ALS embeddings
  - Artifacts: features parquet, feature catalog CSV, feature stats JSON (coverage, winsor caps, checksum)
  - Determinism and winsorization tests
- **Phase 3 — Modeling**
  - Config-driven modeling grids and seeds
  - Training CLI for LR (elastic-net) and LGBM across multiple cutoffs, with calibration (Platt/Isotonic) and selection by mean lift@10 (tie-breaker Brier)
  - Metrics: AUC, PR-AUC, Brier, lift@{5,10,20}%, revenue-weighted lift@K, calibration MAE
  - Artifacts: `metrics.json`, `gains.csv`, `calibration.csv`, `thresholds.csv`, `model_card.json`, SHAP summaries (guarded if SHAP not installed)
  - Guardrails: degenerate classifier check, deterministic LGBM, early stopping, overfit-gap guard, capped `scale_pos_weight`

- **Phase 4 — Whitespace Ranking / Next‑Best‑Action**
  - Signals: calibrated probability (`p_icp` + per‑division percentile), market‑basket affinity (`mb_lift_max`, `mb_lift_mean`), ALS similarity (explicit or embedding‑centroid), expected value proxy (segment‑blended and capped)
  - Normalization: per‑division percentile (default) or pooled across divisions
  - Blending: configurable weights (default `0.60,0.20,0.10,0.10`) → single actionable score
  - Capacity: top‑percent, per‑rep, or hybrid diversification (round‑robin interleave)
  - Gating: ownership, region, recent contact, open deal; cooldown de‑emphasis; structured JSONL logs
  - Artifacts: `whitespace_<cutoff>.csv`, `whitespace_explanations_<cutoff>.csv`, `thresholds_whitespace_<cutoff>.csv`, `whitespace_metrics_<cutoff>.json`, `whitespace_log_<cutoff>.jsonl`, `mb_rules_<division>_<cutoff>.csv`

---

### Quick Start (Windows/PowerShell)

```powershell
# 1) Create/activate venv and install
python -m venv .venv; .venv\Scripts\activate.ps1; pip install -r gosales/requirements.txt

# 2) Phase 0 — Build star schema (curated)
$env:PYTHONPATH = "$PWD"; python -m gosales.etl.build_star --config gosales/config.yaml --rebuild

# 3) Phase 1 — Build labels
$env:PYTHONPATH = "$PWD"; python -m gosales.pipeline.build_labels --division Solidworks --cutoff "2024-06-30" --window-months 6 --mode expansion --config gosales/config.yaml

# 4) Phase 2 — Build features
$env:PYTHONPATH = "$PWD"; python -m gosales.features.build --division Solidworks --cutoff "2024-06-30" --config gosales/config.yaml

# 5) Phase 3 — Train across cutoffs (example)
$env:PYTHONPATH = "$PWD"; python -m gosales.models.train --division Solidworks --cutoffs "2023-06-30,2023-09-30,2023-12-31" --window-months 6 --models logreg,lgbm --calibration platt,isotonic --config gosales/config.yaml

# 6) Score all available division models
$env:PYTHONPATH = "$PWD"; python gosales/pipeline/score_all.py

# 7) Phase 4 — Rank whitespace (example)
$env:PYTHONPATH = "$PWD"; python -m gosales.pipeline.rank_whitespace --cutoff "2024-06-30" --window-months 6 --normalize percentile --capacity-mode top_percent --config gosales/config.yaml
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
| `gosales/outputs/thresholds_whitespace_<cutoff>.csv` | Capacity thresholds (top‑percent/per‑rep/hybrid) | `gosales/pipeline/rank_whitespace.py`
| `gosales/outputs/whitespace_metrics_<cutoff>.json` | Capture@K, division shares, stability, coverage, weights | `gosales/pipeline/rank_whitespace.py`
| `gosales/outputs/whitespace_log_<cutoff>.jsonl` | Structured JSONL logs (division summaries, selection summary) | `gosales/pipeline/rank_whitespace.py`
| `gosales/outputs/mb_rules_<division>_<cutoff>.csv` | Market‑basket rule table with SKU lift and support | `gosales/features/engine.py`
| `gosales/outputs/features_<division>_<cutoff>.parquet` | Feature matrix snapshot | `gosales/features/build.py`
| `gosales/outputs/feature_catalog_<division>_<cutoff>.csv` | Feature names and coverage | `gosales/features/build.py`
| `gosales/outputs/feature_stats_<division>_<cutoff>.json` | Coverage, winsor caps, checksum | `gosales/features/build.py`
| `gosales/outputs/labels_<division>_<cutoff>.parquet` | Labels per (customer, division, cutoff) | `gosales/pipeline/build_labels.py`
| `gosales/outputs/prevalence_<division>.csv` | Label prevalence summary | `gosales/pipeline/build_labels.py`
| `gosales/outputs/cutoffs_<division>.json` | Cutoff metadata summary | `gosales/pipeline/build_labels.py`

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
  - Normalization sanity and pooled/per‑division behavior
  - Weight scaling by coverage (ALS/affinity) and monotonic affinity
  - ALS fallback similarity via embedding centroid
  - Bias share computation and diversification capacity options
  - Deterministic ranking order; Capture@K sanity

---

### Multi-division support

Known divisions are sourced from `etl/sku_map.division_set()`; cross-division features adapt automatically. Scoring auto-discovers models in `models/*_model` and scores each division with an available model. To add a division, update `etl/sku_map.py` (or overrides CSV), rebuild star and features, then train with `--division <Name>`.
