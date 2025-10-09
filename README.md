## GoSales Engine — ICP & Whitespace (Phases 0–6)

A division-focused Ideal Customer Profile (ICP) & Whitespace engine. The pipeline ingests raw sales logs, builds a curated star schema, engineers leakage-safe features at a time cutoff, trains and calibrates per-division models, and produces scores and whitespace opportunities ready for a Streamlit UI.

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
  - Asset features at cutoff: `assets_expiring_{30,60,90}d_<rollup>`, subscription shares (`assets_subs_share_<rollup>`), composition (`assets_on_subs_share_<rollup>`, `assets_off_subs_share_<rollup>`)
  - Artifacts: features parquet, feature catalog CSV, feature stats JSON (coverage, winsor caps, checksum)
  - Determinism and winsorization tests
- **Phase 3 — Modeling**
  - Config-driven modeling grids and seeds
  - Training CLI for LR (elastic-net) and LGBM across multiple cutoffs, with calibration (Platt/Isotonic) and selection by mean lift@10 (tie-breaker Brier)
  - Training cutoffs resolve from configuration (`run.training_cutoffs` or auto-generated via `training_frequency_months`/`training_cutoff_count`) and can be overridden at runtime with `python -m gosales.pipeline.score_all --training-cutoffs 2024-06-30,2024-12-31`
  - LR now trains via a Pipeline: `StandardScaler(with_mean=False)` → `LogisticRegression`; calibration is applied to the entire pipeline; coefficient export unwraps the calibrated pipeline
  - LightGBM remains scale-invariant (no scaler in front)
  - Metrics: AUC, PR-AUC, Brier, lift@{5,10,20}%, revenue-weighted lift@K, calibration MAE
  - Calibration: adaptive CV per cutoff and at final fit. When per-class counts cannot support `cv>=2`, calibration is skipped and uncalibrated probabilities are used; diagnostics record `calibration='none'` with a reason. Isotonic is automatically downgraded to Platt when positives are sparse (see `modeling.sparse_isotonic_threshold_pos`).
  - Diagnostics: emits `diagnostics_<division>.json` with a `results_grid` containing one row per cutoff per model, including `auc`, `lift10`, `brier`, and calibration fields (`calibration`, `calibration_reason`). This confirms every cutoff contributed to aggregation and surfaces any calibration fallbacks.
  - Artifacts: `metrics.json`, `gains.csv`, `calibration.csv`, `thresholds.csv`, `model_card.json`, SHAP summaries (optional; guarded if SHAP not installed)
  - Tuning guide: see `gosales/docs/FEATURES_AND_CONFIG.md` (calibration behavior, diagnostics, and practical tuning tips).
  - Guardrails: degenerate classifier check, deterministic LGBM, early stopping, overfit-gap guard, capped `scale_pos_weight`

- **Phase 4 — Whitespace Ranking / Next‑Best‑Action**
  - Signals: calibrated probability (`p_icp` + per‑division percentile), market‑basket affinity (`mb_lift_max`, `mb_lift_mean`), ALS similarity, expected value proxy (capped)
  - Normalization: per‑division percentile (default) or pooled; pooled normalization preserves per‑division coverage‑adjusted weights when recomputing pooled scores
  - Capacity: top‑percent, per‑rep, or hybrid diversification; gating and cooldown; JSONL logs
  - Artifacts: `whitespace_<cutoff>.csv`, `whitespace_explanations_<cutoff>.csv`, `thresholds_whitespace_<cutoff>.csv`, `whitespace_metrics_<cutoff>.json`, `whitespace_log_<cutoff>.jsonl`, `mb_rules_<division>_<cutoff>.csv`

- **Phase 5 — Forward Validation / Holdout**
  - CLI: `python -m gosales.validation.forward --division Solidworks --cutoff 2024-12-31 --window-months 6 --capacity-grid 5,10,20 --accounts-per-rep-grid 10,25`
  - Artifacts per division/cutoff in `gosales/outputs/validation/<division>/<cutoff>/`: `validation_frame.parquet`, `gains.csv`, `calibration.csv`, `topk_scenarios*.csv`, `segment_performance.csv`, `metrics.json`, `drift.json`
  - Phase 3 emits `train_scores_*` and `train_feature_sample_*` to support drift

- **Phase 6 — Configuration, UX, Observability**
  - Central config precedence and stricter validation
  - Run registry/manifests via `run_context` with per‑run `config_resolved.yaml`
  - Monitoring telemetry: `MonitoringDataCollector` derives processing rate, division activity, and
    customer totals from the latest run context, validation metrics, and live DB counts.
    When artifacts are missing it now records explicit fallbacks so dashboards highlight data gaps
    instead of masking issues.
  - Database connectivity guardrails: pipeline entrypoints call `validate_connection` and honor
    `database.strict_db`. `get_db_connection` now falls back to the configured SQLite URL when
    Azure credentials are absent, keeping orchestration resilient.
  - Validation improvements: weighted PSI(EV vs holdout GP), per‑feature PSI highlights, `alerts.json`
  - Streamlit UI: artifact‑driven pages, validation badges (Cal MAE, PSI, KS), alerts display, caching + refresh

---

### Quick Start (Windows/PowerShell)

```powershell
# 1) Clone the repository and set up the Python environment
git clone https://github.com/your-org/gosales-engine.git && cd gosales-engine
python -m venv .venv
.venv\Scripts\activate.ps1
pip install -r gosales/requirements.txt
#   The bundle now includes Click, which powers all pipeline CLIs

# 2) Place your raw sales data
# The main training data (e.g., 2023-2024)
copy "path\to\your\Sales_Log.csv" "gosales\data\database_samples\"

# The holdout validation data (e.g., 2025 YTD)
copy "path\to\your\Sales Log 2025 YTD.csv" "gosales\data\holdout\"

# 3) Phase 0 — Build star schema (curated)
$env:PYTHONPATH = "$PWD"; python -m gosales.etl.build_star --config gosales/config.yaml --rebuild

# 4) Phase 1 — Build labels
$env:PYTHONPATH = "$PWD"; python -m gosales.pipeline.build_labels --division Solidworks --cutoff "2024-06-30" --window-months 6 --mode expansion --config gosales/config.yaml

# 5) Phase 2 — Build features
$env:PYTHONPATH = "$PWD"; python -m gosales.features.build --division Solidworks --cutoff "2024-06-30" --config gosales/config.yaml

# 6) Phase 3 — Train models across cutoffs (example)
$env:PYTHONPATH = "$PWD"; python -m gosales.models.train --division Solidworks --cutoffs "2023-06-30,2023-09-30,2023-12-31" --window-months 6 --models logreg,lgbm --calibration platt,isotonic --config gosales/config.yaml

# 6) End-to-end: ingest → build star → audit labels → train per-division → score/whitespace
$env:PYTHONPATH = "$PWD"; python gosales/pipeline/score_all.py

# When scoring directly, `gosales/pipeline/score_customers.py` expects each model folder
# to include `metadata.json` with `cutoff_date` and `prediction_window_months`. If those
# fields are missing, supply them via `--cutoff-date` and `--window-months` arguments or
# the scoring run will error.

# 7) Phase 4 — Rank whitespace (example)
$env:PYTHONPATH = "$PWD"; python -m gosales.pipeline.rank_whitespace --cutoff "2024-06-30" --window-months 6 --normalize percentile --capacity-mode top_percent --config gosales/config.yaml

# 8) Launch Streamlit UI (Phase 6)
$env:PYTHONPATH = "$PWD"; streamlit run gosales/ui/app.py
#   Within the UI, open the **Docs** navigation tab to browse repository guides such as the calibration tuning guide.
#   Interactive charts depend on Plotly, which now ships via `gosales/requirements.txt`.
```

---

### Post-rebase updates (August 2025)

- Scoring pipeline now supports sanitized probability fallback when models lack `predict_proba` or expose only `decision_function`. See `gosales/pipeline/score_customers.py`.
- Ranker restored and hardened: eligibility checks, ALS normalization, deterministic `nlargest` selection, and schema validations.
- Whitespace lift builder uses boolean baskets to silence mlxtend deprecation warnings.
- ALS components pass CSR matrices to `implicit` and cap BLAS threads to avoid performance warnings. BLAS thread limit is applied centrally in `gosales/__init__.py`.
- CLI for scoring accepts `--cutoff-date` and `--window-months` fallbacks when model `metadata.json` is missing these fields.

#### Warnings during tests

You may still see warnings from dependencies in CI:
- `implicit` suggests setting `OPENBLAS_NUM_THREADS=1`. We enforce this behavior at runtime using `threadpoolctl` to limit BLAS to 1 thread.
- `mlxtend` deprecation on non-boolean baskets has been addressed by emitting boolean dummies.
- Pandas groupby and sklearn “feature names” notices are benign in tests; we prefer code clarity over suppressing these globally.

---

### Modeling & validation notes

- Class imbalance is handled via class weights (LR) and `scale_pos_weight` (LightGBM).
- Probability calibration curves are exported to `gosales/outputs/calibration_<division>.csv`.
- Holdout labels for 2025 are derived directly from `gosales/data/holdout/Sales Log 2025 YTD.csv` using `Division == 'Solidworks'` and dates in Jan–Jun 2025.

#### Warning handling (pandas/sklearn)
- Pandas groupby: we explicitly set `observed=False` on groupby operations used to build calibration and gains tables to avoid version‑dependent FutureWarnings
- CSV reads: holdout CSVs are read with `dtype=str` and `low_memory=False`, then numerics are coerced explicitly, preventing mixed‑type `DtypeWarning` while keeping behavior deterministic

### Feature Library (Phase 2 highlights)

The feature set includes and extends:
- Core: recency, frequency, monetary; product and SKU diversity.
- Windowed (3/6/12/24m): transaction counts, GP sums, average GP per transaction.
- Temporal dynamics (12m): monthly GP/TX slope and volatility.
- Cadence: tenure_days, interpurchase intervals (median/mean), last_gap_days.
- Seasonality (24m): quarter shares (q1..q4).
- Division mix (12m): per-division GP/TX totals, GP shares, days_since_last_{division}.
- SKU micro-signals (12m): sku_gp, sku_qty, gp_per_unit for key SKUs.
- Industry join and selected interaction terms.

Artifacts: `gosales/outputs/feature_catalog_<division>_<cutoff>.csv` lists feature names and coverage.

---

### Data Flow

The new data flow is designed to prevent leakage by strictly separating past and future data.

```mermaid
graph TD
    subgraph "Training Pipeline"
        A[Raw CSVs <br/> (e.g., 2023-2024)] --> B{ETL}
        B --> C[fact_transactions]
        C --> CE[Eventization <br/> fact_events]
        CE --> D{Feature Engine <br/> cutoff_date='2024-12-31'}
        D --> E[Feature Matrix]
    end

    subgraph "Holdout / Future Data"
        F[Raw CSVs <br/> (e.g., 2025 YTD)] --> G{ETL}
        G --> H[Future Transactions]
    end

    subgraph "Model Training & Validation"
        I(Define Target Labels)
        E --> I
        H -- defines labels for --> I
        I --> J(Train Model)
        J --> K{Trained Model <br/> solidworks_model}
    end

    subgraph "Scoring & UI"
        K --> L(Score All Customers)
        L --> M[icp_scores.csv]
        L --> N[whitespace.csv]
        M & N --> O(Streamlit UI)
    end
```

1.  **ETL**: Raw CSVs are loaded and transformed into a clean `fact_transactions` table (includes `invoice_id` when available).
1a. **Eventization**: `fact_events` aggregates line items by invoice and stamps per-model labels (Printers, SWX_Seats, etc.).
2.  **Feature Engineering**: A `cutoff_date` is used to build features *only* from historical data.
3.  **Target Labeling**: The model is trained to predict purchases that happen in a *future* window.
4.  **Validation**: A separate holdout dataset (e.g., 2025 data) is used to measure the model's true performance.

---

### Leakage Gauntlet (Batch)

Run leakage checks across all divisions at a cutoff and emit a cross-division summary JSON.

```powershell
$env:PYTHONPATH = "$PWD"
python -m gosales.pipeline.run_leakage_all --cutoff 2024-06-30 --window-months 6
# Optional: include shift-grid summaries (non-gating info)
python -m gosales.pipeline.run_leakage_all --cutoff 2024-06-30 --window-months 6 --run-shift-grid

# Limit to specific divisions
python -m gosales.pipeline.run_leakage_all --cutoff 2024-06-30 --divisions "Printers,SWX_Seats,Training"
```

Artifacts
- Per-division report: `gosales/outputs/leakage/<Division>/<cutoff>/leakage_report_<Division>_<cutoff>.json`
- Cross-division summary: `gosales/outputs/leakage/leakage_summary_<cutoff>.json`

Gauntlet gates
- Gates PASS/FAIL on: static_scan, feature_date_audit, shift14 (dynamic), reproducibility.
- Provides non-gating info: top-k ablation (OK/SUSPECT), optional shift-grid summary.

```mermaid
flowchart TD
    A[Start Batch] --> B[Resolve Targets]
    B --> C{For each division}
    C --> D[Static Scan]
    C --> E[Feature-Date Audit]
    C --> F[Shift-14 (LR)]
    C --> G[Reproducibility (LR)]
    D --> H{Gate}
    E --> H
    F --> H
    G --> H
    H -->|Per-division JSON| I[Write Report]
    I --> J{Any FAIL?}
    J -->|Yes| K[Overall FAIL]
    J -->|No| L[Overall PASS]
    K --> M[Write Cross-Div Summary]
    L --> M
```

---

### Multi-division support

Known divisions are sourced from `etl/sku_map.division_set()`; cross-division features adapt automatically.
Training and scoring auto-discover divisions: the `score_all` pipeline now trains and audits labels for every known division, then generates scores/whitespace for any division with an available model.
To add a division: extend `etl/sku_map.py` (or overrides CSV), rebuild the star and features, then either run `score_all` or train explicitly with `gosales/models/train.py`.

#### Targets vs Divisions
- Divisions (reporting): Solidworks, PDM, Simulation, Services, Training, Success Plan, Hardware, CPE, Scanning, CAMWorks, Maintenance.
- Logical models (SKU-based targets): Printers, SWX_Seats, PDM_Seats, SW_Electrical, SW_Inspection, plus divisions that are targets (Services, Training, Simulation, Success_Plan, Scanning, CAMWorks).
- Mapping metadata: each SKU maps to a `division`, with modeling fields `family` and `sale_type` to distinguish targets (e.g., `sale_type=Printer`) from predictors (e.g., `sale_type=Consumable`, `sale_type=Maintenance`).
- AM_Support is routed by the source DB `Division` into Hardware or Scanning during ETL to avoid misclassification.

The orchestrator `pipeline/score_all.py` collects training targets as (divisions minus `{Hardware, Maintenance}`) union the supported logical models from `etl/sku_map.get_supported_models()`. You can also train a specific model directly (e.g., `--division Printers`).

#### Troubleshooting and notes
- If training encounters numeric issues (inf/NaN or unstable features), use the robust trainer `gosales/models/train.py`, which sanitizes features (NaN/inf handling, low‑variance and high‑correlation pruning) and performs hyper‑search across cutoffs.
- If a division has too few positives (e.g., `FDM` at 0 positives in the default window), widen the window, aggregate sub‑divisions, or adjust the SKU map. Label audit artifacts in `gosales/outputs/labels_*` will show prevalence.
- SHAP artifacts are optional; if `shap` is not installed, training proceeds without explainability exports.

---

### How to interpret whitespace metrics (for revenue teams)

- Capture@K
  - Meaning: Of all the wins the model expects in this period, what fraction are covered if you work just the top K% of the ranked list.
  - Why it matters: Indicates how concentrated opportunity is. Higher is better for efficiency.
  - Thumb rule: If `capture@10% = 0.65`, then working the top 10% could capture ~65% of expected wins.

- Division shares in the selected list
  - Meaning: Within your capacity slice (e.g., top 10% or per‑rep list), what fraction comes from each division.
  - Why it matters: Prevents over‑concentration (e.g., 90% Solidworks). If a division exceeds the configured share threshold, the system warns and suggests hybrid capacity to diversify.

- Stability (Jaccard) vs last run
  - Meaning: Overlap between this run’s top‑N and the previous run’s top‑N (0–1).
  - Why it matters: High (~0.7–0.9) = consistent targeting; low (~0.3) = shift due to seasonality, new data, or configuration change.
  - Action: If stability drops unexpectedly, review data recency, config changes, and business events.

- Coverage and weight adjustments
  - What it shows: Coverage for ALS and market‑basket signals (what % of customers have these signals). Low coverage down‑weights that signal and renormalizes.
  - Why it matters: Low coverage isn’t “bad”, it reflects current data sparsity. As coverage improves, those signals earn more weight.

- Thresholds & capacity
  - Top‑percent: Work the top X% across all divisions; thresholds file lists the score cut line and counts.
  - Per‑rep: Each rep gets roughly the top N in their book (requires a `rep` column in features). Encourages fair distribution.
  - Hybrid: Round‑robin across divisions up to capacity to ensure diversification.

- Probability and value
  - `p_icp`: Calibrated probability (0–1) of a positive outcome in the prediction window.
  - `EV_norm`: Normalized expected‑value proxy (segment‑blended and capped at a high percentile to avoid whale effects).

- Explanations (`nba_reason`)
  - Short, human‑readable context such as “High p=0.78; strong affinity; high EV.” For deeper model insights, use Phase‑3 SHAP/coef artifacts.

- Recommended operating mode
  - Pilot: pick one capacity mode (e.g., top‑10% or per‑rep 25) and run 4–6 weeks. Track conversion vs a comparable control group.
  - Review weekly: capture@K, division shares, stability, and rep feedback. Adjust capacity and weights if needed.

- Cautions
  - This is not a price quote or guarantee; it’s a ranked opportunity list.
  - If gating (DNC/legal/open deals/region) removes many accounts, the selected list may shrink and division shares may shift—this is expected.
  - Cooldowns reduce repeated surfacing of the same account that wasn’t actioned; adjust in config if needed.

---

### Utilities / Scripts

- `scripts/metrics_summary.py`: aggregate `metrics_*.json` into a summary CSV
- `scripts/ci_assets_sanity.py`: CI‑style gate for asset rollup coverage and tenure imputation sanity
- `scripts/drift_snapshots.py`: aggregate prevalence and calibration MAE across validation runs
- `scripts/name_join_qa.py`: Moneyball→dim_customer name‑join QA; writes coverage summary and top unmapped names
- `scripts/ablation_assets_off.py`: Train with assets disabled and compare metrics vs baseline; writes ablation JSON
- `scripts/build_features_for_models.py`: Build feature matrices for each trained model’s cutoff to align feature lists
- `scripts/train_all_models.py`: Retrain all target models at a given cutoff (group‑CV, calibration)

---

## Repository structure

```
gosales/
├─ data/
│  ├─ database_samples/     # Primary training data CSVs
│  └─ holdout/              # Holdout validation data (e.g., 2025 YTD)
├─ etl/                      # Ingestion & star-schema builders
├─ features/                 # Time-aware feature engineering
├─ models/                   # Training CLI and artifacts
├─ pipeline/                 # Orchestration scripts (score_all, validate_holdout)
├─ ui/                       # Streamlit application
├─ utils/                    # DB helper, logger, etc.
└─ outputs/                  # All run artifacts (git-ignored)
```

---

## Recent Additions (2025‑09‑05)

- Asset integrations
  - `gosales/etl/assets.py` builds `fact_assets` from Moneyball × items rollup and implements effective purchase date imputation for legacy years. Feature engine merges asset features strictly at cutoff (active counts, expiring 30/60/90d, tenure, bad‑date share).
  - Utilities: `scripts/peek_assets_views.py`, `scripts/build_assets_features.py`.

- Scoring/Ranking improvements
  - Adaptive batch scoring keeps ALS, market-basket, pooled encoders, and missingness flags enabled while capping peak memory; override size with `GOSALES_BATCH_TARGET_MB` (MB, default 160).
  - Scorer reindexes to `feature_list.json` and zero‑fills to avoid LightGBM shape mismatches.
  - Signals propagated to ranker: `mb_lift_max`, `mb_lift_mean`, `als_f*`, and EV proxy. Capacity summary now exported as `capacity_summary_<cutoff>.csv`.
  - Output writer is resilient to Windows file locks on `icp_scores.csv`; a timestamped fallback is written and a warning logged.

- Leakage diagnostics
  - `python -m gosales.pipeline.leakage_diagnostics --division <Div> --cutoff YYYY-MM-DD --window-months 6` runs SAFE-masked permutation tests and importance stability bootstraps; artifacts land in `gosales/outputs/leakage/<Div>/<cutoff>/`.
  - Full Gauntlet automation (`python -m gosales.pipeline.run_leakage_gauntlet`) remains available for legacy gating; see `gosales/docs/legacy/` for the archived playbooks.

- Metrics roll‑up
  - `scripts/metrics_summary.py` creates `gosales/outputs/metrics_summary.csv` from `metrics_*.json` across divisions.
