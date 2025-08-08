# GoSales Engine — ICP & Whitespace (Phases 0–3)

A division-focused Ideal Customer Profile (ICP) & Whitespace engine. The pipeline ingests raw sales logs, builds a curated star schema, engineers leakage-safe features at a time cutoff, trains and calibrates per-division models, and produces scores and whitespace opportunities ready for a Streamlit UI.

---

## What’s implemented (by phase)

- Phase 0 — ETL, Star Schema, Contracts
  - Tidy `fact_transactions` and `dim_customer` with enrichment and fuzzy fallback
  - Contracts: required columns, PK checks, date-bounds; violations CSV
  - Curated Parquet + QA: schema snapshot, row counts, violations, checksums
  - CLI flags: `--config`, `--rebuild`, `--staging-only`, `--fail-soft`
- Phase 1 — Labels
  - Leakage-safe labels per `(customer, division, cutoff)` with modes: `expansion|all`
  - Cohorts (`is_new_logo`, `is_expansion`, `is_renewal_like`), censoring detection
  - Denylist SKUs and GP threshold via config; artifacts: labels parquet, prevalence CSV, cutoff JSON
- Phase 2 — Features
  - RFM windows (3/6/12/24m), trajectory (monthly slope/std), lifecycle (tenure, gaps, active months), seasonality, cross-division shares (EB-smoothed), diversity, returns
  - Optional toggles: market-basket affinity, ALS embeddings
  - Artifacts: features parquet, feature catalog CSV, feature stats JSON (coverage, winsor caps, checksum)
  - Determinism and winsorization tests
- Phase 3 — Modeling (initial)
  - Config-driven modeling grids and seeds
  - Training CLI for LR (elastic-net) and LGBM across multiple cutoffs, with calibration (Platt/Isotonic) and selection by mean lift@10 (tie-breaker Brier)

---

## Quick Start (Windows/PowerShell)

```powershell
# 1) Clone the repository and set up the Python environment
git clone https://github.com/your-org/gosales-engine.git && cd gosales-engine
python -m venv .venv
.venv\Scripts\activate.ps1
pip install -r gosales/requirements.txt

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

# 7) Score and generate whitespace + UI artifacts
$env:PYTHONPATH = "$PWD"; python gosales/pipeline/score_all.py

# 8) Launch the UI
./run_streamlit.ps1
```

---

## Modeling & Validation Notes

- Class imbalance is handled via class weights (LR) and `scale_pos_weight` (LightGBM).
- Probability calibration curves are exported to `gosales/outputs/calibration_<division>.csv`.
- Holdout labels for 2025 are derived directly from `gosales/data/holdout/Sales Log 2025 YTD.csv` using `Division == 'Solidworks'` and dates in Jan–Jun 2025.

## Feature Library (Phase 2 highlights)

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

## 8. Changelog

- feat(features): Phase 2 expansion — windowed RFM, temporal dynamics, cadence, seasonality, division mix, SKU micro-signals.
- feat(validate): derive holdout labels directly from 2025 YTD CSV; append zero-imputed rows for missing buyers in evaluation set.
- feat(modeling): class weighting and probability calibration; export calibration curves.

---

## Data Flow

The new data flow is designed to prevent leakage by strictly separating past and future data.

```mermaid
graph TD
    subgraph "Training Pipeline"
        A[Raw CSVs <br/> (e.g., 2023-2024)] --> B{ETL}
        B --> C[fact_transactions]
        C --> D{Feature Engine <br/> cutoff_date='2024-12-31'}
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

1.  **ETL**: Raw CSVs are loaded and transformed into a clean `fact_transactions` table.
2.  **Feature Engineering**: A `cutoff_date` is used to build features *only* from historical data.
3.  **Target Labeling**: The model is trained to predict purchases that happen in a *future* window.
4.  **Validation**: A separate holdout dataset (e.g., 2025 data) is used to measure the model's true performance.

---

## Multi‑Division support

Known divisions are sourced from `etl/sku_map.division_set()`; cross-division features adapt automatically.
Scoring auto-discovers models in `models/*_model` and scores each division with an available model.
To add a division: extend `etl/sku_map.py` (or overrides CSV) with SKUs and division mapping; rebuild star and features; train with `--division <Name>`.

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
