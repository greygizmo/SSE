# GoSales Engine Architecture Documentation

This directory contains comprehensive Mermaid diagrams documenting every phase of the GoSales Engine architecture. These diagrams illustrate components, data flows, and interactions across ETL, features, training, validation, monitoring, and UI.

## Diagram Overview

### 1. Overall Architecture (`01_overall_architecture.mmd`)
Purpose: High‑level system overview.
Components: External Sources (Azure SQL, Model Registry), Core Pipeline (ETL, Features, Training), Validation, Monitoring, UI, Storage.
Key Flows: Ingestion to curated DB, feature pipeline, training + calibration, monitoring/alerts, dashboard reporting.

### 2. ETL Flow (`02_etl_flow.mmd`)
Purpose: Detailed extract‑transform‑load process.
Phases: Configuration, ingestion, cleaning/standardization, star schema, storage, logging.
Key Components: `ingest.py`, `cleaners.py`, `build_star.py`, `load_csv.py`, `check_connection.py`.

### 3. Feature Engineering Flow (`03_feature_engineering_flow.mmd`)
Purpose: Comprehensive feature pipeline.
Types: Customer (RFM), product, temporal (rolling/seasonality), ALS embeddings, external industry integration, branch/rep features.
Key Components: `engine.py`, `als_embed.py`, `cache.py`, `fact_sales_log_raw`.

### 3b. Feature Families (`03b_feature_families.mmd`)
Purpose: Dedicated view of cycle‑aware recency, offset windows, deltas, pooled encoders, lagged affinity, assets, ALS, and config toggles.

### 4. Model Training Flow (`04_model_training_flow.mmd`)
Purpose: End‑to‑end training pipeline.
Phases: Init/config, dataset prep, model selection, hyper‑opt, evaluation, packaging.
Key Components: `gosales/models/train.py` (robust trainer), LightGBM + LR, MLflow tracking, SHAP explainability.

### 5. Pipeline Orchestration Flow (`05_pipeline_orchestration_flow.mmd`)
Purpose: Full pipeline execution.
Components: Init/config, ETL → Features → Training → Validation sequencing, scoring, whitespace, results persistence.
Key: `score_all.py`, `score_customers.py`, `label_audit.py`.

### 6. Validation & Testing Flow (`06_validation_testing_flow.mmd`)
Purpose: Validation framework.
Includes: Data quality, model performance, holdout testing, deciles, business rules, statistical tests, integration tests.
Key: `data_validator.py`, `validate_holdout.py`, `deciles.py`, `ci_gate.py`.

### 7. Monitoring System Flow (`07_monitoring_system_flow.mmd`)
Purpose: Monitoring and observability.
Includes: System metrics, pipeline health, alerts, lineage, performance analytics, QA monitoring.
Key: `pipeline_monitor.py`, `data_collector.py`.

### 8. UI/Dashboard Flow (`08_ui_dashboard_flow.mmd`)
Purpose: Streamlit UI architecture.
Sections: Overview, model performance + explainability, whitespace, validation results, pipeline history, monitoring.

### 9. Sequence Diagrams (`09_sequence_diagrams.mmd`)
Purpose: Key sequence interactions across subsystems.

### 10. Quality Assurance Flow (`10_quality_assurance_flow.mmd`)
Purpose: Data/model QA and gating steps.

### 11. Prequential Evaluation (`11_prequential_evaluation.mmd`)
Purpose: Monthly forward evaluation from a frozen cutoff with label observability clamp; outputs JSON/CSV/PNG.

### 12. Adjacency Ablation & SAFE (`12_adjacency_ablation_and_safe.mmd`)
Purpose: Full vs No‑Recency/Short vs SAFE; select by holdout AUC, compute ΔAUC, gate; Auto‑SAFE updates config.

### 13. Segments and Embeddings (`13_segments_and_embeddings.mmd`)
Purpose: Segment‑aware weighting and embedding integrations.

## How To View

- VS Code: Mermaid extension renders `.mmd` files inline.
- Web: Copy diagram code to online Mermaid editors.

## Color Legend (diagram styles)

- Setup/Initialization: Light blue
- Data Processing: Purple
- Success States: Green
- Error States: Red
- Processing Steps: Orange
- UI/Dashboard: Pink
- Storage/Output: Gray

## Key Architecture Principles

1) Modular design: independent phases, clear separation of concerns, reusable components.
2) Data quality: strict typing, multi‑stage validation, lineage preservation.
3) Observability: real‑time health, alerts, performance tracking.
4) Scalability: caching, efficient storage, parallelizable workloads.
5) Reliability: resilient error handling, comprehensive logging, CI gates.

## Pipeline Execution Flow

```
Raw Data (Azure SQL)
  -> ETL (ingest.py, cleaners.py, build_star.py)
  -> Feature Engineering (engine.py, als_embed.py)
  -> Model Training (gosales/models/train.py)
  -> Validation (data_validator.py, validate_holdout.py)
  -> Scoring & Analysis (score_all.py, score_customers.py)
  -> Dashboard & Monitoring (gosales/ui/app.py, pipeline_monitor.py)
```

## Monitoring Dashboard Features

- Pipeline Health: real‑time status and metrics
- Data Quality: type consistency and completeness scores
- Performance: throughput, latency, and resource usage
- Alerts: active warnings and historical alerts
- Data Lineage: complete audit trail of transformations
- Configuration: system settings and version tracking

## Key Integration Points

- Database: Azure SQL (source) + SQLite (curated)
- Models: LightGBM and LR with MLflow tracking
- Monitoring: psutil for system metrics (with fallback)
- UI: Streamlit with real‑time data updates
- CI/CD: GitHub Actions with quality gates
- Storage: local file system with structured outputs

## Contributing

When architecture changes:
1) Update the relevant Mermaid diagrams
2) Keep styling consistent
3) Add new components to the overall architecture diagram
4) Document any new integration points
5) Update this README

## Architecture Evolution

This documentation reflects the current system. As it evolves:
- Add diagrams for new features
- Update existing diagrams to match code
- Maintain version history
- Clearly note breaking changes

---

Recent Enhancements (2025‑09)

- Feature Engineering
  - Cycle‑aware recency transforms (log, hazard decays)
  - Offset windows (e.g., 12m ending cutoff‑60d) and 12m vs previous 12m deltas
  - Hierarchical/pooled encoders for industry and sub‑industry (non‑leaky; pre‑cutoff)
  - Lagged market‑basket affinity features with ≥60d embargo

- Training & Evaluation
  - Per-division SAFE policy; selection by lift@K + Brier
  - Model cards include top-K yield summaries and calibration method/MAE
  - Prequential forward-month evaluation (AUC, Lift@10, Brier) with label observability clamp
  - Segment-aware training: CLI `--segment warm|cold|both` overrides config for ad-hoc runs; pipeline and UI respect the same override.
  - Calibration fallback ensures artifacts still emit even when probabilities collapse (`calibration_fallback` recorded in metadata).

- Validation & CI Gates
  - Permutation test (train‑only shuffle within time buckets) with p‑value
  - Shift‑grid {7,14,28,56} non‑improving check
  - Adjacency Ablation Gate (Full ≥ SAFE or adopt SAFE) integrated into `ci_gate`
  - Auto‑SAFE helper updates `modeling.safe_divisions` from ablation artifacts

- UI
  - Feature Guide tab (families + configuration + tuning tips)
  - Business Yield (Top-K) table and coverage curve in Metrics
  - Prequential and Adjacency Ablation viewers in QA
  - Roster diagnostics CSVs documented in the Artifact Catalog for transparency into segment math

