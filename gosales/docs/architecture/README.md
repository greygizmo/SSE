# GoSales Engine Architecture Documentation

This directory contains comprehensive Mermaid diagrams documenting every phase of the GoSales Engine repository architecture. These diagrams provide detailed insights into the system's components, data flows, and interactions.

## 📋 Diagram Overview

### 1. Overall Architecture (`01_overall_architecture.mmd`)
**Purpose:** High-level overview of the entire GoSales Engine system
**Components Shown:**
- External Data Sources (Azure SQL, Model Registry)
- Core Pipeline (ETL, Feature Engineering, Model Training)
- Validation & Testing Framework
- Monitoring System
- User Interface (Streamlit Dashboard)
- Data Storage Layer

**Key Flows:**
- Data ingestion from Azure SQL to SQLite
- Feature engineering pipeline
- Model training and validation
- Real-time monitoring and alerting
- Dashboard visualization and reporting

### 2. ETL Flow (`02_etl_flow.mmd`)
**Purpose:** Detailed ETL (Extract, Transform, Load) process flow
**Phases Covered:**
- Configuration & Setup
- Data Ingestion (Azure SQL queries)
- Data Cleaning & Standardization
- Star Schema Transformation
- Data Loading & Storage
- Monitoring & Logging

**Key Components:**
- `ingest.py` - Data extraction
- `cleaners.py` - Data cleaning
- `build_star.py` - Star schema creation
- `load_csv.py` - Data loading
- `check_connection.py` - Connection validation

### 3. Feature Engineering Flow (`03_feature_engineering_flow.mmd`)
**Purpose:** Comprehensive feature engineering pipeline
**Feature Types:**
- Customer-level features (recency, monetary, frequency)
- Product-level features (popularity, margins)
- Temporal features (rolling metrics, seasonality)
- ALS collaborative filtering embeddings
- External feature integration (industry data)
- Branch/Rep performance features

**Key Components:**
- `engine.py` - Main feature engineering orchestrator
- `als_embed.py` - ALS embedding generation
- `cache.py` - Feature caching system
- `fact_sales_log_raw` - Raw data preservation

### 4. Model Training Flow (`04_model_training_flow.mmd`)
**Purpose:** End-to-end model training pipeline
**Training Phases:**
- Training initialization and configuration
- Data preparation and preprocessing
- Model architecture selection
- Hyperparameter optimization
- Model evaluation and validation
- Model packaging and deployment

**Key Components:**
- `train_division_model.py` - Division-specific training
- LightGBM model architecture
- MLflow integration for tracking
- SHAP value generation for explainability

### 5. Pipeline Orchestration Flow (`05_pipeline_orchestration_flow.mmd`)
**Purpose:** Complete pipeline execution flow
**Orchestration Components:**
- Pipeline initialization and configuration
- Sequential phase execution (ETL → Features → Training → Validation)
- Customer-specific scoring
- Whitespace analysis
- Results processing and storage

**Key Components:**
- `score_all.py` - Full pipeline execution
- `score_customers.py` - Individual customer scoring
- `label_audit.py` - Label quality validation

### 6. Validation & Testing Flow (`06_validation_testing_flow.mmd`)
**Purpose:** Comprehensive validation framework
**Validation Types:**
- Data quality validation
- Model performance validation
- Holdout testing on unseen data
- Decile analysis for ranking quality
- Business logic validation
- Statistical validation
- Integration testing

**Key Components:**
- `data_validator.py` - Data quality validation
- `validate_holdout.py` - Holdout testing
- `deciles.py` - Decile analysis
- `ci_gate.py` - CI/CD integration

### 7. Monitoring System Flow (`07_monitoring_system_flow.mmd`)
**Purpose:** Enterprise monitoring and observability
**Monitoring Capabilities:**
- Real-time system metrics collection
- Pipeline health monitoring
- Alert generation and management
- Data lineage tracking
- Performance analytics
- Quality assurance monitoring

**Key Components:**
- `pipeline_monitor.py` - Pipeline monitoring
- `data_collector.py` - Metrics collection
- Real-time dashboard integration

### 8. UI/Dashboard Flow (`08_ui_dashboard_flow.mmd`)
**Purpose:** User interface and dashboard architecture
**Dashboard Sections:**
- Overview with key metrics
- Model performance and explainability
- Whitespace opportunity analysis
- Validation results
- Pipeline execution history
- Real-time monitoring dashboard

**Key Components:**
- `app.py` - Main Streamlit application
- `utils.py` - Dashboard utilities
- Interactive visualizations and exports

### 9. Sequence Diagrams (`09_sequence_diagrams.mmd`)
**Purpose:** Detailed interaction flows between components
**Diagrams Included:**
- Complete pipeline execution sequence
- Monitoring dashboard data flow
- Customer recommendation workflow
- Automated scheduling and alerting

## 🎯 How to Use These Diagrams

### Viewing Diagrams
1. **GitHub:** Diagrams render automatically when viewing `.mmd` files
2. **Local:** Use a Mermaid-compatible viewer or VS Code with Mermaid extension
3. **Web:** Copy diagram code to online Mermaid editors

### Understanding the Flow
1. **Start:** Look for green "Start" nodes
2. **Flow:** Follow the arrows to understand process sequence
3. **Components:** Each box represents a specific module or process
4. **Decisions:** Diamond shapes show conditional logic
5. **End States:** Green boxes show success, red show failure

### Color Coding
- 🔵 **Setup/Initialization:** Light blue
- 🟣 **Data Processing:** Purple
- 🟢 **Success States:** Green
- 🔴 **Error States:** Red
- 🟠 **Processing Steps:** Orange
- 🩷 **UI/Dashboard:** Pink
- 🩶 **Storage/Output:** Gray

## 🔧 Key Architecture Principles

### 1. Modular Design
- Each phase is independently executable
- Clear separation of concerns
- Reusable components across phases

### 2. Data Quality Focus
- Type consistency enforcement
- Comprehensive validation at each stage
- Data lineage preservation

### 3. Monitoring & Observability
- Real-time health monitoring
- Comprehensive alerting system
- Detailed performance tracking

### 4. Scalability & Performance
- Caching mechanisms for feature matrices
- Parallel processing for model training
- Efficient data storage patterns

### 5. Enterprise-Grade Reliability
- Error handling and recovery
- Comprehensive logging
- CI/CD integration with quality gates

## 🚀 Pipeline Execution Flow

```
Raw Data (Azure SQL)
    ↓
ETL Process (ingest.py, cleaners.py, build_star.py)
    ↓
Feature Engineering (engine.py, als_embed.py)
    ↓
Model Training (train_division_model.py)
    ↓
Validation (data_validator.py, validate_holdout.py)
    ↓
Scoring & Analysis (score_all.py, score_customers.py)
    ↓
Dashboard & Monitoring (app.py, pipeline_monitor.py)
```

## 📊 Monitoring Dashboard Features

- **Pipeline Health:** Real-time status and metrics
- **Data Quality:** Type consistency and completeness scores
- **Performance:** Throughput, latency, and resource usage
- **Alerts:** Active warnings and historical alerts
- **Data Lineage:** Complete audit trail of data transformations
- **Configuration:** System settings and version tracking

## 🔍 Key Integration Points

- **Database:** Azure SQL (source) → SQLite (curated)
- **Models:** LightGBM with MLflow tracking
- **Monitoring:** psutil for system metrics (with fallback)
- **UI:** Streamlit with real-time data updates
- **CI/CD:** GitHub Actions with quality gates
- **Storage:** Local file system with structured outputs

## 📝 Contributing

When making architecture changes:
1. Update relevant diagrams
2. Maintain consistent styling
3. Add new components to overall architecture diagram
4. Document new integration points
5. Update this README with changes

## 🏗️ Architecture Evolution

This documentation reflects the current state of the GoSales Engine architecture. As the system evolves:
- New diagrams will be added for new features
- Existing diagrams will be updated to reflect changes
- Version history will be maintained in the repository
- Breaking changes will be clearly documented

---

*These diagrams were generated to provide complete transparency into the GoSales Engine architecture, supporting development, debugging, and knowledge sharing across the team.*
