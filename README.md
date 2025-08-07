# GoSales Engine (v2)

A **division-focused Ideal Customer Profile (ICP) & Whitespace Engine** for B2B sales organizations.
This project ingests raw CSV sales logs, transforms them into a tidy transaction table, engineers a rich set of behavioral features, and trains predictive models to score every customer's likelihood of purchasing from a specific business division.

The key architectural shift in v2 is from a simple per-product model to a robust, **division-level, time-aware modeling pipeline**. This prevents the common "label leakage" problem and produces realistic, actionable scores.

---

## 1. Core Concepts

| Phase | What it does | Key Files |
|-------|--------------|-----------|
| **ETL** | Loads raw wide CSVs → stages data → unpivots into a tidy `fact_transactions` table where each row is a single SKU sold. | `gosales/etl/build_star.py` |
| **Feature Engineering** | Builds a feature matrix for a specific **division** (e.g., Solidworks) using data **up to a specified `cutoff_date`**. This includes recency, frequency, monetary, and cross-division behavioral features. | `gosales/features/engine.py` |
| **Model Training** | Trains a model to predict which customers will buy from a division in a **future time window** (e.g., 6 months after the cutoff). This time-based split is critical for preventing leakage. | `gosales/models/train_division_model.py` |
| **Customer Scoring** | Uses the trained model to generate ICP scores for all customers. | `gosales/pipeline/score_customers.py` |
| **Whitespace Analysis** | Identifies products and divisions a customer has not purchased, prioritized by their ICP score. | `gosales/pipeline/score_customers.py` |
| **Validation** | Tests the model's performance on a **holdout dataset** (e.g., 2025 data) to get a realistic measure of its predictive power. | `gosales/pipeline/validate_holdout.py` |
| **Dashboard** | A Streamlit app to visualize ICP scores and explore whitespace opportunities. | `gosales/ui/app.py` |

---

## 2. Quick-Start (Windows/PowerShell)

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

# 3) Run the end-to-end training and scoring pipeline
# This trains a model on historical data.
python gosales/pipeline/score_all.py

# 4) (Optional) Run the validation pipeline against holdout data
# This tests the trained model on future data to get a realistic AUC.
python gosales/pipeline/validate_holdout.py

# 5) Launch the Streamlit Dashboard
.\run_streamlit.ps1
```

---

## 3. Data Flow

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

## 4. How to Add a New Division

The new architecture is designed for easy expansion. To add a model for the 'Simulation' division:

1.  **Add to SKU Mapping**: In `gosales/etl/build_star.py`, ensure all relevant 'Simulation' SKUs are mapped to the `Simulation` division.
2.  **Update Feature Engine**: In `gosales/features/engine.py`, add any features specific to Simulation cross-selling (e.g., `has_bought_solidworks`).
3.  **Train the Model**: Run the training pipeline, but change the `target_division` in `gosales/pipeline/score_all.py` to `'Simulation'`.
4.  **Update Scoring**: In `gosales/pipeline/score_customers.py`, add `'Simulation'` to the list of divisions to be scored.

---

## 5. Directory Layout

```
gosales/
├─ data/
│  ├─ database_samples/     # Primary training data CSVs
│  └─ holdout/              # Holdout validation data (e.g., 2025 YTD)
├─ etl/                      # Ingestion & star-schema builders
├─ features/                 # Time-aware feature engineering
├─ models/                   # Training scripts & MLflow model artifacts
├─ pipeline/                 # Orchestration scripts (score_all, validate_holdout)
├─ ui/                       # Streamlit application
├─ utils/                    # DB helper, logger, etc.
└─ outputs/                  # Pipeline results (scores, metrics)
```
