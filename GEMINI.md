# Project: GoSales - ICP & Whitespace Engine (v2)

## Project Overview

This project, codenamed "GoSales," is a **division-focused Ideal Customer Profile (ICP) and whitespace analysis engine**. It is designed to help B2B sales organizations identify high-potential customers by analyzing historical sales data. The system scores existing customers on their likelihood to purchase from a specific business division (e.g., Solidworks, Simulation) and surfaces cross-sell and up-sell opportunities.

The key innovation in v2 is its **time-aware modeling architecture**. By using a strict `cutoff_date` to separate historical feature data from future target labels, the system avoids data leakage and produces realistic, actionable predictive scores.

The project is built in Python and leverages a modern data science stack:

*   **Data Manipulation:** Pandas, Polars
*   **Database Interaction:** SQLAlchemy (for Azure SQL MI or local SQLite)
*   **Machine Learning:** Scikit-learn (for Logistic Regression), LightGBM
*   **Model Management**: MLflow
*   **User Interface:** Streamlit
*   **Linting and Formatting:** Ruff, Black

## Core Architecture

The project follows a modular, pipeline-driven architecture:

1.  **ETL (`gosales/etl/`)**: Ingests raw, wide CSV sales logs and "unpivots" them into a tidy `fact_transactions` table. This is the clean foundation for all downstream analysis.
2.  **Feature Engineering (`gosales/features/`)**: Creates a rich feature matrix for a specified division and `cutoff_date`. Features include RFM (Recency, Frequency, Monetary) metrics, product adoption signals (e.g., seat counts), and cross-divisional behavior.
3.  **Model Training (`gosales/models/`)**: Trains a predictive model (e.g., Logistic Regression) to identify which customers are likely to purchase from the target division within a future prediction window.
4.  **Scoring Pipeline (`gosales/pipeline/`)**: Orchestrates the entire workflow, from ETL to training to final scoring. Includes a separate `validate_holdout.py` script to test model performance on unseen future data.
5.  **UI (`gosales/ui/`)**: A Streamlit dashboard for visualizing customer scores and exploring whitespace opportunities.

## Building and Running

The following steps outline how to set up and run the GoSales project.

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r gosales/requirements.txt
    ```

3.  **Configure environment variables:**
    *   Copy the `.env.template` file to `.env`.
    *   Fill in the `AZSQL_*` variables to connect to an Azure SQL Managed Instance. If left blank, the application will fall back to using a local SQLite database (`gosales.db`).

4.  **Place data files:**
    *   Place primary training data (e.g., 2023-2024 sales logs) into `gosales/data/database_samples/`.
    *   Place holdout validation data (e.g., 2025 YTD sales logs) into `gosales/data/holdout/`.

5.  **Run the pipeline:**
    *   The main training and scoring pipeline is executed via:
        ```bash
        python gosales/pipeline/score_all.py
        ```
    *   To validate the trained model against the holdout data, run:
        ```bash
        python gosales/pipeline/validate_holdout.py
        ```

6.  **Launch the UI:**
    *   To view the results, start the Streamlit application:
        ```bash
        streamlit run gosales/ui/app.py
        ```
    *   On Windows, a convenience script is provided: `.\run_streamlit.ps1`.

## Development Conventions

*   **Modular Design**: The codebase is organized into distinct modules for ETL, features, models, etc., to ensure clarity and maintainability.
*   **Configuration**: Secrets and environment-specific settings are managed in a `.env` file.
*   **Logging**: A custom logger in `utils/logger.py` is used for all informational and error messages.
*   **Type Hinting & Docstrings**: All functions have clear type hints and docstrings.
*   **Linting & Formatting**: `ruff` and `black` are used to enforce a consistent coding style.
*   **Commits**: Commit messages follow the Conventional Commits specification.
