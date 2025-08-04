# Project: GoSales - ICP & Whitespace Engine for Software Sales

## Project Overview

This project, codenamed "GoSales," is a multi-product Ideal Customer Profile (ICP) and whitespace analysis engine designed for software sales. The system analyzes historical sales data to score existing customers on their likelihood to purchase additional products, and to identify "whitespace" opportunities – products a customer has not purchased but is likely to need.

The project is built in Python and leverages a modern data science stack, including:

*   **Data Manipulation:** Pandas, Polars
*   **Database Interaction:** SQLAlchemy (for Azure SQL MI or SQLite)
*   **Machine Learning:** Scikit-learn (for Logistic Regression), LightGBM
*   **Whitespace Analysis:** MLxtend (for Market Basket analysis), Implicit (for Collaborative Filtering)
*   **User Interface:** Streamlit
*   **Workflow Orchestration (planned):** Prefect
*   **Linting and Formatting:** Ruff, Black

The project is designed to be run locally on a single machine, with configuration managed through a `.env` file.

## Building and Running

The following steps outline how to set up and run the GoSales project.

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure environment variables:**
    *   Copy the `.env.template` file to `.env`.
    *   Fill in the `AZSQL_*` variables to connect to an Azure SQL Managed Instance. If left blank, the application will fall back to using a local SQLite database.
    *   Add your `OPENAI_API_KEY` or `GEMINI_API_KEY` if you wish to use the respective LLM vendor.

4.  **Place data files:**
    *   Place the raw CSV data files into the `data/` directory (this directory will need to be created).

5.  **Run the pipeline:**
    *   The main pipeline can be executed by running:
        ```bash
        python pipeline/score_all.py
        ```
    *   Alternatively, a `Makefile` may be provided with a command like `make all`.

6.  **Launch the UI:**
    *   To view the results, start the Streamlit application:
        ```bash
        streamlit run ui/app.py
        ```

## Development Conventions

The project follows a structured and modular design to ensure code quality and maintainability.

*   **Directory Structure:** The codebase is organized into distinct modules for ETL, feature engineering, modeling, whitespace analysis, and UI. This separation of concerns makes the project easier to navigate and extend.
*   **Configuration:** All secrets and environment-specific settings are managed in a `.env` file and should never be hard-coded.
*   **Logging:** All informational and error messages should be logged using the custom logger in `utils/logger.py`. Bare `print()` statements should be avoided.
*   **Type Hinting and Docstrings:** All functions are expected to have clear type hints and docstrings to improve code clarity and maintainability.
*   **Linting and Formatting:** The project uses `ruff` for linting and `black` for code formatting to ensure a consistent coding style.
*   **Testing:** The project will include a suite of `pytest` unit tests to ensure the correctness of individual functions.
*   **Commits:** Commit messages should follow the Conventional Commits specification (`<scope>: <imperative summary>`).