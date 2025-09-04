# Repository Guidelines

## Project Structure & Module Organization
- `src/`: ETL and business logic (e.g., `src/ingest/`, `src/transform/`, `src/reporting/`). Entry points typically live in `src/main.py` or `src/pipeline.py`.
- `tests/`: Unit and integration tests; mirrors `src/` (e.g., `tests/transform/test_cleaning.py`).
- `notebooks/`: Exploratory analysis and prototyping. Keep outputs off by default.
- `sql/`: Parameterized, reusable queries for the GoSales datasets.
- `reports/`: Published report assets (e.g., `.pbix/.pbip`) and exports.
- `data/`: Local-only data (`raw/`, `interim/`, `processed/`). Commit only small, non-sensitive fixtures.
- `config/`: Environment and runtime settings (e.g., `settings.yaml`, `.env.example`).

## Build, Test, and Development Commands
- Setup (Windows): `py -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt`
- Lint/format (if configured): `ruff check .` and `black .`
- Run tests: `pytest -q` (single test: `pytest tests/transform/test_cleaning.py::TestCleaning::test_basic -q`)
- Run pipeline locally: `python -m src.pipeline` (or `python src/main.py` if present)

## Coding Style & Naming Conventions
- Python 3.x, 4-space indent, 120-char line length.
- Names: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Tests: files start with `test_*.py`; use clear Arrange–Act–Assert sections.
- SQL: uppercase keywords; one statement per file when lengthy; keep schema changes separate from queries.
- Reports: name as `gosales_<area>_<description>.<ext>` (e.g., `gosales_sales_monthly.pbip`).

## Testing Guidelines
- Framework: `pytest` with optional coverage (`pytest --cov=src --cov-report=term-missing`).
- Aim for high coverage on core transforms and business rules; add fixture samples under `tests/fixtures/`.
- Mark slow/external tests: `@pytest.mark.integration` and skip by default in CI.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (e.g., `feat: add revenue uplift transform`, `fix(transform): handle null SKUs`).
- PRs: include a concise description, linked issues, before/after screenshots for report changes, and notes on data/SQL impacts.
- Keep changes scoped and reversible; update docs/config examples when behavior changes.

## Security & Configuration Tips
- Never commit secrets or raw sensitive data. Use `.env` (checked-in example: `.env.example`).
- Large files belong in `data/` and are git-ignored; commit only minimal, anonymized samples.
- Validate external connections (DB, APIs) via env vars, not hard-coded strings.
