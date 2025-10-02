# GoSales Engine – Agent Operating Guide
This repository powers a division-aware Ideal Customer Profile (ICP) and whitespace engine. The stack ingests raw GoSales transactions, builds a curated star schema, engineers leakage-safe features, trains calibrated models, and surfaces ranked opportunities for sales and customer-success teams. The guidance below keeps iterations safe, deterministic, and explainable—especially important because the maintainer is iteratively improving the system and relies on coding agents for assistance.
---
## 1. Mission, Priorities, and Non‑Negotiables
1. **Protect determinism and leakage-safety.** Always preserve time-cutoff rules, guarded joins, and deterministic sorting in ETL, features, and scoring. Any stochastic process must be seeded and documented.
2. **Keep orchestrations resilient.** CLI entry points (`gosales.pipeline.*`, `gosales.validation.*`, `gosales.whitespace.*`) must continue to run end-to-end even when optional artifacts (ALS, SHAP, telemetry) are missing. Prefer graceful fallbacks over hard failures.
3. **Guard data and secrets.** Never introduce code that reads from non-configured paths, commits large/raw data, or hard-codes credentials. Environment-dependent information lives in `config.yaml`, `.env`, or runtime arguments.
4. **Explain your work.** Update documentation, config examples, and diagnostics when behavior changes. New features must surface in docs or `reports/` if they affect end users.
5. **Support iterative experimentation.** Favor parameterization and configuration switches over hard-coded values so future iterations can toggle features without code changes.
---
## 2. Repository Orientation
| Area | Purpose | Notes |
| --- | --- | --- |
| `gosales/etl/` | Raw → curated pipelines (`build_star`, contracts, assets) | Respect schema contracts and QA artifacts. |
| `gosales/features/` | Feature engineering engine, ALS embeddings, stats exports | Always align feature catalogs with model expectations and zero-fill optional signals. |
| `gosales/labels/` | Leakage-safe label builders and audits | Modes: expansion/all; enforce censoring and denylist rules. |
| `gosales/models/` | Training CLI, calibration, diagnostics | Grids configured via `config.yaml`; exports metrics, gains, model cards. |
| `gosales/whitespace/` & `gosales/pipeline/` | Scoring, ranking, orchestration, cooldown/capacity logic | Deterministic ranking, guard for missing artifacts, run-context manifests. |
| `gosales/validation/` | Forward validation, drift, PSI | Outputs land under `gosales/outputs/validation/…`. |
| `gosales/monitoring/` | Telemetry aggregation (processing rate, division activity) | Must tolerate missing artifacts and log fallbacks. |
| `gosales/ui/` | Streamlit front-end fed by artifacts | Use cached loaders; avoid heavy computation in callbacks. |
| `gosales/tests/` | Extensive pytest suite mirroring pipeline phases | Read `docs/test_suite_overview.md` before touching tests. |
| `docs/` | Deep dives (calibration, features, artifact catalog, targets) | Update when introducing new artifacts or changing definitions. |
| `scripts/` | Maintenance utilities (metrics roll-up, drift snapshots, QA) | Keep CLI help text accurate and idempotent. |
| `reports/` | Published BI assets (Power BI / pbip) | Include before/after screenshots when altered. |
Support assets:
- Config precedence lives in `gosales/config.yaml` and is enforced by `gosales.utils.config`.
- Paths and outputs are centralized in `gosales.utils.paths`; prefer those helpers over manual `Path` concatenation.
- Logging uses `gosales.utils.logger.get_logger`; keep log messages structured and actionable.
---
## 3. Scoping & Design Checklist
Before changing code:
1. **Identify the phase** you are touching (0–6). Review the relevant doc under `docs/` to understand contracts and invariants.
2. **Trace dependencies.** Use `rg` to locate call sites. If a change affects shared helpers (`gosales.utils`, `gosales.ops`, `gosales.sql`), audit all importers and update tests accordingly.
3. **Plan fallbacks.** For database work, rely on `get_curated_connection()` with a fallback to `get_db_connection()`, and respect `database.strict_db` behavior.
4. **Confirm artifact naming.** Outputs must remain consistent (`whitespace_<cutoff>.csv`, `metrics.json`, etc.) unless you also update downstream consumers and documentation.
5. **Consider performance envelopes.** Feature builds and scoring run on large datasets; prefer vectorized operations (`polars`, `pandas`) and streaming where possible. Avoid quadratic loops.
6. **Document toggles.** If adding config knobs, expose defaults in `config.yaml`, describe them in relevant docs, and ensure tests cover on/off states.
---
## 4. Coding Standards & Patterns
- **Python version:** 3.10+. Use `from __future__ import annotations` where practical.
- **Formatting:** `black`/`ruff` with `line-length = 88` (see `gosales/pyproject.toml` and `gosales/ruff.toml`). Run `ruff check gosales` and `black gosales` before committing when you touch Python code.
- **Imports:** Standard library → third-party → local, separated by blank lines. No try/except around imports. Prefer explicit relative imports within the `gosales` package.
- **Typing:** Add type hints for new functions. When working with `polars`/`pandas`, annotate with `pl.DataFrame`, `pd.DataFrame`, etc. Use `Protocol` or `TypedDict` for structured configs when necessary.
- **Configuration:** Access config via dataclasses returned by `gosales.utils.config.load_config`. Do not assume keys exist—prefer `getattr` with defaults and raise `ValueError` with helpful messages when configuration is invalid.
- **DataFrames:** Keep joins leakage-safe by filtering to `cutoff` dates first. For deterministic outputs, sort by stable keys (`customer_id`, `division`) before writing artifacts. Fill nulls for optional signals (ALS, market-basket) with 0 or descriptive placeholders.
- **Logging & Errors:** Use structured logging (`logger.info("message", extra={...})`) when feasible. When raising exceptions, include actionable hints (e.g., missing artifact path, expected columns).
- **CLI:** Build commands with Click; ensure every option has `--help` text and sensible defaults. Validate user inputs early.
- **Randomness:** When randomness is unavoidable (bootstraps, SHAP sampling), surface the seed in outputs and allow overrides via config/CLI.
- **Testing utilities:** Reuse fixtures under `gosales/tests/fixtures/`. Add new fixtures there when you need deterministic sample data.
---
## 5. Testing & Quality Gates
- **Primary command:** `pytest gosales/tests -q`
  - Use targeted runs for large suites, e.g., `pytest gosales/tests/test_phase4_rank_normalization.py -q`.
  - For expensive suites (ALS, SHAP), mark with `@pytest.mark.slow` and gate behind `PYTEST_ADDOPTS="-m 'not slow'"` in CI.
- **Determinism checks:** Many tests assert checksums and sorted outputs. When modifying feature or scoring logic, update corresponding expected artifacts (usually small CSV/JSON fixtures) and explain changes in the PR description.
- **Static analysis:** Run `ruff check gosales` when Python files are touched. Apply `black gosales` to enforce formatting.
- **Manual validation:** When adjusting Streamlit UI or report assets, capture a screenshot via the browser container and store it under `reports/screenshots/` or attach it to the PR.
- **Data fixtures:** Keep new fixtures tiny, anonymized, and committed under `gosales/tests/fixtures/`. Never write to `gosales/outputs/` inside tests; use `tmp_path`.
Document in your final summary which tests ran and why certain suites were skipped (e.g., environment limits).
---
## 6. Documentation, Artifacts, and Observability
- Update `docs/` when behavior, metrics, or configuration options change. Keep tables and enumerations sorted alphabetically for readability.
- Maintain `GoSales_MVP_PRD.md` when strategic objectives or feature definitions evolve.
- If artifact schemas change, update `docs/artifact_catalog.md` and any schema snapshots referenced in tests.
- When adding telemetry fields, reflect them in monitoring docs and ensure the Streamlit UI surfaces them correctly.
- Keep `reports/` synchronized with the production Power BI or pbip sources. Include before/after notes in PRs for report updates.
---
## 7. Collaboration & Delivery Expectations
- **Commits:** Use Conventional Commit prefixes (`feat`, `fix`, `refactor`, `docs`, `test`, etc.). Keep commits scoped and reversible.
- **Pull Requests:** After committing, call the `make_pr` tool with a concise summary, linked issues (if any), testing evidence, and user-facing impacts. Highlight calibration/modeling changes and any expected downstream re-runs.
- **Final Responses:** Follow system instructions—provide a bullet summary with file citations and list every command run with ✅/⚠️/❌ prefixes.
- **Communication:** When behavior changes require rerunning phases (e.g., new feature columns), state which phases must be re-executed and whether historic artifacts need regeneration.
- **Agent etiquette:** Prefer fewer, higher-quality changes over broad refactors. When uncertain, leave TODO comments only with clear context and follow-up steps.
---
## 8. Quick Reference Commands
```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r gosales/requirements.txt
# Run the curated star schema build
PYTHONPATH="$PWD" python -m gosales.etl.build_star --config gosales/config.yaml --rebuild
# Train models for a division
PYTHONPATH="$PWD" python -m gosales.models.train --division Solidworks --cutoffs "2023-06-30,2023-09-30" --config gosales/config.yaml
# Rank whitespace opportunities
PYTHONPATH="$PWD" python -m gosales.pipeline.rank_whitespace --cutoff "2024-06-30" --window-months 6 --config gosales/config.yaml
# Forward validation example
PYTHONPATH="$PWD" python -m gosales.validation.forward --division Solidworks --cutoff 2024-12-31 --window-months 6
# Streamlit UI
PYTHONPATH="$PWD" streamlit run gosales/ui/app.py
```
Stay deliberate, deterministic, and generous with diagnostics—these qualities keep the GoSales Engine trustworthy as it evolves.
