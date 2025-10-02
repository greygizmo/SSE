<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
# SSE Agent Handbook

<<<<<<< ours
This repository powers the GoSales Sales Strategy Engine (SSE) — an end-to-end
pipeline for leakage-safe modeling, whitespace discovery, reporting, and a
Streamlit decision-support UI. This guide explains how autonomous coding agents
should reason about the project, prioritize tasks, and communicate findings.
Treat it as the single source of truth for execution.

---

## 1. Core Principles
1. **Preserve leakage safety.** Every change must respect time-based cutoffs,
   cohort definitions, and deterministic pipelines. Never join, aggregate, or
   sample data using information that would not exist at the decision date.
2. **Be reproducible first, clever second.** Favor transparent, test-backed
   implementations over heuristics that are hard to audit. New behaviors should
   leave breadcrumbs via metrics, manifests, or logs.
3. **Document the intent.** The repo evolves iteratively; write changelog-style
   commit messages, annotate config knobs, and link new flows to docs whenever
   functionality shifts. Future contributors are often non-engineers relying on
   explicit instructions.
4. **Guard the score surface.** Models, calibration artifacts, and whitespace
   ranking feed customer-facing experiences. Ensure backwards compatibility or
   clearly document breaking changes.
5. **Stay tidy and deterministic.** Tests enforce determinism, hash stability,
   and configuration precedence. Mirror those patterns when adding code.

---

## 2. Repository Map & Agent Expectations

| Area | Purpose | Agent Guidance |
| --- | --- | --- |
| `gosales/etl/` | Star schema builder, contracts, parsing. | Keep SQL-alike logic in Python.<br>When adjusting schemas, update contracts, fixtures, and docs (`docs/Sales_Log_Schema.md`). |
| `gosales/features/` | Feature orchestration and caching. | Respect cutoffs and window configs.<br>Add specs to `docs/FEATURES_AND_CONFIG.md` and extend `tests/test_features*.py`. |
| `gosales/labels/` | Label generation and metadata. | Update leakage docs (`docs/LEAKAGE_GAUNTLET.md`) when label definitions shift. |
| `gosales/models/` | Training, calibration, metrics, saved models. | Avoid committing regenerated binaries.<br>Provide reproduction commands instead. |
| `gosales/pipeline/` | CLI orchestration for phases and scoring. | Maintain idempotent entry points with argparse flags and run-context logging. |
| `gosales/validation/` | Holdout and monitoring checks. | Wire new validations into `validation/ci_gate.py` when they affect release criteria. |
| `gosales/ui/` | Streamlit dashboards. | Keep pages artifact-driven.<br>Document notable UI changes in `docs/STREAMLIT_TODO.md`. |
| `scripts/` | One-off orchestration helpers. | Keep scripts small, self-contained, and documented via docstrings/README notes. |
| `docs/` | Architecture, calibration guides, operations. | Refresh diagrams or call out follow-up work whenever flows change. |
| `gosales/tests/` | Deterministic regression suite. | Add or update tests alongside behavior changes.<br>Mark slow cases with `@pytest.mark.integration`. |

Additional directories:
- `gosales/models/<division>_model/` and `gosales/models/prospects/`: **Read
  only** unless the task explicitly involves artifact regeneration. Provide
  instructions instead of checking in new binaries.
- `gosales_curated.db`: Reference only for smoke-testing; never modify or
  commit replacements.
- `reports/`: Contains exports referenced by downstream consumers. Update
  versioned filenames when regenerating to avoid accidental overwrites.

---

## 3. Task Intake Checklist
1. **Read the prompt carefully.** Confirm whether the change touches pipeline
   behavior, docs, or artifacts. When unclear, assume production impact and add
   regression tests.
2. **Scan relevant docs.** Use `rg` to locate existing references (examples:
   `rg "whitespace" gosales/docs`, `rg "rank_whitespace" -g"*.py"`).
3. **Identify affected tests.** Determine the smallest set of tests guaranteeing
   coverage (unit + integration). Prefer targeted modules over global `pytest`.
4. **Plan for non-engineer handoff.** Provide usage notes or TODOs in docs when
   manual steps remain for the user.

---

## 4. Coding Standards
- **Python**
  - 4-space indent, 120-char soft limit, `ruff` + `black` style. Avoid
    try/except around imports.
  - Type hints required for new public functions. Leverage `typing` and
    `pydantic` patterns already in the repo.
  - Use existing utilities (`gosales.utils.logger`, `run_context`) for logging.
  - Feature/label builders must accept cutoff and config parameters — do not
    hardcode dates or windows.
  - Keep randomness controlled via seeded `numpy.random.Generator` or
    `sklearn` `random_state`.

- **SQL / Query-like Logic**
  - Inline SQL lives in `gosales/sql/queries.py`. Follow the established pattern
    of named query builders returning strings with parameter placeholders.

- **Configuration**
  - Edit `gosales/config.yaml` via safe loaders; mirror new keys in
    `docs/FEATURES_AND_CONFIG.md` and add defaults to `gosales/utils/config.py`.
  - Update `gosales/utils/paths.py` when new artifact folders are introduced.

- **Documentation**
  - Prefer Markdown in `docs/`. Include callouts for manual steps and testing
    instructions. Use tables or bullet lists for clarity.
  - Add diagrams in Mermaid (`.mmd`) when describing flows; keep them
    lightweight and textual.

---

## 5. Testing & Quality Gates
1. **Unit tests** — run module-specific tests (`pytest gosales/tests/test_features.py -q`).
2. **Integration / pipeline smoke** — at minimum run
   `pytest gosales/tests/test_phase3_determinism_pipeline.py -q` when modifying
   training or scoring code.
3. **UI smoke** — for UI changes run `pytest gosales/tests/test_ui_smoke.py -q`.
4. **Static analysis** — `ruff check gosales` and `black --check gosales` when
   altering Python files.
5. **Data-dependent tests** — Some tests rely on Parquet fixtures. Do not modify
   fixture contents unless the task explicitly calls for it; update checksums if
   regeneration is unavoidable.

If tests are slow or fail due to environment constraints, record the limitation
and why the change is still safe.

---

## 6. Safe Handling of Artifacts & Secrets
- Never commit raw sales data, credentials, or regenerated models unless
  instructed. Large binary artifacts should be described in docs with commands
  for users to regenerate locally.
- Use environment variables for database credentials. Sample configs belong in
  `.env.example` (create if missing).
- When a change requires new artifacts (e.g., new metrics CSV), prefer to
  produce a tiny synthetic fixture under `gosales/tests/fixtures/`.

---

## 7. Communication & Git Hygiene
- **Commits** follow Conventional Commit format (`feat:`, `fix:`, `docs:`,
  `refactor:`, etc.). Scope segments like `feat(pipeline): add
  rank-normalization toggle` when appropriate.
- **PR preparation**
  - Summaries should highlight business-facing impact first, then technical
    details.
  - Document testing commands and mention affected phases (0–6).
  - Include screenshots (`streamlit run`) when UI behavior changes.

- **Traceability**
  - Update `docs/targets_and_taxonomy.md` when introducing new KPIs or target
    definitions.
  - For monitoring enhancements, reflect changes in
    `gosales/docs/OPERATIONS.md` and `gosales/monitoring/` README sections if
    added.

---

## 8. Guidance for Iterative Improvements
The repo evolves via iterative loops (experiment → adjust config → update docs).
Agents should:
1. **Propose minimal viable changes** before large refactors. Start with config
   flags or doc updates to unblock users.
2. **Surface follow-ups clearly.** If work remains (e.g., need to retrain
   models), leave actionable TODO comments or doc callouts.
3. **Scorekeeping sensitivity.** When modifying scoring or calibration logic,
   confirm that saved metrics (`metrics.json`, `calibration.csv`,
   `gosales/reports/*`) stay consistent or note expected deltas.
4. **Mentor-mode responses.** When providing explanations in Markdown or docs,
   assume the reader is not an engineer. Use step-by-step guidance and include
   commands ready to copy.

---

## 9. Quick Reference Commands
```bash
# Environment setup (Unix)
python -m venv .venv && source .venv/bin/activate
pip install -r gosales/requirements.txt

# Linting & formatting
ruff check gosales
black --check gosales

# Focused tests
pytest gosales/tests/test_features.py -q
pytest gosales/tests/test_phase3_determinism_pipeline.py -q
pytest gosales/tests/test_score_all_pruning.py -q

# End-to-end scoring dry-run
PYTHONPATH=. python gosales/pipeline/score_all.py --help
```

Always ensure commands are prefixed with `PYTHONPATH=.` when running modules.

---

## 10. Escalation Triggers
Stop and document assumptions when:
- A change alters the meaning of labels, cohorts, or capacity constraints.
- Regenerating models or metrics is unavoidable.
- New external dependencies or services are introduced.
- Test fixtures must be replaced (explain why and how to reproduce).

When in doubt, leave a detailed note in the PR description or update the
relevant doc so non-engineer maintainers understand the impact.

=======
> **Mission**: This repository powers the GoSales ICP + whitespace engine. Every change must keep the ETL → feature engineering → modeling → whitespace ranking pipeline deterministic, auditable, and safe to iterate on. Assume the user is iterating quickly and relies on you to keep the project stable and explain what you touch.

---

## 0. Golden Rules
1. **Preserve determinism and reproducibility.** Never introduce non-seeded randomness, time-dependent defaults, or filesystem writes outside documented artifact folders.
2. **Document the why.** Whenever you add behaviour, update relevant docstrings/config comments and cross-reference README or `docs/` when behaviour changes.
3. **Guard data privacy.** Do not check in raw customer data, secrets, or paths that expose local machines. Keep samples sanitized.
4. **Prefer targeted improvements over sweeping refactors.** The owner is iterating; incremental, well-justified edits with clear rollback paths are strongly preferred.
5. **Fail loudly but informatively.** Raise descriptive errors rather than silent fallbacks. When adding validation, include actionable remediation hints.

---

## 1. Repository Map & Scope Notes
- `gosales/`
  - `etl/`: build curated star schema. Changes must maintain schema contracts (`gosales/etl/contracts/`). If you add columns, update manifests and validation.
  - `features/`: feature builders and catalogs. Keep transformations leakage-safe; if you use future-looking data, you must justify the cutoff guard.
  - `models/`: training, calibration, and diagnostics. Respect config-driven workflows—new hyperparameters belong in `config.yaml` with defaults and validation. LightGBM + LR only unless owner requests expansion.
  - `pipeline/`: orchestration CLIs. Ensure CLI arguments remain backwards-compatible. New flags need docs in README + `--help` text.
  - `validation/`: forward validation + drift metrics. Any new metrics should feed alerts/telemetry consistently.
  - `monitoring/`: telemetry collectors—confirm additions are resilient when optional artifacts are missing.
  - `ui/`: Streamlit app. When touching UI, capture screenshots if the change is user-visible.
  - `tests/`: must stay fast (`pytest -q`). Prefer fixture-driven tests in `tests/fixtures/`. Add coverage alongside new code.
  - `docs/`: markdown guides surfaced in Streamlit. Keep consistent tone and link to new functionality.
- `reports/`: PowerBI/other deliverables. If modifying, update documentation and ensure binary assets remain lightweight.
- `scripts/`: entrypoints or utilities; maintain Windows compatibility when editing PowerShell or batch files.

There are no nested `AGENTS.md` files—this guide applies repo-wide. If you add new subdirectories with unique conventions, include an `AGENTS.md` there.

---

## 2. Coding Standards
- **Language**: Python 3.10+. Use `ruff.toml`/`pyproject.toml` for lint/format defaults (`ruff` + `black`). Run `ruff check` and `black` before committing when Python is touched.
- **Style**: 4 spaces, 120-character soft limit, type hints required for new public functions. Return `TypedDict`/`pydantic` models where structure matters.
- **Logging & Errors**: Use `structlog` where available, else `logging` with structured contexts. Never swallow exceptions—wrap external I/O with explicit messaging.
- **Config access**: Use helpers under `gosales.utils.config` instead of raw `yaml.safe_load`. Always validate with `gosales.validation.config` utilities if available.
- **DataFrames**: Preserve column ordering; prefer `assign` + chaining. Guard dtype assumptions with `assert_frame_equal` tests or explicit `astype`.
- **SQL**: uppercase keywords, parameterize queries, and keep schema evolution separate from query logic.

---

## 3. Workflow for Agents
1. **Understand intent**: Skim relevant docs (`README.md`, `docs/`, feature/model guides) before coding. Clarify assumptions in comments or commit messages.
2. **Check scope**: Identify files affected and ensure no conflicting instructions exist. When unsure, leave notes in PR body.
3. **Implement safely**: Prefer small, well-tested changes. Add/adjust tests as part of the same commit when behaviour changes.
4. **Validate**: Run targeted commands:
   - `pip install -r gosales/requirements.txt` (if deps change)
   - `ruff check .`
   - `black .`
   - `pytest -q`
   - Pipeline smoke: `python -m gosales.pipeline.score_all --dry-run` (only if necessary; avoid long-running end-to-end runs).
   Record exact commands + outcomes for the final report.
5. **Update documentation**: README snippets, CLI usage, config tables, and Streamlit docs must reflect functional changes.
6. **Commit**: Follow Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`). Use scoped subjects when possible (e.g., `feat(models): add isotonic fallback messaging`).
7. **PR message (via `make_pr`)**: Summarize *why* and *how*. Include:
   - Context/problem.
   - Implementation highlights.
   - Testing performed.
   - Any follow-up TODOs or manual steps. Avoid trivial details.

---

## 4. Testing & QA Expectations
- **Unit tests**: required for new logic or bugfixes. Place near related modules under `gosales/tests/...` mirroring module path.
- **Integration tests**: mark with `@pytest.mark.integration` and skip by default; document how to run.
- **Determinism**: Seed RNGs via configs (`random_state`, `np.random.default_rng`). Tests should assert deterministic outputs when feasible.
- **Artifacts**: When tests depend on artifacts, store minimal fixtures under `gosales/tests/fixtures`. Avoid large binary files.
- **Performance**: Be mindful of runtime. Optimise DataFrame operations; avoid O(N²) loops on large tables.

---

## 5. Documentation & Communication
- Keep changelog-style bullet in README or dedicated docs when behaviour changes materially.
- Use Markdown tables for config options; include default, type, description, and impact.
- Reference diagrams (Mermaid) when altering data flow.
- For complex changes, add a short `docs/<topic>.md` explainer with examples and validation steps.

---

## 6. Safeguards & Observability
- Whenever you change data contracts, update:
  - `gosales/etl/contracts/*.yaml`
  - Validation checks (`gosales/validation/*`)
  - Monitoring dashboards/collectors if metrics output changes.
- Maintain parity between training and scoring code paths. Any new feature or preprocessing step must exist in both or be explicitly guarded.
- For calibration/model metrics, ensure JSON artifacts include the new fields and downstream consumers (UI, monitoring) can handle them.

---

## 7. When Unsure
- Add inline TODOs with context and link to issues if known (e.g., `# TODO(#123): describe follow-up`).
- Leave conservative defaults. If a choice could break users, gate it behind config opt-ins.
- In the PR body, state assumptions or follow-up questions for the maintainer.

---

By following this handbook, you help keep the GoSales engine reliable, auditable, and easy to iterate on—even when non-engineers orchestrate changes.
>>>>>>> theirs
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
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
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
