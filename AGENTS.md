# SSE Agent Handbook

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

