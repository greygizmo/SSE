### Phase 6 To-Do (Config, UX, Observability vs playbook)

- Config & validation
  - Harden `utils/config.py` validation (pydantic‑like schema or stricter dataclass checks). TODO
  - On load, always write `outputs/runs/<run_id>/config_resolved.yaml` snapshot. TODO
  - Document precedence (YAML → env → CLI) in README and ensure code reflects it. TODO

- Run registry & logging
  - Extend `gosales/ops/run.py` to create a run_id and JSONL logger bound to it. TODO
  - Registry (SQLite or JSONL) with: `run_id, started_at, finished_at, cutoff, window, divisions, status, artifacts_path, top_metrics`. TODO
  - Write manifest per run with file paths + SHA256 hashes. TODO

- Streamlit UI (artifact‑driven)
  - Metrics page: pick division+cutoff; show AUC/PR, gains, thresholds; download links. TODO
  - Explainability: LR coefficients or SHAP; per‑customer drilldown. TODO
  - Whitespace: ranked table + filters + CSV download; capacity slicer. TODO
  - Validation: gains, reliability bins, drift summary, scenario grid. TODO
  - UI smoke tests; error boundaries; refresh button. TODO

- Drift & alerts
  - Expose PSI/KS thresholds in UI; show Good/Warn/Alert badges. TODO
  - Write `alerts.json` when thresholds breached; include summary and suggested actions. TODO

- Orchestration
  - Add `pipeline/train_all.py`, `pipeline/rank_whitespace.py` updates, and `pipeline/score_all.py` to accept `--divisions` and read from config. TODO
  - Dry‑run mode that skips heavy compute and only verifies artifacts presence. TODO

- Documentation
  - Update README with Phase 6 (run registry, UI pages, alerts), CLI examples, and artifact contracts. TODO

- Tests
  - Config validation failures for bad keys/types. TODO
  - Run registry entry and manifest contain expected files for a dry run. TODO
  - UI smoke: pages render with mock artifacts; no exceptions. TODO
  - Drift badges: PSI>0.25 shows Alert. TODO


