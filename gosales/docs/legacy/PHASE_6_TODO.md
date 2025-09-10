### Phase 6 To-Do (Config, UX, Observability vs playbook)

- Config & validation
  - Harden `utils/config.py` validation (pydantic‑like schema or stricter dataclass checks). PARTIAL
    - Unknown top‑level keys rejected; sanity checks on weights/thresholds/windows. Type guards added where practical.
  - On load, always write `outputs/runs/<run_id>/config_resolved.yaml` snapshot. DONE
  - Document precedence (YAML → env → CLI) in README and ensure code reflects it. DONE

- Run registry & logging
  - Extend `gosales/ops/run.py` to create a run_id and JSONL logger bound to it. DONE
  - Registry (JSONL) with: `run_id, started_at, finished_at, phase, status, artifacts_path`. DONE
  - Write manifest per run with file paths (+ optional checksums in artifacts JSON). PARTIAL

- Streamlit UI (artifact‑driven)
  - Metrics page: pick division+cutoff; show AUC/PR, gains, thresholds; download links. DONE
  - Explainability: LR coefficients or SHAP; per‑customer drilldown. DONE
  - Whitespace: ranked table + filters + CSV download; capacity slicer. DONE
  - Validation: gains, reliability bins, drift summary, scenario grid; badges + alerts. DONE
  - UI smoke tests; error boundaries; refresh button. PARTIAL (badge/alerts utils tested; page render smoke TBD)

- Drift & alerts
  - Expose PSI/KS thresholds in UI; show Good/Warn/Alert badges. DONE
  - Write `alerts.json` when thresholds breached; include summary and suggested actions. DONE

- Orchestration
  - Add `pipeline/train_all.py`, `pipeline/rank_whitespace.py` updates, and `pipeline/score_all.py` to accept `--divisions` and read from config. PARTIAL (auto‑discovery in place; config‑driven division lists TBD)
  - Dry‑run mode that skips heavy compute and only verifies artifacts presence. TODO

- Documentation
  - Update README with Phase 6 (run registry, UI pages, alerts), CLI examples, and artifact contracts. DONE

- Tests
  - Config validation failures for bad keys/types. DONE (unknown keys + sanity checks)
  - Run registry entry and manifest contain expected files for a dry run. TODO (pending dry‑run)
  - UI smoke: pages render with mock artifacts; no exceptions. PARTIAL (utils smoke added)
  - Drift badges: PSI>0.25 shows Alert. DONE


