### Streamlit App To-Do (Phases 0–6 alignment)

Goal: Align the Streamlit UI with pipeline artifacts and UX introduced in Phases 0–6. Cover discoverability, robustness, performance, and observability.

---

Foundations & Navigation
- [ ] Unify navigation: Overview, Metrics, Explainability, Whitespace, Validation, Runs (registry)
- [ ] Global selectors: division/cutoff inputs where applicable; defaults to latest artifacts
- [ ] Config thresholds: load PSI/KS/cal-MAE thresholds once and expose in state
- [ ] Robust artifact path resolution with clear empty/placeholder states and helpful messages
- [ ] Consistent theming + branding (logo, colors, icons)

Runs (Registry, Observability)
- [ ] Read `outputs/runs/runs.jsonl` and list runs with phase, status, timestamps
- [ ] Run detail panel: show `manifest.json` file paths with copy/download actions
- [ ] Link UI pages to a selected run (i.e., auto-fill division/cutoff/artifact paths)
- [ ] Show `config_resolved.yaml` (expandable) for the selected run
- [ ] Surface dry-run entries distinctly; hide non-existent artifacts

Metrics (Phase 3)
- [ ] Training metrics: load `metrics_<division>.json`; display AUC/PR-AUC/Brier, lifts, selection
- [ ] Calibration CSV: plot mean predicted vs fraction positives with download
- [ ] Gains CSV: bar chart + table + download
- [ ] Thresholds CSV: table for top‑K thresholds + download

Explainability (Phase 3)
- [ ] SHAP global CSV (if present): sortable table and bar chart of mean-abs SHAP
- [ ] LR coefficients CSV (if present): sortable table
- [ ] SHAP sample CSV (if present): top N rows with download
- [ ] Helper text/tooltips for interpretation

Whitespace (Phase 4)
- [ ] Ranked table from `whitespace_<cutoff>.csv` with column filters/search
- [ ] Explanations from `whitespace_explanations_<cutoff>.csv` (join or side-by-side)
- [ ] Capacity slicer: display thresholds (`thresholds_whitespace_<cutoff>.csv`)
- [ ] Summary KPIs from `whitespace_metrics_<cutoff>.json`: capture@K, division shares, stability, coverage, weights
- [ ] Structured log preview from `whitespace_log_<cutoff>.jsonl`

Validation (Phase 5)
- [ ] Gains (holdout) from `gains.csv` and calibration bins from `calibration.csv`
- [ ] Scenarios table: `topk_scenarios_sorted.csv` (contacts, capture, precision, rev_capture, CIs)
- [ ] Segment performance table: `segment_performance.csv`
- [ ] Drift summary from `drift.json`: weighted PSI(EV vs holdout GP), KS(p_hat train vs holdout), per-feature PSI
- [ ] Metrics detail from `metrics.json`: AUC, PR-AUC, Brier, cal-MAE, capture grid, drift_highlights
- [ ] Downloads for all validation artifacts

Badges & Alerts (Phase 6)
- [ ] Badges for cal-MAE, PSI(EV vs GP), KS(train vs holdout) with thresholds from config
- [ ] Load and render `alerts.json` if present; show alert items and threshold values
- [ ] Inline help and “What this means” popovers for each badge

Error Boundaries & UX Guardrails
- [ ] Try/except around all file loads; show st.info/warning with next actions
- [ ] Validate artifact schemas lightly; fail gracefully and log
- [ ] Refresh button to re-read artifacts without restarting app

Performance & Caching
- [ ] Cache artifact reads with `st.cache_data` (versioned by file hash/mtime)
- [ ] Avoid loading large CSV/Parquet until referenced by user selections
- [ ] Column type enforcement and memory reduction (downcasting numerics)

Multi-division Support
- [ ] Division selector: discover from `models/*_model` and/or `etl/sku_map`
- [ ] Cutoff selector: discover from artifacts present in `outputs/validation/<division>/`
- [ ] Handle single-division deployments cleanly

Testing & QA
- [ ] UI smoke test: render all pages with mock artifacts (no exceptions)
- [ ] Badge unit test: thresholds produce correct OK/ALERT states
- [ ] Link check: all download buttons point to existing files
- [ ] Scenario math sampling: verify scenario CSV columns exist if file present

Docs
- [ ] README section: UI overview, page descriptions, thresholds, badges, alerts
- [ ] Screenshots for each page (optional)
- [ ] Troubleshooting guide (missing artifacts, permissions, caching)

Acceptance Criteria
- [ ] All listed pages render with existing sample artifacts without errors
- [ ] Badges reflect config thresholds; alerts render when `alerts.json` present
- [ ] Downloads work for every artifact surfaced
- [ ] Registry page provides run-level traceability to artifacts and config snapshot
- [ ] Caching reduces reload latency while reflecting file changes via refresh


