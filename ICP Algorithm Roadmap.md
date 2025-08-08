ICP Algorithm Roadmap — Cursor (GPT-5, Ultrathink)
Purpose: concise, high-leverage plan to push intelligence and feature robustness of the ICP/whitespace engine—without over-specifying implementation. Cursor will fill in details using repo context.

Operating Mode (for the Agent)
Reasoning: default reasoning_effort=high (Ultrathink) for planning/features/modeling/refactors; minimal for tiny diffs.

Preambles: before edits, restate goal + list 3–6 steps; after, summarize diffs & artifacts.

Guardrails: no label leakage (features use data ≤ cutoff_date; targets live after the cutoff).

Each PR ships: short changelog, acceptance checks, rollback hint.

What “More Intelligent” Means
Predictive power: higher AUC/PR-AUC & better lift at top-decile on future periods.

Calibration: probabilities behave like probabilities (low binned-MAE).

Stability: similar performance across time splits; no single feature dominates.

Actionability: one whitespace_rank that blends model probability, affinity signals, and expected value.

Phase 0 — Baseline Repro & Contracts (Day 0–1)
Goal: deterministic pipeline; safe data contracts.

Strict dtype & currency/date cleaners; block on PK/null violations.

Extract SKU→Division mapping to a module; unit-test it.

Confirm SQLite fallback; re-run ETL twice → identical row counts.
Outputs: data-contract log, row counts, sample cleaned rows.

Phase 1 — Target/Label Engineering (Day 1–2)
Goal: leakage-safe, business-true labels.

Params: division_name, cutoff_date, prediction_window_months (3–6).

Positive: bought in window after cutoff; negatives: no such purchase.

Cohort flags for analysis: is_new_logo, is_expansion, is_renewal_like.
Outputs: label prevalence by division; window dates; cohort counts.

Phase 2 — Feature Library (Day 2–4)
Goal: robust, time-aware features; light catalog; no leakage.

R/F/M: last-order days; counts & GP over 3/6/12/24-mo windows; averages.

Trajectory: YoY deltas; momentum (windowed slope); volatility (stdev GP).

Cross-division: per-division counts/GP; product diversity; services/training GP.

Lifecycle: account tenure; days since first/last division purchase; gaps.

Industry/size: use enrichment if present; otherwise coarse dummies.

Seasonality: month/quarter dummies; optional peak-season flag.

Market-basket: P(Y|X), lift for common pairs (from tidy transactions).

ALS (optional): implicit factors if available; otherwise skip gracefully.

Emit feature catalog (name/type/short description).
Outputs: feature_catalog_[division].csv, feature coverage %.

Phase 3 — Modeling & Calibration (Day 4–5)
Goal: reliable models per division; calibrated probabilities.

Keep Logistic Regression baseline + LightGBM for accuracy.

Time-aware splits; prefer time-based CV if feasible.

Class imbalance: class weights (or focal-style params); avoid naive downsampling.

Light hyperparameter search; fixed seeds; record chosen params.

Probability calibration (Platt or isotonic) on validation folds.
Outputs: AUC/PR-AUC, calibration CSV, selected params, model in models/{division}_model.

Phase 4 — Whitespace Ranking (Day 5)
Goal: unify signals into one actionable rank.

For each customer × not-owned division, compute:

icp_score (model probability)

lift_z (normalized lift; 0 if unavailable)

als_score (0 if unavailable)

expected_gp proxy (e.g., trailing avg GP × icp_score)

whitespace_rank = w1*icp + w2*lift_z + w3*als + w4*EV (weights in config).

Add nba_reason (short rationale: top features/affinities).
Outputs: outputs/whitespace.csv with columns above + rank order.

Phase 5 — Validation on Future Data (Day 5–6)
Goal: prove forward lift; avoid duplicated ETL.

Refactor shared unpivot helper; reuse across train/holdout.

Backtest with cutoff_date=YYYY-MM-DD, window=3–6; evaluate post-cutoff only.

Emit: gains table (top-K lift), calibration bins, revenue-weighted metrics.
Outputs: outputs/validation/{division}/gains.csv, calibration.csv, summary JSON.

Phase 6 — Config, UX, Observability (Day 6)
Goal: zero-edit reconfig; demo-ready app.

gosales/config.yaml: divisions, cutoff/window, whitespace weights, paths, log level.

Streamlit tabs: Metrics (AUC/gains/calibration), Explainability (SHAP top-N), Opportunities (whitespace table + filters).

Drift checks: feature means/std vs training; soft alerts.
Outputs: working config + UI with metrics and downloads.

Guardrails & Anti-Goals
Never compute features using post-cutoff data.

Don’t add exotic models unless gains/interpretability justify it.

Avoid tight Azure coupling; keep SQLite first-class.

Fail soft: when enrichment/ALS missing, continue without breaking.

Minimal Acceptance Heuristics
Power: +X% lift at top decile vs baseline (set per division).

Calibration: low binned-MAE.

Stability: metrics within tight bands across backtests.

Actionability: whitespace.csv with rank + reason; top-50 pass sniff test.

Config Nits (for the Agent)
config.yaml: divisions_to_train, cutoff_date, prediction_window_months, weights: {icp, lift, als, ev}, log_level.

CLI flags: --division, --cutoff_date, --window_months, --config on train/score/validate.

Starter Tasks (Execute in Order)
Data contracts + SKU map + currency/date normalization.

Phase-1 labels + prevalence report; hard leakage checks.

Phase-2 feature families + feature catalog.

Train & calibrate; export metrics & SHAP CSV.

Unified whitespace rank (graceful degradation when signals missing).

Holdout backtest; emit gains & calibration artifacts.

Introduce config.yaml; surface metrics in UI.