
# PHASE 5 — Forward Validation / Holdout (Cursor GPT‑5 Playbook)

**Audience:** Cursor GPT‑5 coding agent.  
**Mode:** Use *Ultrathink*; evaluate **frozen** models on a true **future** window. Quantify capture, calibration, and drift with confidence intervals.

---

## North Star
With models frozen at a cutoff (e.g., `2024-12-31`), measure how well the system would have performed in the holdout window (e.g., `Jan–Jun 2025`): **top‑K capture**, **revenue lift**, and **probability reliability**.

---

## Operating Rules
- No retraining on holdout.  
- Use same label contracts and eligibility as Phases 1 & 4.  
- Include uncertainty via bootstrap CIs.  
- Produce decision‑ready scenario tables for capacity/threshold picks.

---

## Success Criteria
- Per‑division validation artifacts: `metrics.json`, `gains.csv`, `calibration.csv`, `drift.json`, `topk_scenarios.csv`.  
- 95% CIs for key metrics.  
- Drift diagnostics computed (feature PSI, score KS, SHAP drift if applicable).  
- Tests green (windowing, censoring, bootstrap determinism, drift, calibration).

---

## Evaluation Frame
1) Load **frozen** model, calibrator, and feature list for the chosen cutoff.  
2) Build features **≤ cutoff** for all customers; score to `p_hat`.  
3) Join **holdout** labels (same logic as Phase 1) and EV proxy.  
4) Apply **eligibility** rules to form candidate set.  
5) Persist `validation_frame.parquet` for reproducibility.

---

## Metrics
- **Ranking**: AUC, PR‑AUC, gains/lift by decile.  
- **Business**: top‑K capture (5/10/20%), revenue‑weighted capture, expected GP @ capacity, precision@K.  
- **Calibration**: Brier, cal‑MAE, reliability bins (10–20).  
- **Stability**: by cohort/industry/size/region segments.

---

## Confidence Intervals
- **Block bootstrap by customer** (1,000 resamples). Report 95% CIs for capture@K, revenue capture, Brier, cal‑MAE, precision@K. Seeded for determinism.

---

## Drift Diagnostics
- **Feature drift**: PSI (or Jensen‑Shannon) between train and holdout for key features; flag PSI > 0.25.  
- **Score drift**: KS on `p_hat` (train vs holdout); flag KS > 0.15.  
- **SHAP drift**: compare top features and mean |SHAP| between periods (LGBM only).

---

## Scenarios (capacity & thresholds)
- Grid of top‑N% and per‑rep capacities.  
- For each scenario: `contacts, precision, recall (capture), expected_GP, realized_GP (historical), 95% CI`.  
- Rank by expected GP if calibration is strong; otherwise by capture@K.

---

## CLI Entrypoint
```bash
python -m gosales.validation.forward   --division "Solidworks"   --cutoff "2024-12-31"   --window-months 6   --capacity-grid "5,10,20"   --bootstrap 1000   --config gosales/config.yaml
```

---

## Artifacts
- `validation/{division}/{cutoff}/metrics.json`  
- `validation/{division}/{cutoff}/gains.csv`  
- `validation/{division}/{cutoff}/calibration.csv`  
- `validation/{division}/{cutoff}/drift.json`  
- `validation/{division}/{cutoff}/topk_scenarios.csv`

---

## Guardrails
- Censoring: if holdout not fully covered, flag and exclude; log counts.  
- Base‑rate collapse: warn if prevalence < 0.2% or > 50%.  
- EV outliers: cap at p95; log #capped.  
- Segment failure: if any segment’s top‑decile capture < baseline by > 5 pts, flag.

---

## Module Skeletons
```python
def build_validation_frame(division, cutoff, cfg): ...
def gains_and_lift(df): ...
def calibration_bins(df, n_bins=20): ...
def bootstrap_ci(metric_fn, df, n=1000, seed=42): ...
def drift_report(train_sample, holdout_sample): ...
def scenarios(df, capacities, cost): ...
```

---

## Tests
- Window integrity: no post‑cutoff features.  
- Censoring behavior.  
- Bootstrap determinism: same seed → stable CIs.  
- Drift smoke: injected shift triggers PSI flag.  
- Calibration sanity: synthetic sigmoid recovers low Brier.  
- Scenario math: correct counts and expected GP.

---

## Acceptance Checklist
- [ ] Full artifact set written per division.  
- [ ] 95% CIs reported for key metrics.  
- [ ] Drift computed; thresholds applied; warnings surfaced.  
- [ ] Scenario grid present and sensible.  
- [ ] Deterministic; tests green.

---

## Final Notes to the Agent
Keep holdout **sacred**—do not use it for tuning. If results underwhelm, propose changes for the **next** cutoff and log the rationale in a validation report.
