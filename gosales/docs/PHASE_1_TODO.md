### Phase 1 To-Do (labels vs playbook)

- Units & windows
  - Use `(customer_id, division)`; read `run.cutoff_date`, `run.prediction_window_months`. TODO

- Modes
  - Implement `expansion` (pre-cutoff activity) and `all` (all known customers). TODO

- Positives
  - Label positive if target-division net GP in window > `gp_min_threshold` (default 0). TODO
  - Optional denylist of SKUs to exclude from target computation. TODO

- Cohorts
  - Compute `is_new_logo`, `is_expansion`, `is_renewal_like` from pre-cutoff activity. TODO

- Censoring
  - If data max(order_date) < window_end, flag `censored_flag=1` and exclude from training; still report. TODO

- Artifacts
  - Write `labels_{division}_{cutoff}.parquet` (customer_id, division, label, window_start, window_end, cohorts, censored_flag). TODO
  - Write `label_prevalence_{division}_{cutoff}.csv` and `cutoff_report_{division}_{cutoff}.json`. TODO

- CLI
  - Add `gosales/pipeline/build_labels.py` with flags: `--division`, `--cutoff`, `--window-months`, `--mode`, `--gp-min-threshold`, `--config`. TODO

- Tests
  - Synthetic tests: positives/returns-only, modes, censoring, one-row-per-pair. TODO

- Guardrails
  - Assert one row per `(customer, division)` and no feature leakage (window-only use). TODO


