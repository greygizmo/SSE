## Operations Guide

### Environments and config

- Central config at `gosales/config.yaml`. Precedence: YAML < env vars < CLI overrides.
- Key toggles:
  - `features.use_als_embeddings`: join ALS embedding features if present
  - `whitespace.weights`: blend weights for Phase-4 ranker
  - `whitespace.capacity_mode`: `top_percent|per_rep|hybrid`
  - `whitespace.shadow_mode`: when true, also emits legacy heuristic whitespace and overlap metrics
  - `whitespace.challenger_enabled`: enables challenger meta-learner score `score_challenger`

### Typical runs (PowerShell)

```powershell
$env:PYTHONPATH = "$PWD"; python gosales/pipeline/score_all.py
```

Artifacts appear in `gosales/outputs/` including `icp_scores.csv`, `whitespace_<cutoff>.csv`, thresholds, metrics, schema reports, and `run_context_<run_id>.json`.

### Promotion and gates

CI runs:
- Ruff lint, pytest
- Schema validation (`schema_icp_scores.json`, `schema_whitespace*.json`)
- Holdout validation metrics; if `status=fail` build fails
- Drift `alerts.json` is logged as a warning, not a failure

To adjust gates, set thresholds in your validation runner (Phase‑5) or customize `gosales/validation/ci_gate.py` if needed.

### Troubleshooting quick list

- No rows in `whitespace` → ensure models exist and `icp_scores.csv` saved; check runtime alerts in `run_context_*.json`.
- Zero prevalence warning → inspect `fact_transactions` window vs `metadata.division` string; see `gosales/docs/TROUBLESHOOTING.md`.
- Schema failures → open the corresponding `schema_*.json` for missing columns/type issues.
- Determinism → rerun the same commit; ranked outputs checksum in `whitespace_metrics_*.json` should be identical.


