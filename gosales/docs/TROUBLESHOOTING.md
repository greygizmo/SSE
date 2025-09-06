## Troubleshooting

### Division prevalence is zero at scoring

- Symptom: Alert `ZERO_PREVALENCE_UNEXPECTED` and division skipped.
- Checks:
  - `models/<division>_model/metadata.json` has correct `division` casing and `cutoff_date`, `prediction_window_months`.
  - `fact_transactions.product_division` matches metadata division after trim/casefold.
  - There are transactions in `(cutoff, cutoff+window]`.

### Schema validation failing

- Open `schema_icp_scores.json` or `schema_whitespace*.json` and inspect `missing_columns` and `type_issues`.
- Ensure `run_id` is included in whitespace outputs; pipeline inserts when manifest exists.

### Drift/calibration alerts

- `alerts.json` contains warnings about prevalence/calibration drift. Review training `metadata.json` and compare.
- These do not fail CI by default; adjust thresholds if needed.

### Determinism mismatch

- If ranked checksum in `whitespace_metrics_*.json` differs between identical runs, check:
  - non-deterministic sorts (ensure `kind='mergesort'`),
  - environment/library changes,
  - data inputs/order changes.

### ALS/affinity has low coverage

- Ranker down-weights low-coverage signals automatically. Improve data coverage over time or adjust `whitespace.weights`.

### `icp_scores.csv` write fails on Windows

- If `icp_scores.csv` is open in Excel, Windows may lock the file. The scorer now writes a timestamped fallback file in the same folder and logs a warning. Close Excel and re-run if you need the canonical filename.

### `lift_norm` or `als_norm` appear as all zeros

- Scorer now propagates `mb_lift_max/mb_lift_mean` and `als_f*` into ranking. If zeros persist:
  - Reduce basket-lift `min_support` in the features engine for sparse divisions.
  - Increase `features.als_lookback_months` in `config.yaml` to widen the interaction window.
  - Ensure `implicit` is installed for ALS; a TruncatedSVD fallback is provided but requires recent activity.


