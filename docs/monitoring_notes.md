# Monitoring Notes

This note summarizes how the monitoring collector discovers validation metrics and assembles pipeline telemetry.

## Validation Metrics Discovery

- Legacy top-level files: `gosales/outputs/validation_metrics.json` and `gosales/outputs/validation_metrics_*.json` remain supported.
- Canonical nested structure: monitoring now crawls `gosales/outputs/validation/<division>/<cutoff>/metrics.json` and related suffixed variants.
- De-duplication and deterministic ordering ensure the most recent artifacts are considered without double counting.

Code reference: `gosales/monitoring/data_collector.py:550`.

## Data Quality Score

- The score now derives from validation status, gate coverage, missing metrics, and alert severity rather than a fixed 99-point fallback.
- Metrics that meet or exceed thresholds keep the score near 100; missing artifacts fall back to 40 and failed gates drive the score toward 0.
- Alerts apply additional penalties (warn/error/critical), and results are clamped to a 0–100 range for UI display.

Code reference: `gosales/monitoring/data_collector.py:61`.

## Lineage

- Lineage rows are sourced from `run_context_*.json` files only; each manifest step contributes status, record counts, duration, and source when present.
- When no manifest is available, the collector logs the condition and returns an empty lineage list instead of a fabricated table.

Code reference: `gosales/monitoring/data_collector.py:665`.

## Resilience

- If no validation artifacts are present, monitoring still emits an informational success alert, but the score drops to its fallback value to signal missing data.
- All monitoring paths use graceful fallbacks to tolerate missing artifacts or database connectivity issues per the repository's resilience guidelines.
