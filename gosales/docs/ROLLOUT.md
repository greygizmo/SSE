## Rollout Plan (Phase‑4)

### 1) Shadow mode

- Enable `whitespace.shadow_mode: true` in `gosales/config.yaml`.
- Run pipeline; compare `whitespace_<cutoff>.csv` (champion) with `whitespace_legacy_<cutoff>.csv` (legacy heuristic).
- Inspect `whitespace_overlap_<cutoff>.json` for Jaccard overlap at top‑10%.

### 2) Business preview

- Share top‑N per division with selected AEs.
- Gather feedback on explanation clarity and account relevance.

### 3) Cutover

- Switch downstream consumers to `whitespace_<cutoff>.csv`.
- Keep shadow artifacts for one cycle to monitor stability.

### 4) Post‑cutover monitoring

- Watch `alerts.json`, `whitespace_metrics_*`, and validation artifacts in CI.
- Adjust `whitespace.weights`, capacity mode, and feature toggles as needed.


