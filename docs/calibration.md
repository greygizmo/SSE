# Probability Calibration & Tuning Guide

This guide explains how Phase 3 training calibrates probabilities, how the new adaptive behavior works, and how to tune it for your data.

## Overview
- Methods: Platt (logistic `sigmoid`) and Isotonic regression.
- Where used:
  - Per-cutoff model selection for LR and LGBM.
  - Final model fit (winner) over the last cutoff.
- Goals: well-calibrated probabilities for downstream planning (capacity, top‑K thresholds, yield).

## Adaptive Behavior (per‑cutoff and final)
- Dynamic CV: `n_splits = min(modeling.folds, #pos_train, #neg_train)`.
  - If `n_splits < 2`, calibration is skipped and the uncalibrated probabilities are used.
  - Diagnostics record `calibration='none'` and a reason (`insufficient_per_class` or `single_class_train`).
- Sparse downgrade: when positives in the training fold are very sparse, isotonic is automatically downgraded to Platt.
  - Threshold: `modeling.sparse_isotonic_threshold_pos`.
- Selection metric during per‑cutoff fitting: lowest Brier score on the validation fold (tiebreakers by existing logic).

## Diagnostics & Artifacts
- File: `gosales/outputs/diagnostics_<division>.json`.
  - `results_grid`: one row per cutoff per model with: `cutoff`, `model`, `auc`, `lift10`, `brier`, `calibration` (`platt`|`isotonic`|`none`), `calibration_reason` (when skipped), plus LR details (`converged`, `n_iter`).
  - Confirms that every cutoff contributed metrics even if calibration was skipped.
- Model cards and metrics JSON remain unchanged aside from reflecting whichever method was applied at final fit.

## Tuning Knobs (config)
Edit `gosales/config.yaml` under `modeling`:
- `folds` (int, default 3)
  - Increase for larger, well‑balanced folds; decrease only if you consistently see `insufficient_per_class` and cannot increase data.
- `calibration_methods` (list)
  - Order doesn’t enforce preference; the implementation evaluates each and picks the best by Brier score.
- `sparse_isotonic_threshold_pos` (int, default 1000)
  - If training positives < threshold, prefer Platt for stability.
- `class_weight` (LR) and `use_scale_pos_weight` + `scale_pos_weight_cap` (LGBM)
  - Influence raw score balance before calibration; helpful for extreme imbalance.

Example:
```yaml
modeling:
  folds: 3
  calibration_methods: [platt, isotonic]
  sparse_isotonic_threshold_pos: 800
  class_weight: balanced
  use_scale_pos_weight: true
  scale_pos_weight_cap: 10.0
```

## Rules of Thumb
- Use isotonic when you have ample positives per fold (e.g., >= 200–300 per fold for smoother reliability curves).
- Prefer Platt for smaller datasets or thin cutoffs; it’s less prone to overfitting than isotonic.
- If you frequently see `single_class_train` or `insufficient_per_class`:
  - Increase `window_months` (wider label horizon), or
  - Reduce `folds` slightly, or
  - Aggregate more events (e.g., relax segment filters) to raise positives.
- Evaluate calibration quality via:
  - Brier score (lower is better),
  - Weighted calibration MAE (`cal_mae` in metrics),
  - Reliability plots (`calibration_<division>.csv`).

## Troubleshooting
- `single_class_train`: all‑negative or all‑positive training fold for a cutoff.
  - Widen the window, or skip calibration (already handled), consider data sufficiency.
- `insufficient_per_class`: per‑class counts too small for `cv>=2`.
  - Similar actions; you can also decrease `folds` as a last resort.
- `calibration_error:<Type>`: estimator raised during calibration.
  - Check logs for the exception; fall back to raw probabilities is already applied; ensure features/labels are sane and not degenerate.

## How Selection Works
- Per‑cutoff: choose the best uncalibrated model by AUC on the validation slice; then evaluate calibration methods and keep the one with the best Brier score for reporting.
- Aggregation across cutoffs: average `lift10`, `brier`, and `auc` by model; select winner by mean `lift10` (tie‑break by Brier).
- Final model: fit on the last cutoff; attempt calibration with dynamic CV; skip gracefully if infeasible.

## Related References
- Configuration: `gosales/config.yaml` → `modeling.*`
- Training implementation: `gosales/models/train.py`
- Diagnostics: `gosales/outputs/diagnostics_<division>.json`

---

## Quick Tuning Recipes

Use these ready-to-copy YAML snippets to adjust calibration for common scenarios. Place them under `modeling:` in `gosales/config.yaml`.

1) Thin cohorts (frequent sparse positives)
```yaml
modeling:
  folds: 3                     # keep moderate folds
  calibration_methods: [platt, isotonic]
  sparse_isotonic_threshold_pos: 2000  # prefer Platt more often
  class_weight: balanced       # helps LR under imbalance
  use_scale_pos_weight: true   # helps LGBM under imbalance
  scale_pos_weight_cap: 10.0
```

2) Highly imbalanced, moderate data volume
```yaml
modeling:
  folds: 3
  calibration_methods: [platt, isotonic]
  sparse_isotonic_threshold_pos: 1000
  class_weight: balanced
  use_scale_pos_weight: true
  scale_pos_weight_cap: 20.0    # allow stronger tilt for LGBM
```

3) Prefer isotonic (ample data per cutoff)
```yaml
modeling:
  folds: 5                      # more folds for smoother reliability
  calibration_methods: [isotonic, platt]
  sparse_isotonic_threshold_pos: 200   # allow isotonic even with fewer positives
  class_weight: none
  use_scale_pos_weight: false
```

4) Fast dev mode (smoke/regression speed)
```yaml
modeling:
  folds: 2                      # reduce CV cost
  calibration_methods: [platt]  # skip isotonic for speed
  sparse_isotonic_threshold_pos: 1000
```

5) Balanced, moderate dataset (default-ish)
```yaml
modeling:
  folds: 3
  calibration_methods: [platt, isotonic]
  sparse_isotonic_threshold_pos: 1000
  class_weight: balanced
  use_scale_pos_weight: true
  scale_pos_weight_cap: 10.0
```

Tips:
- If you regularly see `insufficient_per_class`, try increasing the label window or reducing `folds`.
- If reliability curves look piecewise/steppy, increase `folds` and prefer isotonic (when positives allow).
- Track Brier and `cal_mae` when comparing recipes; prefer the configuration with lower values on validation.
