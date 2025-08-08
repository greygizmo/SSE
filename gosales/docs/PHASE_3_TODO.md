### Phase 3 To-Do (modeling & calibration vs playbook)

- Config & seeds
  - Add modeling section to config: seeds, model grids (LR/LGBM), folds, capacity thresholds. TODO

- Cutoffs & splits
  - Support multiple cutoffs (comma-separated) for rolling-origin CV. TODO
  - Time-aware internal split per cutoff (train/valid) or K-fold time series. TODO

- Champion–challenger
  - Implement LR (elastic-net, standardized) with small grid. TODO
  - Implement LGBM challenger with small grid + early stopping; cap scale_pos_weight. TODO
  - Selection by revenue_lift_top10 with cal-MAE tie-breaker; log decision. TODO

- Calibration
  - Cross-fold or holdout calibration; implement sigmoid (Platt) and isotonic; choose by Brier + cal-MAE. TODO
  - Emit calibration bins CSV per cutoff and final model. TODO

- Metrics & artifacts
  - Compute AUC, PR-AUC, lift@{5,10,20}%, Brier, cal-MAE; revenue-weighted lift. TODO
  - Export gains.csv, calibration.csv, metrics.json, thresholds.csv (top-N%, capacity). TODO
  - Export model artifacts: model, scaler (LR), calibrator, feature_list.json, coef_.csv (LR) or SHAP summaries (LGBM). TODO
  - Model card (JSON): data versions, prevalence, params, seed, selected model. TODO

- CLI
  - `gosales/models/train.py` with flags: `--division`, `--cutoffs`, `--window-months`, `--models`, `--calibration`, `--config`. TODO

- Guardrails
  - Degenerate classifier check (std(p) < 0.01) abort. TODO
  - Overfit/early stop guard; seed determinism. TODO

- Tests
  - Determinism test (same config/seed ⇒ same metrics/hash). TODO
  - Leakage probe (future feature injection ⇒ no AUC gain). TODO
  - Calibration test on synthetic logits. TODO
  - Threshold math correctness for top-N. TODO


