### Phase 3 To-Do (modeling & calibration vs playbook)

- Config & seeds
  - Add modeling section to config: seeds, model grids (LR/LGBM), folds, capacity thresholds. DONE

- Cutoffs & splits
  - Support multiple cutoffs (comma-separated) for rolling-origin CV. DONE
  - Time-aware internal split per cutoff (train/valid) or K-fold time series. DONE (recency-aware split w/ stratified fallback)

- Champion–challenger
  - Implement LR (elastic-net, standardized) with small grid. DONE
  - Implement LGBM challenger with small grid + early stopping; cap scale_pos_weight. DONE (small grid)
  - Selection by revenue_lift_top10 with cal-MAE tie-breaker; log decision. DONE

- Calibration
  - Cross-fold or holdout calibration; implement sigmoid (Platt) and isotonic; choose by Brier + cal-MAE. DONE
  - Emit calibration bins CSV per cutoff and final model. DONE (final model)

- Metrics & artifacts
  - Compute AUC, PR-AUC, lift@{5,10,20}%, Brier, cal-MAE; revenue-weighted lift. PARTIAL (AUC, PR-AUC, Brier, lift@K, rev_lift@K, cal-MAE implemented for final)
  - Export gains.csv, calibration.csv, metrics.json, thresholds.csv (top-N%, capacity). DONE (final model)
  - Export model artifacts: model, scaler (LR), calibrator, feature_list.json, coef_.csv (LR) or SHAP summaries (LGBM). DONE (with SHAP guarded if not installed)
  - Model card (JSON): data versions, prevalence, params, seed, selected model. DONE

- CLI
  - `gosales/models/train.py` with flags: `--division`, `--cutoffs`, `--window-months`, `--models`, `--calibration`, `--config`. DONE

- Guardrails
  - Degenerate classifier check (std(p) < 0.01) abort. DONE
  - Overfit/early stop guard; seed determinism. PARTIAL (seed set)

- Tests
  - Determinism test (same config/seed ⇒ same metrics/hash). TODO
  - Leakage probe (future feature injection ⇒ no AUC gain). PARTIAL (drop_leaky_features helper + tests)
  - Calibration test on synthetic logits. DONE (bins + MAE sanity)
  - Threshold math correctness for top-N. DONE


