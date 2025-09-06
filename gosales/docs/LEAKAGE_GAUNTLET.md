## Leakage Gauntlet

This suite verifies that features and splits are leakage-safe and that “too good to be true” metrics do not persist under stress tests.

Implemented checks

- Group overlap audit (GroupKFold by `customer_id`): ensures no customer appears in both train and validation.
- Feature-date audit: verifies latest event dates contributing to features are <= cutoff.
- Static scan: detects banned time calls that may read “now” during feature construction.
- Shift-14 scaffold: optional training at `cutoff-14d` and comparison vs baseline; flags suspicious improvements.
- Top-K ablation: ranks features by importance and (optionally) retrains after dropping top-K; flags suspicious improvements.

Run

```powershell
$env:PYTHONPATH = "$PWD"; python -m gosales.pipeline.run_leakage_gauntlet --division Printers --cutoff 2024-12-31 --window-months 6 --no-static-only --run-shift14-training --shift14-eps-auc 0.01 --shift14-eps-lift10 0.25
```

Top-K ablation example

```powershell
$env:PYTHONPATH = "$PWD"; python -m gosales.pipeline.run_leakage_gauntlet --division Printers --cutoff 2024-12-31 --no-static-only --run-topk-ablation --topk-list 10,20
```

Artifacts (gosales/outputs/leakage/)

- `leakage_report_<division>_<cutoff>.json` – consolidated PASS/FAIL + artifact paths
- `fold_customer_overlap_<division>_<cutoff>.csv` – per-fold overlaps (must be zero)
- `feature_date_audit_<division>_<cutoff>.csv` – latest event date per feature (must be <= cutoff)
- `static_scan_<division>_<cutoff>.json` – banned time call findings
- `shift14_metrics_<division>_<cutoff>.json` – Shift-14 prevalence and (optional) metric comparison
- `ablation_topk_<division>_<cutoff>.csv` – ranked features by importance
- `ablation_topk_<division>_<cutoff>.json` – Top-K drop summary and PASS/FAIL when training is run

Configuration

- Thresholds for Shift-14 are set in `gosales/config.yaml` under `validation`:

```
validation:
  shift14_epsilon_auc: 0.01
  shift14_epsilon_lift10: 0.25
```

- Override at runtime with `--shift14-eps-auc` and `--shift14-eps-lift10`.

- Thresholds for Top-K ablation are set under `validation`:

```
validation:
  ablation_epsilon_auc: 0.01
  ablation_epsilon_lift10: 0.25
```

- Override at runtime with `--ablation-eps-auc` and `--ablation-eps-lift10`.

Failure behavior

- The CLI exits non-zero if any check fails (overlap/date/static). CI can use the exit code to fail a build.
