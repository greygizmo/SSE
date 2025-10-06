# SAFE Mode Review Playbook

This playbook captures the checks to run before removing the SAFE designation from a division (for example, Solidworks). Follow these steps so the next pipeline execution preserves leakage guards and meets validation gates.

## 1. Capture the historical reason
- Pull the original SAFE enablement decision from `gosales/docs/legacy/TODO_after_gauntlet_pass.md` or the associated PR/incident log.
- Summarize which adjacency families or recency windows were implicated and whether the SAFE policy was compensating for known leakage risks or noisy behavior.

## 2. Regenerate comparison models
1. Ensure the latest configuration and data refresh are in place:
   ```bash
   PYTHONPATH="$PWD" python -m gosales.etl.build_star --config gosales/config.yaml --rebuild
   PYTHONPATH="$PWD" python -m gosales.labels.build --division Solidworks --config gosales/config.yaml
   PYTHONPATH="$PWD" python -m gosales.features.build --division Solidworks --config gosales/config.yaml
   ```
2. Train paired models on identical cutoffs:
   ```bash
   PYTHONPATH="$PWD" python -m gosales.models.train --division Solidworks --config gosales/config.yaml --safe-mode
   PYTHONPATH="$PWD" python -m gosales.models.train --division Solidworks --config gosales/config.yaml --variant full
   ```
   - Retain the SAFE run output as the control. The `--variant full` flag forces the adjacency-rich configuration that SAFE normally disables.

## 3. Run adjacency ablations
- Execute the CI gate check locally to compare SAFE vs. Full:
  ```bash
  PYTHONPATH="$PWD" python -m gosales.validation.ci_gate --division Solidworks --config gosales/config.yaml --compare safe,full
  ```
- Inspect the produced `ablation_summary.csv` and confirm whether SAFE still wins by ≥0.005 AUC. If SAFE is still better, removing the designation will fail CI.

## 4. Inspect feature leakage signals
- Review the generated `feature_catalog.json` and `label_audit/` prevalence reports for Solidworks.
- Validate that high-weight features (SHAP or gains) no longer rely on adjacency, expiring-asset tails, or short recency windows that previously required SAFE masking.

## 5. Validate downstream scoring
1. Score with both artifacts:
   ```bash
   PYTHONPATH="$PWD" python -m gosales.pipeline.score_all --divisions Solidworks --config gosales/config.yaml --skip-train
   ```
   - Before running, drop the prior Solidworks model from `gosales/outputs/models/` and symlink either the SAFE or Full run so the scorer picks the intended artifact.
2. Compare holdout metrics under `gosales/outputs/validation/` and confirm the Full variant does not regress precision, lift, or drift guards.

## 6. Decide and document
- If the Full variant matches or exceeds SAFE across CI gate, holdout, and leakage inspections, update `gosales/config.yaml` to remove Solidworks from `modeling.safe_divisions` in the same PR.
- Document the rationale in `docs/` (append to this file or a change log) and note which artifacts need reruns (Phase 0–5 for Solidworks) so consumers know a new baseline is required.

Following this checklist keeps the SAFE designation decision auditable and ensures Solidworks only reverts to the Full model when it is genuinely safe to do so.
