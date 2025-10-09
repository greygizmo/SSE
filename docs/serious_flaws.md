# Serious Flaws Identified in GoSales Engine

## 1. TLS is effectively disabled for Azure SQL connections
The Azure SQL connection helpers append `TrustServerCertificate=yes` to every connection string, which skips certificate validation and makes the pipeline vulnerable to man-in-the-middle attacks whenever it connects to production data sources.

## 2. Holdout validation script persists future data in the curated warehouse
Status: Fixed - the legacy DB-writing entry point has been removed. Holdout evaluation now routes through `gosales.validation.forward`, which reads outcomes without mutating curated tables (`gosales/pipeline/validate_holdout.py`).

## 3. Holdout validation rewrites fact tables without transactional safety
Status: Fixed - `validate_against_holdout` has been replaced with a no-op compatibility shim. There are no transactional writes to `fact_transactions`; users must run `gosales.validation.forward` for holdout analysis (`gosales/pipeline/validate_holdout.py`).

## 4. End-to-end pipeline treats holdout validation as best-effort
Status: Fixed - `gosales.pipeline.score_all` now raises when holdout validation errors or gates fail unless `validation.holdout_required` is disabled (gosales/pipeline/score_all.py).
Previously, `score_all()` swallowed errors from `validate_holdout` and continued reporting success, so no regression gate could fail. The pipeline now fails the run unless holdout gating is explicitly disabled.

## 5. Holdout gate evaluates against labels that are always zero
Status: Fixed - the gate now enriches `icp_scores` with true holdout outcomes sourced via the configured holdout loaders before computing metrics (`gosales/pipeline/validate_holdout.py`, `gosales/validation/holdout_data.py`).

# Serious Issues Identified in GoSales Engine

No outstanding serious issues are currently documented. If a regression or high-severity gap is discovered, add it back to this list with supporting context and file references.
