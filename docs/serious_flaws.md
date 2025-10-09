# Serious Flaws Identified in GoSales Engine

## 1. TLS is effectively disabled for Azure SQL connections
The Azure SQL connection helpers append `TrustServerCertificate=yes` to every connection string, which skips certificate validation and makes the pipeline vulnerable to man-in-the-middle attacks whenever it connects to production data sources.

## 2. Holdout validation script persists future data in the curated warehouse
Status: Fixed - the legacy DB-writing entry point has been removed. Holdout evaluation now routes through `gosales.validation.forward`, which reads outcomes without mutating curated tables (`gosales/pipeline/validate_holdout.py`).

## 3. Holdout validation rewrites fact tables without transactional safety
Status: Fixed - `validate_against_holdout` has been replaced with a no-op compatibility shim. There are no transactional writes to `fact_transactions`; users must run `gosales.validation.forward` for holdout analysis (`gosales/pipeline/validate_holdout.py`).

## 4. End-to-end pipeline treats holdout validation as best-effort
`score_all()` always swallows errors from `validate_holdout` and keeps reporting success, so no regression gate can ever fail a release run. This defeats the purpose of the holdout guardrails the repository documents.

## 5. Holdout gate evaluates against labels that are always zero
The holdout gate reads `bought_in_division` directly from `icp_scores.csv`, but those values come from the inference feature matrix and reflect whether a customer bought before the cutoff. During scoring there are no post-cutoff purchases, so the column is uniformly zero and every metric the gate computes is meaningless.

# Serious Issues Identified in GoSales Engine

No outstanding serious issues are currently documented. If a regression or high-severity gap is discovered, add it back to this list with supporting context and file references.
