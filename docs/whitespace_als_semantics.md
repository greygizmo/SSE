# Whitespace ALS Semantics (Updated)

This note documents the current behavior of ALS-derived signals within Phase 4 (Whitespace Ranking), including blending and coverage fallbacks. The goal is to preserve deterministic ordering and leakage safety while maximizing coverage when embeddings are sparse.

## Blending
- Mixed-signal rows (transaction ALS present, and optionally assets ALS) use a normalized blend of `txn_als_norm` and `assets_norm` per `whitespace.als_blend_weights`.
- Asset-only rows (no transaction ALS signal, assets ALS present) now take the full `assets_norm` without down-weighting. This preserves the documented fallback guarantee for cold accounts.

## Coverage Fallbacks
- When ALS coverage falls below `whitespace.als_coverage_threshold`, zero-signal rows may receive imputed `als_norm` from:
  - assets ALS percentiles, and/or
  - item2vec similarity when `features.use_item2vec` is enabled or vectors are present.
- To protect ordering, imputed `als_norm` for zero-signal rows is clipped below the strongest genuine transaction-backed ALS value in the cohort. Zero-signal accounts cannot leapfrog customers with real ALS embeddings.

## Configuration Knobs
- `whitespace.als_blend_weights`: `[transaction ALS, assets ALS]` blend weights; normalized internally.
- `whitespace.als_coverage_threshold`: minimum fraction of rows with ALS signal before fallbacks engage.
- `features.use_item2vec`: enables item2vec-based imputation when vectors are available.

## Determinism and Audits
- All operations are vectorized and leverage deterministic percentile/clip operations.
- Diagnostics write coverage metadata (`als`, `als_txn`, `als_assets`, `affinity_source_column`) to the whitespace log where applicable.

## Validation
- Tests assert that:
  - asset-only rows inherit `assets_norm` (no blend penalty), and
  - zero-signal rows receiving imputed ALS do not outrank transaction-backed ALS rows.

For detailed artifact context, see `docs/artifact_catalog.md` (Phase 4 section).
