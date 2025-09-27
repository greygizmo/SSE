# Segments and Outputs

This document describes how the GoSales Engine now treats three customer segments and the artifacts produced for each.

## Segments
- Warm: Recent transactional activity (`rfm__all__tx_n__12m > 0`). Targeted primarily for Sales.
- Cold: No recent transactions but owns assets (`assets_active_total > 0` or `assets_on_subs_total > 0`). Targeted for Customer Success and Lifecycle Marketing.
- Prospects: No transaction history and no assets; targeted for Demand Generation and top-slice to Sales when capacity allows.

## Key Features and Policies
- Per-division ownership flags at cutoff (`owns_assets_div_<division>`) and former-owner flags (`former_owner_div_<division>`).
- Per-division days since expiration for re-activation (`assets_days_since_last_expiration_div_<division>`).
- Eligibility: `exclude_if_active_assets`; `reinclude_if_assets_expired_days` (e.g., 365).
- Embeddings:
  - ALS embeddings when available from transactions.
  - Assets-based i2v fallback (TruncatedSVD on `assets_rollup_*` matrix) when ALS coverage is low or absent (cold/prospects).

## Artifacts (outputs/)
- `whitespace.csv`: Global ranked opportunities across segments.
- `whitespace_warm.csv`, `whitespace_cold.csv`, `whitespace_prospect.csv`: Ranked per-segment.
- `whitespace_selected.csv`: Capacity-selected global list.
- `whitespace_selected_warm.csv`, `whitespace_selected_cold.csv`, `whitespace_selected_prospect.csv`: Capacity-selected per-segment lists.
- `assets_join_metrics.json`, `assets_unmatched.csv`: QA for internalid-first join.

## UI (Streamlit)
- Whitespace tab: now surfaces capacity-selected lists and per-segment tabs for quick download.
- ICP Scores: still available with filters; segments can be derived with the same definitions.

## Notes
- Industry fuzzy matching is disabled by default for scalability; a cache path can be used to apply curated enrichments.
- Training for Solidworks uses 2024-06-30 cutoff to preserve 2025 data for forward validation.
