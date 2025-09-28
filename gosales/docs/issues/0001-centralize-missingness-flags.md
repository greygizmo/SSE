Title: Refactor missingness flags: centralize NaN fill and mask capture

Status: Open
Date: 2025-09-28
Labels: tech-debt, enhancement, data-integrity

Problem
- The current pipeline contains many per-column `.fillna(...)` calls during feature construction. Even with the recent fix (compute `_missing` from a just-in-time mask and defer the global fill), any columns that are locally filled earlier by design won’t receive `_missing` indicators, leading to inconsistent missingness semantics across feature families.

Goal
- Unify missingness handling so that `_missing` flags consistently reflect original nulls for all engineered columns, while preserving downstream type stability and model compatibility.

Proposed Approach
1) Stage feature construction without global fills. Avoid scattered `.fillna(...)` except where required for safe arithmetic; where unavoidable, gate behind a config toggle or use intermediate variables to preserve original NaNs.
2) Capture a comprehensive NaN mask for all engineered columns after features are assembled.
3) Generate `_missing` flag columns from this mask.
4) Apply a centralized final fill pass with family-specific defaults (e.g., numeric 0.0, durations 999, strings "missing", small-int to Int8) controlled by config.
5) Remove or gate redundant inline fills to rely on the centralized pass.

Acceptance Criteria
- `_missing` flags exist for all non-key, non-target columns and correctly reflect pre-fill NaNs.
- Global centralized fill runs after `_missing` generation and produces stable dtypes used in training.
- Backward compatibility: behind `features.centralized_fill_enable` (default false). With flag off, behavior matches current pipeline.
- Tests cover: (a) base features, (b) derived post-assembly features, (c) string/categorical features, (d) windowed aggregates. Customers without transactions get `_missing == 1`; active buyers remain 0.

Scope / Affected Files
- `gosales/features/engine.py` (feature assembly, centralized fill and mask capture)
- `gosales/utils/config.py` and `gosales/config.yaml` (add `features.centralized_fill_enable`, optional per-family fill defaults)
- `gosales/tests/test_features.py` and any phase tests asserting feature presence/typing

Risks & Mitigations
- Semantics drift: inline fills currently define some ratios/logs; mitigate by preserving safe arithmetic via temporary variables and capturing masks pre-fill.
- Performance/memory: mask for wide frames; mitigate by boolean dtype and avoiding copies where possible.
- Type invariants: ensure final centralized fill enforces expected dtypes.

Testing Plan
- Add parameterized tests asserting `_missing` correctness across representative feature families.
- Golden stats consistency: winsor caps, null counts before/after centralization.
- CLI smoke (build + checksum); full test suite.

Rollout Plan
- Phase 1: Implement under `features.centralized_fill_enable=false` and run A/B in CI.
- Phase 2: Flip default to true after validation; keep escape hatch for rollback.

Notes
- Related work: PR fixing just-in-time mask and deferred global fill for `_missing` columns.

