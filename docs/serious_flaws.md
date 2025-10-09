#1. build_als builds its item-name index with original values but looks items up with str(name), so any non-string SKU (e.g., None, integers) will trigger a KeyError and abort whitespace ALS generation for those divisions.

Resolved 2025-10-09: ALS item identifiers are now canonicalized with deterministic typed keys, so non-string SKUs (including None/NaN) map safely during matrix construction and recommendation output.

#2. validate_against_holdout assumes fact_transactions already exists and immediately reads it; on fresh or SQLite builds the table is absent, so the holdout validation crashes before it can restore state or emit metrics (see failing test run).

Resolved 2025-10-09: The compatibility shim now locates the latest icp_scores artifact, runs validate_holdout, and skips gracefully when prerequisites are missing, so legacy holdout gates no longer fail when fact_transactions is absent.

#3. rank_whitespace only caps fallback ALS scores inside the low-coverage branch; when overall ALS coverage is above the threshold, pure assets rows can keep their raw percentile (often ≥ genuine ALS rows), causing fallback accounts to outrank real embeddings (demonstrated by the failing weight-scaling test).

Resolved 2025-10-09: Fallback ALS rows are now clipped below the strongest transaction embeddings regardless of coverage, preserving deterministic ordering while retaining positive scores for assets/item2vec fallbacks.

#4. The same function tries to fill zero-ALS rows with item2vec scores, but those rows remain flagged as zero-signal (_als_signal_strength never updated). Subsequent weighting treats them as uncovered and pushes their ALS percentile back to the floor, so the advertised item2vec fallback never surfaces in the output (also exposed by the failing test).

#5. When segment-based weighting is enabled, rank_whitespace writes segment scores but still references champion_score—which is only defined in the non-segment path—so any run with segment_columns configured will raise an UnboundLocalError before returning results.
