### Phase 4 To-Do (Whitespace Ranking / Next‑Best‑Action vs playbook)

- Config & setup
  - Add `whitespace` section to `gosales/config.yaml` with: weights, normalization mode, eligibility rules, capacity modes, EV cap pctl, ALS coverage thresholds, diversification/bias guard thresholds. DONE
  - Snapshot resolved config; log settings at run start. TODO

- Candidates & eligibility
  - Build candidate set per `(customer, division, cutoff)` excluding divisions already owned pre‑cutoff (or last N months, configurable). PARTIAL (derived from div-scope tx_n/recency if present)
  - Enforce region/territory allow‑list, DNC/legal/compliance holds, and open‑deal exclusion; log counts per rule. PARTIAL (per-rule counts captured if columns present)
  - Deterministic customer/division ordering for reproducible IDs. TODO

- Signals
  - p_icp (primary): load Phase‑3 calibrated model per division; score candidates; compute per‑division percentile `p_icp_pct`. PARTIAL (model scoring + pct wired)
  - Affinity lift: pre‑cutoff market‑basket rules; compute `lift_max`, `lift_mean` per candidate; normalize (percentile/z) → `lift_norm`. PARTIAL (engineered affinity wired)
  - ALS similarity: reuse Phase‑2 embeddings; compute similarity to target division; quantile‑normalize → `als_norm`; if unavailable, set 0 and flag coverage. PARTIAL (ALS fallback via centroid or `als_sim_division` computed in features build)
  - Expected value proxy (EV): segment medians (industry/size/region) blended with global; cap at p95; normalize → `EV_norm`. PARTIAL (segment blend + cap + normalization)

- Normalization & comparability
  - Implement per‑division percentile normalization (default) with option for pooled recalibration; verify approx. uniform per division. DONE (pooled mode added; pooled normalization recomputes scores using preserved per‑division coverage‑adjusted weights)
  - Graceful degradation: if a signal missing (e.g., ALS), set to 0, reduce weight if below coverage threshold; log weight adjustments. DONE (dynamic weight scaling by coverage, renormalized)

- Scoring & ranking
  - Champion blend (static): `score = w1*p_icp_pct + w2*lift_norm + w3*als_norm + w4*EV_norm` with defaults 0.60/0.20/0.10/0.10 and per‑division overrides. DONE (configurable weights)
  - Tie‑breakers: higher `p_icp`, higher `EV`, fresher activity, `customer_id` asc; deterministic stable sort. DONE (p_icp/EV/customer_id asc)
  - Optional challengers: meta‑learner or pairwise LTR on [p_icp, lift, als, EV] (behind flag). TODO

- Business‑rule gating & capacity
  - Apply gating AFTER scoring; log rule counts (kept/removed). PARTIAL (per-rule counts aggregated when columns available)
  - Capacity slicing modes: top‑N%, per‑rep capacity, hybrid with diversification; configurable and logged. PARTIAL (top_percent + per_rep + hybrid interleave; bias share check)
  - Cooldown logic: de‑emphasize accounts surfaced recently without action. DONE (config + score scaling when `days_since_last_surfaced` present)

- Explanations
  - Generate short human‑readable `nba_reason` (<150 chars) using 1–2 strongest drivers (e.g., high p, strong affinity, EV ~$Xk). PARTIAL (basic template)
  - Content guard: no sensitive attributes; fallback reason if signals are weak. TODO

- Artifacts
  - `whitespace_{cutoff}.csv` with: `customer_id, division, score, p_icp, p_icp_pct, lift_norm, als_norm, EV_norm, nba_reason`. DONE (core and explanation columns written)
  - `whitespace_explanations_{cutoff}.csv` with expanded fields if needed. DONE
  - `whitespace_metrics_{cutoff}.json` (capture@K, diversity by division/segment, stability vs prior run). PARTIAL (rows, checksum, weights, coverage, per-division eligibility counts, division shares; add capture/stability)
  - `thresholds_whitespace_{cutoff}.csv` (capacity/threshold grid). DONE (top_percent)
  - Deterministic checksum for ranked CSV. DONE

- CLI
  - `gosales/pipeline/rank_whitespace.py` with flags: `--cutoff`, `--window-months`, `--weights`, `--normalize`, `--capacity-mode`, `--accounts-per-rep`, `--config`. PARTIAL (skeleton + percentile norm + blend)
  - Wire to existing Phase‑2/3 artifacts (features for EV segments; models for p_icp; ALS if enabled). PARTIAL (features + model scoring wired)

- Guardrails
  - Cross‑division bias: if one division > X% of top‑N, warn; optional diversification slice. PARTIAL (warn on share > threshold)
  - EV outliers capped at p95; log number capped. TODO
  - ALS sparse coverage (< threshold): auto‑reduce weight and log. DONE (weight scaling; coverage emitted)
  - Affinity lift requires min support/confidence; otherwise set `lift_norm=0` and log. TODO
  - Full determinism: stable sort + checksum; seed any randomness. TODO

- Tests
  - Normalization: verify per‑division percentiles ~ uniform on synthetic data. PARTIAL (percentile normalize sanity test)
  - Degradation: drop ALS/lift → scores still produced; explanations fallback present. TODO
  - EV cap: inject outliers and assert capped; coverage of cap logged. PARTIAL (cap sanity test)
  - Bias/diversity: craft skewed distribution; assert warning and diversification option. TODO
  - Explanation: length <150 chars; contains expected tokens; no sensitive terms. PARTIAL (length/tokens test)
  - Determinism: same inputs/config → identical ranked order & checksum. TODO

- Performance & logging
  - Batched scoring I/O; vectorized normalization; memory‑safe joins. TODO
  - Structured logs with rule counts, weight adjustments, capacity outcomes. DONE (division summaries + selection summary JSONL)

- Documentation
  - Update README: Phase 4 overview, CLI usage, artifacts list, glossary entries. TODO
  - Add Phase 4 section to artifacts glossary. TODO


