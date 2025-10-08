# Critical Issues Identified

## 1. `score_all` cannot run due to a syntax error
The orchestration pipeline defines a `try` block around the model-training loop but the body of the block is mis-indented, so Python sees the `cmd = [...]` line as outside the `try`. This breaks module import with a `SyntaxError`, preventing any end-to-end scoring run or associated tests from even importing `score_all`.【F:gosales/pipeline/score_all.py†L293-L315】

## 2. Training CLI rejects programmatic callers
`gosales.models.train.main` exposes a `--segment` option, but the Python signature omits the default value. When tests or other code call `main.callback(...)` (the Click entry point) without explicitly passing `segment`, Click does not supply a value and Python raises `TypeError: main() missing 1 required positional argument: 'segment'`, blocking the training harness used in automated checks.【F:gosales/models/train.py†L740-L779】

## 3. ALS fallback logic can outrank true signals
When ALS coverage falls below the configured threshold, `_rank_whitespace` overwrites `als_norm` for zero-signal rows with percentiles from asset ALS or item2vec without bounding them relative to customers that actually have transaction ALS strength. Fallback rows can therefore receive higher normalized scores than accounts with legitimate ALS embeddings, violating the expectations codified in the Phase 4 tests.【F:gosales/pipeline/rank_whitespace.py†L868-L889】

## 4. ALS blending underweights rows that only have asset embeddings
The same ranking routine always multiplies transaction embeddings by `blend_txn` and asset embeddings by `blend_assets`. When a customer lacks transaction ALS but has asset ALS, their `als_norm` is still scaled by the blend weight (default 0.5), cutting the score in half compared with the raw asset percentile and breaking the asset fallback guarantees.【F:gosales/pipeline/rank_whitespace.py†L814-L835】

## 5. Stability controls are impossible to configure via `config.yaml`
`ModelingConfig` omits any `stability` dataclass, yet the training code expects `cfg.modeling.stability` to exist and exposes knobs such as `penalty_lambda`, `cv_guard_max`, and backbone feature lists. Without a structured configuration object these levers cannot be set in `config.yaml`, so stability weighting and guards are effectively hard-coded defaults rather than operator-tunable parameters.【F:gosales/utils/config.py†L141-L166】【F:gosales/models/train.py†L808-L820】
