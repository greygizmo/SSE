# Serious Issues Identified in GoSales Engine

1. **`score_all` hard-codes stale training windows and cutoffs.** The orchestrator fixes the label cutoff to `2024-06-30` and trains every division on the static list `["2023-03-31", "2023-09-30", "2024-03-31", "2024-06-30"]` (see `gosales/pipeline/score_all.py:270`). These values ignore the configurable `run.cutoff_date` and any newly ingested data, so running the pipeline today retrains on outdated horizons unless developers edit the source code before every release.
