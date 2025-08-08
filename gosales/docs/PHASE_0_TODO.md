### Phase 0 To-Do (repo hardening vs playbook)

- Config system
  - Add `gosales/config.yaml` with paths, db, run, etl, logging. DONE
  - Implement `gosales/utils/config.py` loader with precedence YAML → env → CLI; ensure dirs exist. DONE
  - On run, persist resolved snapshot to `outputs/runs/<run_id>/config_resolved.yaml`. DONE

- Ingest & staging
  - Current `etl/load_csv.py` robustly reads with encoding fallback. Keep. Add manifest/checksums in later pass. TODO
  - Add header normalization and column profiling to staging parquet. TODO

- Contracts & coercion
  - Existing `etl/contracts.py` implements required/PK/dupe checks. Keep.
  - Add Pandera or extended coercion checks (qty types, date bounds). TODO
  - Wire fail-soft option in star build. DONE

- Keys
  - Add `etl/keys.py` with deterministic `txn_key`, `customer_key`, `date_key`. DONE
  - Add unit tests. DONE

- SKU map
  - Single source at `etl/sku_map.py` already present; ensure overrides later. TODO

- Star schema
  - Extend `etl/build_star.py` to emit curated Parquet and build `dim_date` & `dim_product`. DONE
  - Deterministic ordering and parquet checksums (checksum helper in place). PARTIAL
  - FK integrity checks and quarantine. TODO

- Observability
  - Add `ops/run.py` with JSONL logging and run ids. DONE
  - Emit QA summaries (`outputs/qa/phase0_report.md`, `summary.json`). TODO

- CLI
  - Add args: `--config`, `--rebuild`, `--staging-only`, `--fail-soft`. DONE

- Tests
  - Add tests for parsers and keys. DONE
  - Add golden snapshot and FK integrity tests. TODO

- Docs
  - Update quick commands in `gosales/README.md`. DONE
  - Keep `docs/Sales_Log_Schema.md` as reference. KEEP


