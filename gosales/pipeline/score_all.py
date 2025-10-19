"""End-to-end orchestration entry point for the GoSales scoring pipeline.

Running this module bootstraps raw extracts into the curated warehouse, retrains
models where necessary, scores customers, validates holdouts, and emits final
deliverables.  It exists so operations teams have a single command that mirrors
our production DAG.
"""

from pathlib import Path
import os
import argparse
import json
from datetime import datetime
from typing import Optional, Sequence, Iterable

from gosales.utils.db import get_db_connection, validate_connection
from gosales.etl.load_csv import load_csv_to_db
from gosales.etl.build_star import build_star_schema
import sys
import subprocess
from sqlalchemy import inspect
from gosales.pipeline.label_audit import compute_label_audit
from gosales.pipeline.score_customers import generate_scoring_outputs
from gosales.utils.logger import get_logger
from gosales.utils.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from gosales.etl.sku_map import division_set, get_supported_models
from gosales.utils.run_context import default_manifest, emit_manifest
from gosales.pipeline.validate_holdout import validate_holdout
from gosales.ops.run import run_context
from gosales.utils.config import load_config
from gosales.utils.normalize import normalize_model_key
from shutil import rmtree
from dateutil.relativedelta import relativedelta

logger = get_logger(__name__)


_ACTIVE_MODEL_ALIASES: tuple[str, ...] = ("", "_cold")
_DEFAULT_TRAINING_CUTOFFS: tuple[str, ...] = (
    "2023-03-31",
    "2023-09-30",
    "2024-03-31",
    "2024-06-30",
)


def _segment_arg_to_list(segment: str | None) -> list[str] | None:
    if not segment:
        return None
    seg = str(segment).strip().lower()
    if seg == "both":
        return ["warm", "cold"]
    if seg in {"warm", "cold"}:
        return [seg]
    return None


def _derive_targets():
    """Return the sorted list of divisions/models to score."""

    normalized = {}

    def add_candidates(candidates):
        for candidate in candidates:
            if not candidate:
                continue
            norm = normalize_model_key(candidate)
            if not norm:
                continue
            normalized.setdefault(norm, candidate)

    try:
        divisions = division_set()
    except Exception:
        divisions = ("Solidworks",)

    add_candidates(
        (
            "Printers",
            "SWX_Seats",
            "PDM_Seats",
            "SW_Electrical",
            "SW_Inspection",
            "Success Plan",
        )
    )

    add_candidates(divisions)
    add_candidates(("Training", "Services", "Simulation", "Scanning", "CAMWorks"))

    models = get_supported_models()
    add_candidates(models)

    return sorted(normalized.values())


def _prune_legacy_model_dirs(targets, models_dir, log=logger):
    normalized_targets = {normalize_model_key(t) for t in targets if t}
    keep_aliases: set[str] = set()

    for target in normalized_targets:
        if not target:
            continue
        base_variants = {
            target,
            target.replace(" ", "_"),
            target.replace(" ", "-"),
        }
        for variant in base_variants:
            variant = variant.strip()
            if not variant:
                continue
            for alias in _ACTIVE_MODEL_ALIASES:
                suffix = f"{alias}_model" if alias else "_model"
                keep_aliases.add(f"{variant}{suffix}".casefold())

    for path in models_dir.glob("*_model"):
        name_cf = path.name.casefold()
        if name_cf in keep_aliases:
            continue

        raw_base = path.name[:-6] if path.name.lower().endswith("_model") else path.name
        aliasless_candidates = {raw_base}
        raw_base_cf = raw_base.casefold()
        for alias in _ACTIVE_MODEL_ALIASES:
            if alias and raw_base_cf.endswith(alias):
                aliasless_candidates.add(raw_base[: -len(alias)])

        aliasless_candidates = {c for c in aliasless_candidates if c}
        if any(normalize_model_key(candidate) in normalized_targets for candidate in aliasless_candidates):
            continue

        log.info(f"Pruning legacy model directory: {path}")
        try:
            rmtree(path)
        except Exception as err:
            log.warning(f"Failed to prune {path}: {err}")


def _parse_cutoff(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d")
    except Exception:
        return None


def _normalize_cutoff_list(values: Iterable[str], *, log) -> list[str]:
    normalized: set[datetime] = set()
    invalid: list[str] = []
    for value in values or []:
        parsed = _parse_cutoff(value)
        if parsed is None:
            invalid.append(str(value))
            continue
        normalized.add(parsed)
    if invalid:
        log.warning("Ignoring invalid cutoff date(s) %s; expected YYYY-MM-DD.", invalid)
    return [dt.strftime("%Y-%m-%d") for dt in sorted(normalized)]


def _generate_cutoff_series(base: datetime, frequency_months: int, count: int) -> list[str]:
    if frequency_months <= 0 or count <= 0:
        return []
    series = [
        base - relativedelta(months=frequency_months * idx)
        for idx in range(count)
    ]
    return [dt.strftime("%Y-%m-%d") for dt in sorted(set(series))]


def _resolve_training_cutoffs(
    run_cfg,
    *,
    cutoff_date: Optional[str],
    override_cutoffs: Optional[Sequence[str]],
    log,
) -> list[str]:
    if override_cutoffs:
        normalized = _normalize_cutoff_list(override_cutoffs, log=log)
        if normalized:
            return normalized
        log.warning("Training cutoff override did not include valid dates; falling back to config.")

    cfg_cutoffs = getattr(run_cfg, "training_cutoffs", None)
    normalized_cfg = _normalize_cutoff_list(cfg_cutoffs or [], log=log)
    if normalized_cfg:
        return normalized_cfg

    base_candidate = cutoff_date or getattr(run_cfg, "cutoff_date", None)
    base_dt = _parse_cutoff(base_candidate)
    frequency = getattr(run_cfg, "training_frequency_months", 0) or 0
    count = getattr(run_cfg, "training_cutoff_count", 0) or 0
    try:
        frequency = int(frequency)
    except Exception:
        log.warning("Invalid training_frequency_months=%r; expected integer months.", frequency)
        frequency = 0
    try:
        count = int(count)
    except Exception:
        log.warning("Invalid training_cutoff_count=%r; expected integer.", count)
        count = 0

    if base_dt and frequency > 0 and count > 0:
        generated = _generate_cutoff_series(base_dt, frequency, count)
        if generated:
            return generated

    if base_dt:
        fallback = [base_dt.strftime("%Y-%m-%d")]
        log.info("Using primary cutoff %s for training because no additional cutoffs were derived.", fallback[0])
        return fallback

    log.warning("Unable to resolve training cutoffs; defaulting to legacy values %s.", _DEFAULT_TRAINING_CUTOFFS)
    return list(_DEFAULT_TRAINING_CUTOFFS)


def _record_run_failure(ctx, message: str) -> None:
    """Persist run metadata indicating the pipeline terminated with an error."""

    logger.error(message)
    try:
        ctx["write_manifest"]({"status": "error", "message": message})
    except Exception as manifest_err:
        logger.warning(f"Unable to write failure manifest: {manifest_err}")
    try:
        ctx["append_registry"]({
            "phase": "pipeline_score_all",
            "status": "error",
            "error": message,
            "artifact_count": 0,
        })
    except Exception as registry_err:
        logger.warning(f"Unable to append failure entry to registry: {registry_err}")


def _star_build_successful(result, curated_engine) -> bool:
    """Best-effort check whether the star schema build succeeded."""

    if isinstance(result, dict):
        if "status" in result:
            return str(result["status"]).lower() in {"ok", "success"}
        if "success" in result:
            return bool(result["success"])
    if isinstance(result, bool):
        return result

    # Fallback: inspect curated database for required tables
    try:
        inspector = inspect(curated_engine)
        tables = set(inspector.get_table_names())
    except Exception:
        # When inspection is unavailable (e.g., stub engine in tests), assume success
        # so downstream phases can surface their own errors.
        return True

    required_tables = {"dim_customer", "fact_transactions"}
    return required_tables.issubset(tables)

def score_all(
    segment: str | None = None,
    *,
    training_cutoffs: Sequence[str] | None = None,
    use_line_items: bool | None = None,
):
    """
    Orchestrates the entire GoSales pipeline from data ingestion to final scoring.

    This master script executes the following steps in order:
    1.  Loads all raw CSV data into a staging table in the database.
    2.  Builds the clean, tidy star schema (`dim_customer`, `fact_transactions`).
    3.  Trains a new machine learning model for the 'Solidworks' division.
    4.  Generates and saves the final ICP scores and whitespace opportunities.
    """
    logger.info("Starting the full GoSales scoring pipeline...")

    seg_list = _segment_arg_to_list(segment)
    prev_segment_env = os.environ.get("GOSALES_POP_BUILD_SEGMENTS")
    if seg_list is not None:
        os.environ["GOSALES_POP_BUILD_SEGMENTS"] = ",".join(seg_list)
        logger.info("Segment override for pipeline: %s", ",".join(seg_list))

    try:
        with run_context("pipeline_score_all") as ctx:
            # --- 1. Setup ---
            db_engine = get_db_connection()          # source (Azure)
            backend = getattr(getattr(db_engine, "dialect", None), "name", "")
            is_azure_like = backend in {"mssql"}
            if not is_azure_like:
                logger.info(
                    "Primary database engine '%s' detected; forcing CSV ingest for local sample tables.",
                    backend or "unknown",
                )
            from gosales.utils.db import get_curated_connection
            curated_engine = get_curated_connection()  # curated (local sqlite)
            # Connection health checks
            cfg = None
            try:
                cfg = load_config()
                if use_line_items is not None:
                    try:
                        cfg.etl.line_items.use_line_item_facts = bool(use_line_items)
                    except Exception:
                        logger.warning("Unable to override line-item toggle on loaded config")
                strict = bool(getattr(getattr(cfg, 'database', object()), 'strict_db', False))
            except Exception:
                strict = False
            if not validate_connection(db_engine):
                msg = "Primary database connection is unhealthy."
                if strict:
                    raise RuntimeError(msg)
                logger.warning(msg + " Proceeding with best-effort fallback where applicable.")
            if not validate_connection(curated_engine):
                msg2 = "Curated database connection is unhealthy."
                if strict:
                    raise RuntimeError(msg2)
                logger.warning(msg2)
            targets = _derive_targets()
            env_override = None
            if seg_list is not None:
                env_override = os.environ.copy()
                env_override["GOSALES_POP_BUILD_SEGMENTS"] = ",".join(seg_list)

            # --- 2. ETL Phase ---
            logger.info("--- Phase 1: ETL ---")
            # Skip local CSV ingest when a database source is configured
            if cfg is None:
                cfg = load_config()
                if use_line_items is not None:
                    try:
                        cfg.etl.line_items.use_line_item_facts = bool(use_line_items)
                    except Exception:
                        logger.warning("Unable to override line-item toggle on loaded config")
            run_cfg = getattr(cfg, "run", None)
            cutoff_date = str(getattr(run_cfg, "cutoff_date", "2024-06-30") or "2024-06-30")
            prediction_window_months = int(getattr(run_cfg, "prediction_window_months", 6) or 6)
            resolved_training_cutoffs = _resolve_training_cutoffs(
                run_cfg,
                cutoff_date=cutoff_date,
                override_cutoffs=training_cutoffs,
                log=logger,
            )
            logger.info("Resolved training cutoffs for training phase: %s", ", ".join(resolved_training_cutoffs))
            src = getattr(getattr(cfg, 'database', object()), 'source_tables', {}) or {}
            sl_src = str(src.get('sales_log', '')).strip()
            use_db_source = bool(sl_src and sl_src.lower() != 'csv' and is_azure_like)
            if use_db_source:
                logger.info("Sales Log source is mapped to DB object '%s'; skipping local CSV ingest.", sl_src)
            else:
                if not is_azure_like and sl_src and sl_src.lower() != 'csv':
                    logger.info(
                        "Overriding configured Sales Log source '%s' for local engine '%s'; loading sample CSVs.",
                        sl_src,
                        backend or "unknown",
                    )
                # Define the CSV files and their corresponding table names
                csv_files = {
                    "Sales_Log.csv": "sales_log",
                    "TR - Industry Enrichment.csv": "industry_enrichment",
                }
                for file_name, table_name in csv_files.items():
                    file_path = DATA_DIR / "database_samples" / file_name
                    load_csv_to_db(file_path, table_name, db_engine)
    
            try:
                star_result = build_star_schema(
                    db_engine,
                    use_line_item_facts=use_line_items,
                )
            except Exception as exc:
                message = f"Star schema build raised an exception: {exc}"
                logger.exception(message)
                _record_run_failure(ctx, message)
                raise
            if not _star_build_successful(star_result, curated_engine):
                detail = star_result if star_result is not None else "missing curated tables"
                message = f"Star schema build failed: {detail}"
                _record_run_failure(ctx, message)
                raise RuntimeError(message)
            # Build invoice-level events for leakage-safe feature engineering
            try:
                from gosales.etl.events import build_fact_events
                build_fact_events()
            except Exception as e:
                logger.warning(f"Eventization step failed (non-blocking): {e}")
            logger.info("--- ETL Phase Complete ---")
    
            # --- 3. Label Audit (Phase 2) ---
            logger.info("--- Phase 2: Label audit (leakage-safe targets) ---")
            # Use configured run cutoff and prediction window for label audit scope
            for div in targets:
                try:
                    compute_label_audit(curated_engine, div, cutoff_date, prediction_window_months)
                except Exception as e:
                    logger.warning(f"Label audit failed for {div}: {e}")
            logger.info("--- Label audit complete ---")
    
            # --- 4. Feature Library emission (catalog) ---
            # Build a feature matrix per division to emit the feature catalog before training
            try:
                from gosales.features.engine import create_feature_matrix
                for div in targets:
                    try:
                        create_feature_matrix(curated_engine, div, cutoff_date, prediction_window_months)
                    except Exception as e:
                        logger.warning(f"Feature catalog emission failed for {div} (non-blocking): {e}")
                logger.info("--- Feature catalogs emitted ---")
            except Exception as e:
                logger.warning(f"Feature catalog emission failed (non-blocking): {e}")
    
            # --- 5. Model Training Phase ---
            logger.info("--- Phase 3: Training models for all targets (robust trainer) ---")
            cutoffs_arg = ",".join(resolved_training_cutoffs)
            for div in targets:
                try:
                    logger.info(f"Training model for target: {div} (cutoffs={cutoffs_arg})")
                    cmd = [
                        sys.executable,
                        "-m",
                        "gosales.models.train",
                        "--division",
                        div,
                        "--cutoffs",
                        cutoffs_arg,
                        "--window-months",
                        str(prediction_window_months),
                    ]
                    if segment:
                        cmd.extend(["--segment", segment])
                    subprocess.run(cmd, check=True, env=env_override)
                except Exception as e:
                    logger.warning(f"Training failed for {div}: {e}")
            logger.info("--- Model Training Phase Complete ---")
    
            # Prune legacy model directories not in current targets
            try:
                _prune_legacy_model_dirs(targets, MODELS_DIR, log=logger)
            except Exception as e:
                logger.warning(f"Model pruning step failed: {e}")
    
            # --- 6. Scoring Phase ---
            logger.info("--- Phase 4: Generating Scores and Whitespace ---")
            # Create run manifest and record high-level context
            run_manifest = default_manifest(pipeline_version="0.1.0")
            run_manifest["cutoff"] = cutoff_date
            run_manifest["window_months"] = int(prediction_window_months)
            run_manifest["training_cutoffs"] = list(resolved_training_cutoffs)
    
            # Generate outputs; function will update manifest details (divisions scored, alerts)
            icp_scores_path = generate_scoring_outputs(
                curated_engine,
                run_manifest=run_manifest,
                cutoff_date=cutoff_date,
                prediction_window_months=prediction_window_months,
                segment=segment,
            )
    
            # Persist manifest alongside outputs and append to registry via run_context
            try:
                manifest_path = emit_manifest(OUTPUTS_DIR, run_manifest["run_id"], run_manifest)
                logger.info(f"Wrote run manifest to {manifest_path}")
                whitespace_entry = run_manifest.get("whitespace_artifact")
                if whitespace_entry:
                    ws_path = Path(whitespace_entry)
                    if not ws_path.exists():
                        ws_path = OUTPUTS_DIR / "whitespace.csv"
                    whitespace_str = str(ws_path)
                else:
                    whitespace_str = str(OUTPUTS_DIR / "whitespace.csv")
                icp_entry = run_manifest.get("icp_scores")
                if icp_entry:
                    icp_path = Path(icp_entry)
                else:
                    icp_path = icp_scores_path if icp_scores_path is not None else OUTPUTS_DIR / "icp_scores.csv"
                ctx["write_manifest"](
                    {
                        "run_manifest": str(manifest_path),
                        "icp_scores": str(icp_path),
                        "whitespace": whitespace_str,
                    }
                )
                ctx["append_registry"](
                    {
                        "phase": "pipeline_score_all",
                        "divisions": targets,
                        "cutoff": cutoff_date,
                        "window_months": int(prediction_window_months),
                        "training_cutoffs": list(resolved_training_cutoffs),
                        "artifact_count": 3,
                        "status": "finished",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to write run manifest/registry: {e}")
            logger.info("--- Scoring Phase Complete ---")
    
            # --- 7. Hold-out validation & gates (Phase 5) ---
            try:
                holdout_cfg = getattr(cfg, "validation", object())
                holdout_required = bool(getattr(holdout_cfg, "holdout_required", True))
            except Exception:
                holdout_required = True

            try:
                icp_path = Path(run_manifest.get("icp_scores") or (icp_scores_path or OUTPUTS_DIR / "icp_scores.csv"))
                if icp_path.exists():
                    year_tag = None
                    try:
                        y = int(str(run_manifest.get("cutoff", "")).split("-")[0]) if isinstance(run_manifest, dict) else None
                        if y:
                            year_tag = str(y + 1)
                    except Exception:
                        year_tag = None
                    result_path = validate_holdout(icp_scores_csv=str(icp_path), year_tag=year_tag)
                    holdout_status = None
                    if result_path:
                        try:
                            payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
                            holdout_status = str(payload.get("status", "")).strip().lower()
                        except FileNotFoundError:
                            if holdout_required:
                                raise
                            logger.warning("Hold-out metrics file %s missing (continuing: holdout_required=False)", result_path)
                        except Exception as parse_err:
                            if holdout_required:
                                raise RuntimeError(f"Failed to inspect hold-out metrics: {parse_err}") from parse_err
                            logger.warning("Failed to inspect hold-out metrics (%s); continuing because holdout_required is false", parse_err)
                        if holdout_status and holdout_status not in {"ok"}:
                            message = f"Hold-out validation reported failing status '{holdout_status}'"
                            if holdout_required:
                                raise RuntimeError(message)
                            logger.warning(message + " (continuing because holdout_required is false)")
            except Exception as e:
                message = f"Hold-out validation step failed: {e}"
                if holdout_required:
                    logger.exception(message)
                    _record_run_failure(ctx, message)
                    raise RuntimeError(message) from e
                logger.warning(message + " (continuing because validation.holdout_required is false)")
            
            logger.info("GoSales scoring pipeline finished successfully!")
    finally:
        if seg_list is not None:
            if prev_segment_env is None:
                os.environ.pop("GOSALES_POP_BUILD_SEGMENTS", None)
            else:
                os.environ["GOSALES_POP_BUILD_SEGMENTS"] = prev_segment_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full GoSales scoring pipeline")
    parser.add_argument(
        "--segment",
        choices=["warm", "cold", "both"],
        help="Override population.build_segments for this pipeline run",
    )
    parser.add_argument(
        "--training-cutoffs",
        help="Comma-separated list of training cutoff dates (YYYY-MM-DD). Overrides configuration.",
    )
    parser.add_argument(
        "--use-line-items",
        dest="use_line_items",
        action="store_true",
        help="Enable line-item fact build for the ETL phase of this run.",
    )
    parser.add_argument(
        "--no-use-line-items",
        dest="use_line_items",
        action="store_false",
        help="Disable line-item fact build regardless of configuration.",
    )
    parser.set_defaults(use_line_items=None)
    args = parser.parse_args()
    overrides = None
    if args.training_cutoffs:
        overrides = [c.strip() for c in args.training_cutoffs.split(",") if c.strip()]
    score_all(segment=args.segment, training_cutoffs=overrides, use_line_items=args.use_line_items)
