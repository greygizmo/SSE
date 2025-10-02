"""End-to-end orchestration entry point for the GoSales scoring pipeline.

Running this module bootstraps raw extracts into the curated warehouse, retrains
models where necessary, scores customers, validates holdouts, and emits final
deliverables.  It exists so operations teams have a single command that mirrors
our production DAG.
"""

from pathlib import Path

from gosales.utils.db import get_db_connection, validate_connection
from gosales.etl.load_csv import load_csv_to_db
from gosales.etl.build_star import build_star_schema
import sys
import subprocess
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

logger = get_logger(__name__)


_ACTIVE_MODEL_ALIASES: tuple[str, ...] = ("", "_cold")


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

def score_all():
    """
    Orchestrates the entire GoSales pipeline from data ingestion to final scoring.

    This master script executes the following steps in order:
    1.  Loads all raw CSV data into a staging table in the database.
    2.  Builds the clean, tidy star schema (`dim_customer`, `fact_transactions`).
    3.  Trains a new machine learning model for the 'Solidworks' division.
    4.  Generates and saves the final ICP scores and whitespace opportunities.
    """
    logger.info("Starting the full GoSales scoring pipeline...")

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
        try:
            cfg = load_config()
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

        # --- 2. ETL Phase ---
        logger.info("--- Phase 1: ETL ---")
        # Skip local CSV ingest when a database source is configured
        cfg = load_config()
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

        build_star_schema(db_engine)
        # Build invoice-level events for leakage-safe feature engineering
        try:
            from gosales.etl.events import build_fact_events
            build_fact_events()
        except Exception as e:
            logger.warning(f"Eventization step failed (non-blocking): {e}")
        logger.info("--- ETL Phase Complete ---")

        # --- 3. Label Audit (Phase 2) ---
        logger.info("--- Phase 2: Label audit (leakage-safe targets) ---")
        # Training cutoff chosen so the 6-month target window is within training data (Jul-Dec 2024)
        cutoff_date = "2024-06-30"
        prediction_window_months = 6
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
        cut_list = ["2023-03-31", "2023-09-30", "2024-03-31", "2024-06-30"]
        cutoffs_arg = ",".join(cut_list)
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
                subprocess.run(cmd, check=True)
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

        # Generate outputs; function will update manifest details (divisions scored, alerts)
        icp_scores_path = generate_scoring_outputs(
            curated_engine,
            run_manifest=run_manifest,
            cutoff_date=cutoff_date,
            prediction_window_months=prediction_window_months,
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
                    "artifact_count": 3,
                    "status": "finished",
                }
            )
        except Exception as e:
            logger.warning(f"Failed to write run manifest/registry: {e}")
        logger.info("--- Scoring Phase Complete ---")

        # --- 7. Hold-out validation & gates (Phase 5) ---
        try:
            icp_path = Path(run_manifest.get("icp_scores") or (icp_scores_path or OUTPUTS_DIR / "icp_scores.csv"))
            if icp_path.exists():
                # Derive a year tag from cutoff (simple heuristic: cutoff year + 1)
                year_tag = None
                try:
                    y = int(str(run_manifest.get("cutoff", "")).split("-")[0]) if isinstance(run_manifest, dict) else None
                    if y:
                        year_tag = str(y + 1)
                except Exception:
                    year_tag = None
                validate_holdout(icp_scores_csv=str(icp_path), year_tag=year_tag)
        except Exception as e:
            logger.warning(f"Hold-out validation step failed (non-blocking): {e}")

        logger.info("GoSales scoring pipeline finished successfully!")

if __name__ == "__main__":
    score_all()
