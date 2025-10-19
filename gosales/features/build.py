"""Command-line entry point for generating model feature matrices.

It wraps :func:`gosales.features.engine.create_feature_matrix` with CLI options
for division, cutoff, and optional embedding sources, writing the resulting
parquet/csv artifacts into the outputs directory for training and QA workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
import click
import pandas as pd
import polars as pl

from gosales.utils.config import load_config
from gosales.utils.db import get_db_connection, get_curated_connection, validate_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger
from gosales.features.engine import create_feature_matrix
from gosales.features.utils import compute_sha256
from gosales.features.als_embed import customer_als_embeddings
from gosales.ops.run import run_context


logger = get_logger(__name__)


@click.command()
@click.option("--division", required=True)
@click.option("--cutoff", required=True, help="YYYY-MM-DD (or list: 2024-03-31,2024-06-30)")
@click.option("--windows", default="3,6,12,24")
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
@click.option("--with-eb/--no-eb", default=True)
@click.option("--with-affinity/--no-affinity", default=True)
@click.option("--with-als/--no-als", default=False)
def main(division: str, cutoff: str, windows: str, config: str, with_eb: bool, with_affinity: bool, with_als: bool) -> None:
    cfg = load_config(config)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    # Use curated engine for features (fact_transactions, dim_customer live here)
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()
    # Connection health check
    try:
        strict = bool(getattr(getattr(cfg, 'database', object()), 'strict_db', False))
    except Exception:
        strict = False
    if not validate_connection(engine):
        msg = "Database connection is unhealthy."
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    # For now, engine already computes a comprehensive set; toggles can be wired later
    cutoffs = [c.strip() for c in cutoff.split(",") if c.strip()]
    artifacts: dict[str, str] = {}
    with run_context("features_build") as ctx:
        for cut in cutoffs:
            fm = create_feature_matrix(engine, division, cut, cfg.run.prediction_window_months)
            # Optional ALS embeddings
            if cfg.features.use_als_embeddings:
                als_factors = 16
                try:
                    lag_days = int(getattr(cfg.features, 'affinity_lag_days', 60) or 60)
                except Exception:
                    lag_days = 60
                als_df = customer_als_embeddings(
                    engine,
                    cut,
                    factors=als_factors,
                    lookback_months=cfg.features.als_lookback_months,
                    lag_days=lag_days,
                )
                if "customer_id" in als_df.columns and als_df.height > 0:
                    fm_id_dtype = fm.schema.get("customer_id")
                    try:
                        als_id_dtype = als_df.schema.get("customer_id")
                    except AttributeError:
                        als_id_dtype = None
                    if fm_id_dtype is not None and als_id_dtype is not None and fm_id_dtype != als_id_dtype:
                        try:
                            als_df = als_df.with_columns(
                                pl.col("customer_id").cast(fm_id_dtype).alias("customer_id")
                            )
                        except Exception:
                            pass
                    fm = fm.join(als_df, on="customer_id", how="left")
                als_cols = [f"als_f{i}" for i in range(als_factors)]
                als_exprs = []
                for col in als_cols:
                    if col in fm.columns:
                        als_exprs.append(pl.col(col).cast(pl.Float64).fill_null(0.0).alias(col))
                    else:
                        als_exprs.append(pl.lit(0.0, dtype=pl.Float64).alias(col))
                if als_exprs:
                    fm = fm.with_columns(als_exprs)
            if fm.is_empty():
                logger.warning(f"Empty feature matrix for cutoff {cut}")
                continue
            # Artifacts naming
            base = f"{division.lower()}_{cut}"
            # Deterministic sort
            fm = fm.sort(["customer_id"])
            if getattr(cfg.features, "use_market_basket", False):
                try:
                    lag_days = int(getattr(cfg.features, "affinity_lag_days", 60) or 60)
                except Exception:
                    lag_days = 60
                mb_cols = [
                    f"mb_lift_max_lag{lag_days}d",
                    f"mb_lift_mean_lag{lag_days}d",
                    f"affinity__div__lift_topk__12m_lag{lag_days}d",
                ]
                mb_exprs = []
                for col in mb_cols:
                    if col in fm.columns:
                        mb_exprs.append(pl.col(col).cast(pl.Float64).fill_null(0.0).alias(col))
                    else:
                        mb_exprs.append(pl.lit(0.0, dtype=pl.Float64).alias(col))
                if mb_exprs:
                    fm = fm.with_columns(mb_exprs)
            # If ALS embeddings available and division centroid similarity needed for Phase 4, compute quick sim feature (optional)
            try:
                if cfg.features.use_als_embeddings:
                    # Compute centroid among pre-cutoff owners if possible
                    pdf = fm.to_pandas()
                    als_cols = [c for c in pdf.columns if str(c).startswith("als_f")]
                    if als_cols:
                        # Approximate ownership with div-scope tx_n
                        div_cols = [f"rfm__div__tx_n__{w}m" for w in cfg.features.windows_months]
                        have_cols = [c for c in div_cols if c in pdf.columns]
                        owned = (pdf[have_cols].sum(axis=1) > 0) if have_cols else pd.Series(False, index=pdf.index)
                        centroid = (
                            pdf.loc[owned, als_cols].mean(axis=0).values
                            if owned.any()
                            else pdf[als_cols].mean(axis=0).values
                        )
                        sim = pdf[als_cols].fillna(0.0).values.dot(centroid)
                        pdf["als_sim_division"] = sim
                        fm = pl.from_pandas(pdf)
            except Exception as e:
                logger.exception("Failed to compute ALS similarity for cutoff %s", cut)
                raise
            # Stats JSON (coverage + winsor caps if applicable)
            fm_pd = fm.to_pandas()
            stats = {
                "columns": {
                    col: {
                        "dtype": str(fm_pd[col].dtype),
                        "non_null": int(fm_pd[col].notna().sum()),
                        "coverage": float(round(fm_pd[col].notna().mean(), 6)),
                    }
                    for col in fm_pd.columns
                }
            }
            # Winsor caps for gp_sum features per config
            try:
                p = float(cfg.features.gp_winsor_p)
                winsor_caps = {}
                for col in fm_pd.columns:
                    if (
                        col.endswith("gp_sum__3m")
                        or col.endswith("gp_sum__6m")
                        or col.endswith("gp_sum__12m")
                        or col.endswith("gp_sum__24m")
                    ):
                        s = pd.to_numeric(fm_pd[col], errors="coerce")
                        lower = float(s.quantile(0.0))
                        upper = float(s.quantile(p))
                        winsor_caps[col] = {"lower": lower, "upper": upper}
                if winsor_caps:
                    stats["winsor_caps"] = winsor_caps
            except Exception as e:
                logger.exception("Failed to compute winsor caps for cutoff %s", cut)
                raise
            feat_path = OUTPUTS_DIR / f"features_{base}.parquet"
            fm.write_parquet(feat_path)
            artifacts[f"features_{base}.parquet"] = str(feat_path)
            catalog_path = OUTPUTS_DIR / f"feature_catalog_{base}.csv"
            pd.DataFrame(
                [{"name": col, "dtype": str(fm_pd[col].dtype), "coverage": float(round(fm_pd[col].notna().mean(), 6))} for col in fm_pd.columns]
            ).to_csv(catalog_path, index=False)
            artifacts[catalog_path.name] = str(catalog_path)
            stats["checksum"] = compute_sha256(feat_path)
            stats_path = OUTPUTS_DIR / f"feature_stats_{base}.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            artifacts[stats_path.name] = str(stats_path)
            logger.info(f"Wrote features and stats for {division} @ {cut}")
        try:
            ctx["write_manifest"](artifacts)
            ctx["append_registry"](
                {
                    "phase": "features_build",
                    "division": division,
                    "cutoffs": cutoffs,
                    "artifact_count": len(artifacts),
                }
            )
        except Exception as e:
            logger.exception("Failed to record feature build artifacts for %s", division)
            raise


if __name__ == "__main__":
    main()


