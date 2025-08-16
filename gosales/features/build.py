from __future__ import annotations

import json
from pathlib import Path
import click
import pandas as pd
import polars as pl

from gosales.utils.config import load_config
from gosales.utils.db import get_db_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger
from gosales.features.engine import create_feature_matrix
from gosales.features.utils import compute_sha256
from gosales.utils.config import load_config
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
    engine = get_db_connection()

    # For now, engine already computes a comprehensive set; toggles can be wired later
    cutoffs = [c.strip() for c in cutoff.split(",") if c.strip()]
    artifacts: dict[str, str] = {}
    with run_context("features_build") as ctx:
        for cut in cutoffs:
            fm = create_feature_matrix(engine, division, cut, cfg.run.prediction_window_months)
            # Optional ALS embeddings
            if cfg.features.use_als_embeddings:
                als_df = customer_als_embeddings(
                    engine,
                    cut,
                    factors=16,
                    lookback_months=cfg.features.als_lookback_months,
                )
                if not als_df.is_empty():
                    fm = fm.join(als_df, on='customer_id', how='left').fill_null(0)
            if fm.is_empty():
                logger.warning(f"Empty feature matrix for cutoff {cut}")
                continue
            # Artifacts naming
            base = f"{division.lower()}_{cut}"
            # Deterministic sort
            fm = fm.sort(["customer_id"])
            # If ALS embeddings available and division centroid similarity needed for Phase 4, compute quick sim feature (optional)
            try:
                if cfg.features.use_als_embeddings:
                    # Compute centroid among pre-cutoff owners if possible
                    pdf = fm.to_pandas()
                    als_cols = [c for c in pdf.columns if str(c).startswith('als_f')]
                    if als_cols:
                        # Approximate ownership with div-scope tx_n
                        div_cols = [f'rfm__div__tx_n__{w}m' for w in cfg.features.windows_months]
                        have_cols = [c for c in div_cols if c in pdf.columns]
                        owned = (pdf[have_cols].sum(axis=1) > 0) if have_cols else pd.Series(False, index=pdf.index)
                        centroid = pdf.loc[owned, als_cols].mean(axis=0).values if owned.any() else pdf[als_cols].mean(axis=0).values
                        sim = pdf[als_cols].fillna(0.0).values.dot(centroid)
                        pdf['als_sim_division'] = sim
                        fm = pl.from_pandas(pdf)
            except Exception:
                pass
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
                cfg = load_config()
                p = float(cfg.features.gp_winsor_p)
                winsor_caps = {}
                for col in fm_pd.columns:
                    if col.endswith("gp_sum__3m") or col.endswith("gp_sum__6m") or col.endswith("gp_sum__12m") or col.endswith("gp_sum__24m"):
                        s = pd.to_numeric(fm_pd[col], errors="coerce")
                        lower = float(s.quantile(0.0))
                        upper = float(s.quantile(p))
                        winsor_caps[col] = {"lower": lower, "upper": upper}
                if winsor_caps:
                    stats["winsor_caps"] = winsor_caps
            except Exception:
                pass
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
            ctx["append_registry"]({"phase": "features_build", "division": division, "cutoffs": cutoffs, "artifact_count": len(artifacts)})
        except Exception:
            pass


if __name__ == "__main__":
    main()


