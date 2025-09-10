from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import click
import numpy as np
import pandas as pd
import sys

# Ensure repository root is importable when running as a script
try:
    import gosales  # type: ignore
except Exception:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gosales.sql.queries import moneyball_assets_select
from gosales.utils.config import load_config
from gosales.utils.db import get_curated_connection, get_db_connection
from gosales.utils.logger import get_logger
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.sql import ensure_allowed_identifier, validate_identifier


logger = get_logger(__name__)


def _norm(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _load_dim_customer_norms() -> Tuple[pd.DataFrame, set[str], pd.DataFrame]:
    """Load curated dim_customer and return (df, normalized_name_set, ambiguous_df)."""
    try:
        cur = get_curated_connection()
        dc = pd.read_sql(
            "SELECT customer_id, customer_name, customer_name_norm FROM dim_customer",
            cur,
        )
    except Exception as e:
        # Fallback: try to derive from sales_log in source DB
        logger.warning(
            "Failed to read dim_customer from curated DB (%s); falling back to dbo.saleslog unique names",
            e,
        )
        src = get_db_connection()
        sl = pd.read_sql(
            "SELECT [Customer] AS customer_name, [CompanyId] AS customer_id FROM dbo.saleslog",
            src,
        )
        sl["customer_name_norm"] = _norm(sl["customer_name"])
        dc = (
            sl.dropna(subset=["customer_name_norm"])  # type: ignore[reportUnknownArgumentType]
            .groupby("customer_name_norm")[
                ["customer_id", "customer_name"]
            ]
            .first()
            .reset_index()
        )

    if "customer_name_norm" not in dc.columns:
        dc["customer_name_norm"] = _norm(dc["customer_name"])  # type: ignore[reportUnknownMemberType]
    # Identify ambiguous norm -> multiple ids
    try:
        amb = (
            dc.groupby("customer_name_norm")["customer_id"]
            .nunique()
            .reset_index(name="n_ids")
        )
        amb = amb[amb["n_ids"] > 1].sort_values("n_ids", ascending=False)
        ambiguous = amb
    except Exception:
        ambiguous = pd.DataFrame(columns=["customer_name_norm", "n_ids"])  # empty

    norm_set: set[str] = set(_norm(dc["customer_name_norm"]).astype(str))
    return dc, norm_set, ambiguous


def _iter_moneyball(view_ident: str, chunksize: int = 250_000):
    eng = get_db_connection()
    sql = moneyball_assets_select(view_ident)
    # Stream in chunks where available
    try:
        it = pd.read_sql_query(sql, eng, chunksize=chunksize)
        for chunk in it:
            yield chunk
    except Exception:
        yield pd.read_sql(sql, eng)


@click.command()
@click.option(
    "--moneyball-view",
    default=None,
    help="DB object name for Moneyball Assets (defaults to config.database.source_tables.moneyball_assets)",
)
@click.option("--top", default=50, type=int, help="Top N unmapped names to export")
@click.option(
    "--outputs",
    default=str((OUTPUTS_DIR / "name_join_qa").resolve()),
    help="Output directory for QA artifacts",
)
@click.option(
    "--config",
    default=str((Path(__file__).parents[1] / "gosales" / "config.yaml").resolve()),
    help="Path to config.yaml (to resolve allow-list and defaults)",
)
def main(moneyball_view: str | None, top: int, outputs: str, config: str) -> None:
    """Name-join QA: coverage of Moneyball names -> dim_customer.customer_id.

    Writes summary JSON and CSVs for top unmapped names and per-department coverage.
    """
    cfg = load_config(config)
    db = getattr(cfg, "database", None)
    tables = dict(getattr(db, "source_tables", {}) or {})
    mb_view = moneyball_view or tables.get("moneyball_assets", "dbo.[Moneyball Assets]")

    # Validate identifier (and enforce allow-list if configured)
    try:
        allow = set(getattr(db, "allowed_identifiers", []) or [])
        if allow:
            mb_view = ensure_allowed_identifier(str(mb_view), allow)
        else:
            validate_identifier(str(mb_view))
    except Exception as e:
        raise ValueError(f"Invalid Moneyball view identifier: {e}")

    # Load dim_customer norms
    dc, dim_norms, ambiguous = _load_dim_customer_norms()
    logger.info("Loaded dim_customer with %d rows; %d unique normalized names", len(dc), len(dim_norms))

    # Aggregation state
    total_rows = 0
    mapped_rows = 0
    unique_norms: set[str] = set()
    mapped_unique_norms: set[str] = set()
    dept_totals: Dict[str, int] = defaultdict(int)
    dept_mapped: Dict[str, int] = defaultdict(int)
    # unmapped name -> (display_name, rows_count, qty_sum)
    unmapped: Dict[str, Tuple[str, int, float]] = {}

    for chunk in _iter_moneyball(mb_view):
        if chunk is None or chunk.empty:
            continue
        # Select minimal columns
        cols = [c for c in ["customer_name", "qty", "department"] if c in chunk.columns]
        df = chunk[cols].copy()
        df["customer_name_norm"] = _norm(df["customer_name"])
        df["qty_num"] = pd.to_numeric(df.get("qty", 1.0), errors="coerce").fillna(1.0)

        norms = df["customer_name_norm"].astype(str)
        is_mapped = norms.isin(dim_norms)

        total_rows += int(len(df))
        mapped_rows += int(is_mapped.sum())

        # Unique norms
        unique_chunk = set(norms.unique())
        unique_norms.update(unique_chunk)
        mapped_unique_norms.update({n for n in unique_chunk if n in dim_norms})

        # Department coverage
        dept_col = "department" if "department" in df.columns else None
        if dept_col:
            dept_vals = df[dept_col].astype(str).fillna("<NA>")
            for dep, m in zip(dept_vals, is_mapped):
                dept_totals[dep] += 1
                if bool(m):
                    dept_mapped[dep] += 1

        # Unmapped accumulation
        df_un = df.loc[~is_mapped].copy()
        for name_norm, g in df_un.groupby("customer_name_norm"):
            rows = int(len(g))
            qty_sum = float(g["qty_num"].sum()) if "qty_num" in g.columns else float(rows)
            # Pick the most frequent display variant
            try:
                display = (
                    g["customer_name"].astype(str).value_counts().idxmax()  # type: ignore
                )
            except Exception:
                display = str(name_norm)
            if name_norm in unmapped:
                prev_disp, prev_rows, prev_qty = unmapped[name_norm]
                unmapped[name_norm] = (
                    prev_disp,
                    prev_rows + rows,
                    prev_qty + qty_sum,
                )
            else:
                unmapped[name_norm] = (display, rows, qty_sum)

    unique_total = len(unique_norms)
    unique_mapped = len(mapped_unique_norms)
    cov_rows = float(mapped_rows / total_rows) if total_rows else 0.0
    cov_unique = float(unique_mapped / unique_total) if unique_total else 0.0

    # Build outputs
    out_dir = Path(outputs)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    summary = {
        "total_rows": int(total_rows),
        "mapped_rows": int(mapped_rows),
        "coverage_rows": cov_rows,
        "unique_names": int(unique_total),
        "mapped_unique_names": int(unique_mapped),
        "coverage_unique_names": cov_unique,
        "ambiguous_dim_norm_names": int(len(ambiguous)),
    }
    (out_dir / "name_join_qa_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Unmapped top-N
    if unmapped:
        un_rows = [
            {
                "customer_name_norm": k,
                "customer_name_example": v[0],
                "row_count": v[1],
                "qty_sum": v[2],
            }
            for k, v in unmapped.items()
        ]
        un_df = pd.DataFrame(un_rows).sort_values(
            ["row_count", "qty_sum"], ascending=[False, False]
        )
        un_df.head(int(top)).to_csv(
            out_dir / f"unmapped_names_top_{int(top)}.csv", index=False
        )

    # Department coverage
    if dept_totals:
        dep_rows = []
        for dep, tot in dept_totals.items():
            m = int(dept_mapped.get(dep, 0))
            dep_rows.append(
                {
                    "department": dep,
                    "rows_total": int(tot),
                    "rows_mapped": m,
                    "coverage_rows": float(m / tot) if tot else 0.0,
                }
            )
        pd.DataFrame(dep_rows).sort_values(
            ["coverage_rows", "rows_total"], ascending=[True, False]
        ).to_csv(out_dir / "coverage_by_department.csv", index=False)

    # Ambiguous dim_customer norms
    if len(ambiguous):
        ambiguous.to_csv(out_dir / "ambiguous_dim_customer_norms.csv", index=False)

    logger.info(
        "Name-join QA complete: %.2f%% row coverage; %.2f%% unique coverage",
        cov_rows * 100,
        cov_unique * 100,
    )
    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
