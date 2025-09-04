from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from sqlalchemy import text

from gosales.utils.db import get_db_connection
from gosales.utils.config import load_config
from gosales.utils.logger import get_logger


logger = get_logger(__name__)


def _preview_sql_for_table(dialect: str, table: str) -> str:
    name = dialect.lower() if isinstance(dialect, str) else ""
    if name.startswith("mssql") or name == "pyodbc":
        return f"SELECT TOP 1 * FROM {table}"
    return f"SELECT * FROM {table} LIMIT 1"


@click.command()
@click.option("--table", "table_name", default=None, help="Logical table to test (e.g., sales_log)")
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
@click.option("--verbose/--no-verbose", default=True)
def main(table_name: Optional[str], config: str, verbose: bool) -> None:
    """Check DB connectivity and preview configured source tables.

    Resolves logical names via config.database.source_tables and runs a TOP 1 / LIMIT 1 query
    for DB-backed sources. For sources set to 'csv', prints the configured path.
    """
    cfg = load_config(config)
    engine = get_db_connection()
    src = getattr(getattr(cfg, "database", object()), "source_tables", {}) or {}
    etl = getattr(cfg, "etl", object())
    dialect = engine.dialect.name
    logger.info("Connected using SQLAlchemy dialect: %s", dialect)

    tests = []
    if table_name:
        tests = [table_name]
    else:
        tests = ["sales_log", "industry_enrichment"]

    ok = True
    for logical in tests:
        concrete = str(src.get(logical, logical)).strip()
        if not concrete:
            logger.warning("No mapping found for '%s'", logical)
            ok = False
            continue
        if concrete.lower() == "csv":
            if logical == "industry_enrichment":
                csv_path = getattr(etl, "industry_enrichment_csv", None)
                logger.info("%s -> CSV (%s)", logical, csv_path)
                try:
                    p = Path(csv_path) if csv_path else None
                    if p and p.exists():
                        df = pd.read_csv(p, nrows=1)
                        logger.info("CSV preview columns: %s", list(df.columns))
                    else:
                        logger.warning("CSV path missing: %s", csv_path)
                        ok = False
                except Exception as e:
                    logger.warning("CSV preview failed for %s: %s", logical, e)
                    ok = False
            else:
                logger.warning("%s mapped to CSV but no CSV preview implemented", logical)
            continue

        # DB-backed
        logger.info("%s -> %s", logical, concrete)
        try:
            sql = _preview_sql_for_table(dialect, concrete)
            with engine.connect() as conn:
                rs = conn.execute(text(sql))
                row = rs.fetchone()
                cols = rs.keys()
            if verbose:
                logger.info("Preview: columns=%s, sample_row=%r", list(cols), row)
        except Exception as e:
            logger.warning("DB preview failed for %s (%s): %s", logical, concrete, e)
            ok = False

    if not ok:
        logger.warning("One or more source checks failed.")
        sys.exit(1)
    logger.info("All configured sources are reachable.")


if __name__ == "__main__":
    main()

