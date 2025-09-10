from __future__ import annotations

from pathlib import Path
from typing import Optional
import click
import pandas as pd
from sqlalchemy import text

from gosales.utils.db import get_db_connection
from gosales.utils.config import load_config
from gosales.utils.logger import get_logger
from gosales.utils.sql import validate_identifier
from gosales.sql.queries import top_n_preview


logger = get_logger(__name__)


@click.command()
@click.option("--view", required=True, help="Schema-qualified view/table name, e.g., dbo.saleslog")
@click.option("--rows", default=5, help="Number of sample rows to show")
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
def main(view: str, rows: int, config: str) -> None:
    """Inspect a DB view: list columns and show a few sample rows (no writes)."""
    _ = load_config(config)
    eng = get_db_connection()
    # Minimal identifier validation to mitigate injection
    try:
        validate_identifier(view)
    except Exception as e:
        logger.error("Invalid view identifier: %s", e)
        return
    with eng.connect() as conn:
        # Columns via INFORMATION_SCHEMA when available; fallback to LIMIT 0 select
        cols = []
        try:
            q = text(
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = :tname"
            )
            # Extract table_name portion if schema-qualified
            tname = view.split(".")[-1]
            res = conn.execute(q, {"tname": tname})
            cols = [r[0] for r in res.fetchall()]
        except Exception:
            try:
                res = conn.execute(text(f"SELECT * FROM {view} WHERE 1=0"))
                cols = list(res.keys())
            except Exception:
                cols = []
        logger.info("%s columns (%d): %s", view, len(cols), cols)

        try:
            dialect = eng.dialect.name
            top_sql = top_n_preview(view, dialect, n=int(rows))
            df = pd.read_sql(top_sql, eng)
        except Exception:
            df = pd.read_sql(top_n_preview(view, "sqlite", n=int(rows)), eng)
        if not df.empty:
            logger.info("Sample rows:\n%s", df.to_string(index=False))
        else:
            logger.info("No rows returned for sample.")


if __name__ == "__main__":
    main()
