import os
import urllib.parse
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

from gosales.utils.logger import get_logger
from gosales.utils.paths import ROOT_DIR
from gosales.utils.config import load_config

load_dotenv()

logger = get_logger(__name__)

def _build_pyodbc_conn(server: str, database: str, username: str, password: str, driver: str) -> str:
    params = (
        f"DRIVER={{{{drv}}}};SERVER={server};DATABASE={database};UID={username};PWD={password};"  # placeholders
        .replace("{{drv}}", driver)
    )
    # Default to encrypted connection; trust cert to simplify local dev
    params += "Encrypt=yes;TrustServerCertificate=yes;"
    odbc_connect = urllib.parse.quote_plus(params)
    return f"mssql+pyodbc:///?odbc_connect={odbc_connect}"

def _build_url_style_conn(server: str, database: str, username: str, password: str, driver: str) -> str:
    # Normalize server (drop tcp: prefix if present)
    srv = server
    if srv.lower().startswith("tcp:"):
        srv = srv[4:]
    user_enc = urllib.parse.quote_plus(username)
    pwd_enc = urllib.parse.quote_plus(password)
    drv_enc = urllib.parse.quote_plus(driver)
    return (
        f"mssql+pyodbc://{user_enc}:{pwd_enc}@{srv}/{database}?driver={drv_enc}&Encrypt=yes&TrustServerCertificate=yes"
    )

def get_db_connection():
    """Return a SQLAlchemy engine based on configuration and environment.

    * When ``cfg.database.engine`` is ``"azure"`` (or ``"auto"``) and the ``AZSQL_*``
      credentials are present, connect to Azure SQL via ``pyodbc``.
    * When the configured engine is ``"sqlite"`` or Azure credentials are missing,
      build a SQLite engine using the configured ``sqlite_path``.
    * When the configured engine is ``"duckdb"``, connect using the DuckDB driver and
      configured path (``database.duckdb_path`` when available, otherwise ``sqlite_path``).
    """

    server = os.getenv("AZSQL_SERVER")
    database = os.getenv("AZSQL_DB")
    username = os.getenv("AZSQL_USER")
    password = os.getenv("AZSQL_PWD")

    cfg = None
    strict_db = False
    sqlite_path = ROOT_DIR.parent / "gosales.db"
    engine_choice = "sqlite"
    duckdb_path = None

    try:
        cfg = load_config()
        db_cfg = getattr(cfg, "database", None)
        if db_cfg:
            engine_choice = str(getattr(db_cfg, "engine", engine_choice) or engine_choice).lower()
            strict_db = bool(getattr(db_cfg, "strict_db", strict_db))
            sqlite_path = Path(getattr(db_cfg, "sqlite_path", sqlite_path))
            duckdb_path = getattr(db_cfg, "duckdb_path", None)
            if duckdb_path:
                duckdb_path = Path(duckdb_path)
    except Exception as exc:
        logger.debug("Unable to load config; defaulting DB engine handling. Error: %s", exc)

    azure_credentials = all([server, database, username, password])

    def _connect_sqlite(path: Path):
        resolved = Path(path)
        logger.info("Using SQLite database at: %s", resolved)
        return create_engine(f"sqlite:///{resolved}")

    def _connect_duckdb(path: Path):
        resolved = Path(path)
        logger.info("Using DuckDB database at: %s", resolved)
        return create_engine(f"duckdb:///{resolved}")

    def _connect_azure() -> "Engine":
        logger.info("Connecting to Azure SQL database: %s/%s", server, database)
        last_err = None
        for drv in ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"):
            try:
                url = _build_pyodbc_conn(server, database, username, password, drv)
                eng = create_engine(url)
                with eng.connect() as conn:
                    conn.exec_driver_sql("SELECT 1")
                logger.info("Azure SQL connection established using driver: %s", drv)
                return eng
            except Exception as e:
                last_err = e
                try:
                    url2 = _build_url_style_conn(server, database, username, password, drv)
                    eng2 = create_engine(url2)
                    with eng2.connect() as conn:
                        conn.exec_driver_sql("SELECT 1")
                    logger.info("Azure SQL connection established using driver: %s (URL mode)", drv)
                    return eng2
                except Exception as e2:
                    last_err = e2
                    continue
        msg = (
            "Failed to connect to Azure SQL via pyodbc. Ensure Microsoft ODBC Driver 18 or 17 for SQL Server is installed "
            "and reachable. Last error: %r" % (last_err,)
        )
        raise RuntimeError(msg)

    engine_choice = engine_choice or "sqlite"

    if engine_choice in {"azure", "auto"}:
        if azure_credentials:
            return _connect_azure()
        if strict_db:
            raise RuntimeError("database.strict_db=True but AZSQL_* environment variables are not set")
        logger.warning(
            "Azure engine requested but AZSQL_* credentials are missing; falling back to SQLite at %s",
            sqlite_path,
        )
        return _connect_sqlite(sqlite_path)

    if engine_choice == "sqlite":
        return _connect_sqlite(sqlite_path)

    if engine_choice == "duckdb":
        target = duckdb_path or Path(sqlite_path).with_suffix(".duckdb")
        return _connect_duckdb(target)

    logger.warning(
        "Unknown database engine '%s'; defaulting to SQLite at %s",
        engine_choice,
        sqlite_path,
    )
    return _connect_sqlite(sqlite_path)


def get_curated_connection():
    """Return an engine for the curated target (local SQLite by default).

    Respects `database.curated_target` and `database.curated_sqlite_path` in config. If
    set to 'sqlite', returns a SQLite engine at the configured path; if 'db' (or
    unspecified), returns the primary DB engine from `get_db_connection()`.
    """
    try:
        cfg = load_config()
        db = getattr(cfg, 'database', None)
        target = str(getattr(db, 'curated_target', 'db')).lower() if db else 'db'
        if target == 'sqlite':
            p = getattr(db, 'curated_sqlite_path', None)
            if not p:
                p = ROOT_DIR.parent / 'gosales_curated.db'
            else:
                p = Path(p)
            return create_engine(f"sqlite:///{p}")
        # default: use primary connection
        return get_db_connection()
    except Exception:
        # Fallback to local curated.sqlite
        p = ROOT_DIR.parent / 'gosales_curated.db'
        return create_engine(f"sqlite:///{p}")


def validate_connection(engine) -> bool:
    """Validate DB connection health by executing a trivial query."""
    try:
        with engine.connect() as conn:
            try:
                conn.exec_driver_sql("SELECT 1")
            except Exception:
                conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error("Database connection validation failed: %s", e)
        return False
