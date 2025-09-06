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
    """Establish a connection to Azure SQL when AZSQL_* are present; fallback to SQLite.

    Tries multiple ODBC drivers for robustness (18, then 17) and enables encryption by default.
    """

    server = os.getenv("AZSQL_SERVER")
    database = os.getenv("AZSQL_DB")
    username = os.getenv("AZSQL_USER")
    password = os.getenv("AZSQL_PWD")

    # Strict mode: require Azure SQL and do not fall back when configured
    try:
        cfg = load_config()
        strict_db = bool(getattr(getattr(cfg, 'database', object()), 'strict_db', False))
    except Exception:
        strict_db = False

    if all([server, database, username, password]):
        logger.info(f"Connecting to Azure SQL database: {server}/{database}")
        last_err = None
        for drv in ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"):
            try:
                # First try odbc_connect param style
                url = _build_pyodbc_conn(server, database, username, password, drv)
                eng = create_engine(url)
                # Lazy connect test: try a trivial query
                with eng.connect() as conn:
                    conn.exec_driver_sql("SELECT 1")
                return eng
            except Exception as e:
                last_err = e
                # Fallback: try URL style with driver query param
                try:
                    url2 = _build_url_style_conn(server, database, username, password, drv)
                    eng2 = create_engine(url2)
                    with eng2.connect() as conn:
                        conn.exec_driver_sql("SELECT 1")
                    return eng2
                except Exception as e2:
                    last_err = e2
                    continue
        # If all attempts failed, raise a helpful message
        msg = (
            "Failed to connect to Azure SQL via pyodbc. Ensure Microsoft ODBC Driver 18 or 17 for SQL Server is installed "
            "and reachable. Last error: %r" % (last_err,)
        )
        raise RuntimeError(msg)
    else:
        if strict_db:
            raise RuntimeError("database.strict_db=True but AZSQL_* environment variables are not set")
        logger.info("Azure SQL credentials not found, falling back to SQLite.")
        db_path = ROOT_DIR.parent / "gosales.db"
        logger.info(f"Using SQLite database at: {db_path}")
        return create_engine(f"sqlite:///{db_path}")


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
