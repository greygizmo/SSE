import os
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

from gosales.utils.logger import get_logger
from gosales.utils.paths import ROOT_DIR

load_dotenv()

logger = get_logger(__name__)

def get_db_connection():
    """Establishes a connection to the database using credentials from the .env file.

    Returns:
        sqlalchemy.engine.base.Engine: The database engine.
    """

    # Get database credentials from environment variables
    server = os.getenv("AZSQL_SERVER")
    database = os.getenv("AZSQL_DB")
    username = os.getenv("AZSQL_USER")
    password = os.getenv("AZSQL_PWD")

    if all([server, database, username, password]):
        logger.info(f"Connecting to Azure SQL database: {server}/{database}")
        connection_string = (
            "mssql+pyodbc:///?odbc_connect="
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password}"
        )
        return create_engine(connection_string)
    else:
        logger.info("Azure SQL credentials not found, falling back to SQLite.")
        db_path = ROOT_DIR.parent / "gosales.db"
        logger.info(f"Using SQLite database at: {db_path}")
        return create_engine(f"sqlite:///{db_path}")
