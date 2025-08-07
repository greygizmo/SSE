import pandas as pd
from gosales.utils.db import get_db_connection
from gosales.utils.paths import DATA_DIR
from gosales.utils.logger import get_logger

logger = get_logger(__name__)

def load_csv_to_db(file_path: str, table_name: str, engine):
    """Loads a CSV file into a database table.

    Args:
        file_path (str): The path to the CSV file.
        table_name (str): The name of the table to create.
        engine (sqlalchemy.engine.base.Engine): The database engine.
    """
    logger.info(f"Loading {file_path} into table {table_name}...")
    
    # Try different encodings to handle encoding issues
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"Successfully read {file_path} with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"Failed to read {file_path} with encoding {encoding}: {e}")
            continue
    
    if df is None:
        raise ValueError(f"Could not read {file_path} with any of the attempted encodings")
    
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    logger.info(f"Successfully loaded {file_path} into table {table_name}.")

if __name__ == "__main__":
    # Get database connection
    db_engine = get_db_connection()

    # Define the CSV files and their corresponding table names
    csv_files = {
        "Analytics_order_tags.csv": "analytics_order_tags",
        "Analytics_SalesLogs.csv": "analytics_sales_logs",
        "Sales_Log.csv": "sales_log",
    }

    # Load each CSV file into the database
    for file_name, table_name in csv_files.items():
        file_path = DATA_DIR / "database_samples" / file_name
        load_csv_to_db(file_path, table_name, db_engine)
