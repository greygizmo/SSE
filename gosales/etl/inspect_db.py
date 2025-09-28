"""Utility for snapshotting the current SQL schema into YAML documentation."""

import yaml
from sqlalchemy import inspect
from gosales.utils.db import get_db_connection
from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.logger import get_logger

logger = get_logger(__name__)

def inspect_db(engine, output_path):
    """Inspects the database and generates a YAML file with the schema.

    Args:
        engine (sqlalchemy.engine.base.Engine): The database engine.
        output_path (str): The path to the output YAML file.
    """
    logger.info(f"Inspecting database and generating schema...")
    inspector = inspect(engine)
    schema = {}
    for table_name in inspector.get_table_names():
        schema[table_name] = []
        for column in inspector.get_columns(table_name):
            schema[table_name].append(
                {
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                    "default": column["default"],
                }
            )

    with open(output_path, "w") as f:
        yaml.dump(schema, f, default_flow_style=False)

    logger.info(f"Successfully generated schema to {output_path}")

if __name__ == "__main__":
    # Get database connection
    db_engine = get_db_connection()

    # Define the output path for the schema YAML file
    output_path = OUTPUTS_DIR / "schema.yml"

    # Create the outputs directory if it doesn't exist
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Inspect the database and generate the schema
    inspect_db(db_engine, output_path)
