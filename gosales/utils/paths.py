from pathlib import Path

# Define the project root directory
ROOT_DIR = Path(__file__).parent.parent

# Define paths to other important directories
DATA_DIR = ROOT_DIR / "data"
ETL_DIR = ROOT_DIR / "etl"
FEATURES_DIR = ROOT_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
UI_DIR = ROOT_DIR / "ui"
UTILS_DIR = ROOT_DIR / "utils"
PIPELINE_DIR = ROOT_DIR / "pipeline"
OUTPUTS_DIR = ROOT_DIR / "outputs"
