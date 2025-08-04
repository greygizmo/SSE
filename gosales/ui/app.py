import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from gosales.utils.paths import OUTPUTS_DIR
except ImportError:
    # Fallback if gosales module isn't found
    OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

st.set_page_config(layout="wide")

st.title("GoSales - ICP & Whitespace Engine")

st.write("This application displays the results of the GoSales analysis.")

st.header("ICP Scores")

try:
    icp_scores = pd.read_csv(OUTPUTS_DIR / "icp_scores.csv")
    st.dataframe(icp_scores)
except FileNotFoundError:
    st.warning("ICP scores not available.")


st.header("Whitespace Opportunities")

try:
    whitespace = pd.read_csv(OUTPUTS_DIR / "whitespace.csv")
    st.dataframe(whitespace)
except FileNotFoundError:
    st.warning("Whitespace opportunities not available.")
