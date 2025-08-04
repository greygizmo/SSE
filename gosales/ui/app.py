import streamlit as st
import pandas as pd
from gosales.utils.paths import OUTPUTS_DIR

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
