"""Streamlit dashboard for {{ project_name }}."""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="{{ project_name }}",
    page_icon="📊",
    layout="wide",
)

st.title("📊 {{ project_name }}")
st.markdown("*Hackathon Dashboard*")

# Sidebar
st.sidebar.header("Configuration")
model_type = st.sidebar.selectbox("Model Type", ["Random Forest", "XGBoost", "LightGBM"])

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Overview")
    st.info("Upload your data or use the competition data in `data/raw/`")

with col2:
    st.subheader("Model Performance")
    st.metric("Accuracy", "—", help="Train a model first")

st.divider()

st.subheader("Make Predictions")
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
    if st.button("Predict"):
        st.success("Predictions generated! (Connect your model here)")
