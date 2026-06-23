import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================================
# 1. PAGE CONFIGURATION & ASSET LOADING
# =====================================================================
st.set_page_config(
    page_title="Retail Analytics Dashboard", 
    page_icon="🛍️", 
    layout="wide"
)

@st.cache_resource
def load_production_assets():
    """Loads the saved models from your folder safely"""
    try:
        scaler = joblib.load('scaler.pkl')
        kmeans = joblib.load('kmeans_model.pkl')
        # 🌟 UPDATED LINE: Read the new lightweight parquet file format
        similarity_matrix = pd.read_parquet('item_similarity.parquet')
        return scaler, kmeans, similarity_matrix
    except FileNotFoundError as e:
        st.error(f"❌ Missing required file: {e.filename}. Make sure you copied your assets into this folder!")
        return None, None, None

scaler, kmeans, similarity_matrix = load_production_assets()

# =====================================================================
# THE REST OF YOUR APP.PY CODE STAYS EXACTLY THE SAME FROM BEFORE
# =====================================================================