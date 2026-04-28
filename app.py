#Car Price Prediction App - Main Application File
#s16798

## Import required libraries
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# Import page functions from pages folder

# Import Data Explorer page function
# This function renders the "📊 Data Explorer" page
from pages.data_explorer import page_data_explorer

# Import global plot layout configuration
# Used for consistent Plotly styling across pages
from utils.config import PLOT_LAYOUT

# Import regression/clustering page function
from pages.regression import page_regression

# Import Data visualisations page function
from pages.visualisations import page_visualisations

# Import hypothesis testing page function
from pages.hypothesis_testing import page_hypotesting

## Import help  page function
from pages.help import page_help

#Streamlit Page Configuration
st.set_page_config(
    page_title="Car.LK - Sri Lanka Car Market",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load External CSS File (using UTF-8)
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f: 
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/styles.css")


# DATA LOADING 
from utils.data_loader import load_data
df = load_data()

# SIDEBAR
with st.sidebar:
    try:
        st.image("logo.png", use_container_width=True)
    except Exception:
        st.markdown("""
        <div style='text-align:center;padding:20px 0 10px 0;'>
          <span style='font-family:Syne;font-size:2.2rem;font-weight:800;color:#f0c040;'>🚗car.LK</span><br>
          <span style='color:#7d8590;font-size:0.92rem;letter-spacing:2px;text-transform:uppercase;'>Sri Lanka Car Market</span>
        </div>""", unsafe_allow_html=True)
        
        

    st.markdown("<hr>", unsafe_allow_html=True)
    
    page = st.radio("Navigate",
        ["📊 Data Explorer","📉 Visualisations","📈Car Price Prediction","🧪 Hypothesis Testing","❓ Help"],
        label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.78rem;color:#7d8590;line-height:2.2;'>
      📦 <span style='color:#f0c040;font-weight:600;'>{len(df):,}</span> listings<br>
      🏷️ <span style='color:#f0c040;font-weight:600;'>{df["Brand"].nunique()}</span> brands<br>
      📅 <span style='color:#f0c040;font-weight:600;'>{df["YOM"].min()}–{df["YOM"].max()}</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:1.0rem;color:#7d8590;line-height:2.2;'>
      👤 <span style='color:#f0c040;font-weight:600;'>Dinusha Priyshan(s16798)</span><br>
      ✨ <span style='color:#f0c040;font-weight:600;'>ST3011</span><br>
      🎓 <span style='color:#f0c040;font-weight:600;'>Department of Statistics</span><br>
      🎓 <span style='color:#f0c040;font-weight:600;'>University of Colombo</span>
    </div>""", unsafe_allow_html=True)


# ROUTER — call the right function based on sidebar

if page == "📊 Data Explorer":
    page_data_explorer(df)
elif page == "📈Car Price Prediction":
    page_regression(df)
elif page == "📉 Visualisations":
    page_visualisations(df)
elif page == "🧪 Hypothesis Testing":
    page_hypotesting(df)
elif page == "❓ Help":
    page_help()



#Global Footer (Appears on All Pages) 
st.markdown("""
<div class="footer">
    🚗 <strong>Car.LK – Sri Lanka Car Market Analysis Dashboard</strong><br>
    © 2026 Dinusha Priyashan | Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)


