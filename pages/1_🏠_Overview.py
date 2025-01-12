import streamlit as st
import pandas as pd
import plotly.express as px
from config.config import Config
from utils.styling import load_css





st.set_page_config(page_title="Project Overview", page_icon="🏠", layout="wide")
load_css()

st.title("🏠 Boston House Price Prediction")

# Project Overview Section
st.markdown("""
## About This Project
This project uses machine learning to predict house prices in Boston based on various features. The model takes into 
account factors like crime rate, room numbers, and neighborhood characteristics to estimate property values.

### Dataset Information
The Boston Housing Dataset contains information collected by the U.S Census Service concerning housing in the area of Boston MA.
""")

# Load and cache data
@st.cache_data
def load_data():
    return pd.read_csv(Config.DATA_PATH)

try:
    df = load_data()
    
    # Dataset Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Houses", f"{len(df):,}")
    with col2:
        st.metric("Average Price", f"${df['MEDV'].mean():,.2f}")
    with col3:
        st.metric("Features", f"{len(Config.FEATURE_COLUMNS)}")

    # Feature Descriptions
    st.header("📝 Feature Descriptions")
    descriptions = pd.DataFrame.from_dict(
        Config.FEATURE_DESCRIPTIONS, 
        orient='index',
        columns=['Description']
    )
    st.table(descriptions)

    # Data Sample
    with st.expander("View Sample Data"):
        st.dataframe(df.head())

    # Basic Statistics
    st.header("📊 Basic Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig = px.histogram(
            df, 
            x='MEDV',
            title="House Price Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Price by Room Count")
        fig = px.box(
            df, 
            x='RM', 
            y='MEDV',
            title="Price Distribution by Number of Rooms"
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")