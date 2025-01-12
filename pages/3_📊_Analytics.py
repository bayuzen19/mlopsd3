import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from config.config import Config
import json
from utils.styling import load_css

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

load_css()

# Load and cache data with proper cache decorator
@st.cache_data(ttl=3600)
def load_data():
    """Load and cache the dataset"""
    try:
        return pd.read_csv(Config.DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data(ttl=3600)
def load_model_artifacts():
    """Load and cache model metrics and feature importance"""
    try:
        metrics = None
        feature_importance = None
        
        # Load metrics if available
        if Config.METRICS_PATH.exists():
            with open(Config.METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        
        # Load feature importance if available
        if Config.FEATURE_IMPORTANCE_PATH.exists():
            with open(Config.FEATURE_IMPORTANCE_PATH, 'r') as f:
                feature_importance = json.load(f)
                
        return metrics, feature_importance
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

try:
    # Load data and model artifacts
    df = load_data()
    metrics, feature_importance = load_model_artifacts()

    if df is not None:
        st.title("📊 Data Analytics & Model Performance")

        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "Data Analysis", 
            "Feature Relationships", 
            "Model Performance"
        ])

        with tab1:
            st.header("Data Distribution Analysis")
            
            feature = st.selectbox(
                "Select Feature",
                Config.FEATURE_COLUMNS
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df, 
                    x=feature,
                    marginal="box",
                    title=f"Distribution of {feature}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                stats = df[feature].describe()
                st.dataframe(stats)

        with tab2:
            st.header("Feature Relationships")
            
            corr = df[Config.FEATURE_COLUMNS + [Config.TARGET_COLUMN]].corr()
            fig = px.imshow(
                corr,
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("Select X-axis feature", Config.FEATURE_COLUMNS)
            with col2:
                y_feature = st.selectbox(
                    "Select Y-axis feature", 
                    [Config.TARGET_COLUMN], 
                    index=1 if len([Config.TARGET_COLUMN]) > 1 else 0
                )
            
            fig = px.scatter(
                df,
                x=x_feature,
                y=y_feature,
                title=f"{x_feature} vs {y_feature}",
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.header("Model Performance")
            
            if metrics and feature_importance:
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test R² Score", f"{metrics['test_r2']:.4f}")
                with col2:
                    st.metric("Test RMSE", f"${metrics['test_rmse']:,.2f}")
                with col3:
                    st.metric("Test MAE", f"${metrics['test_mae']:,.2f}")
                
                # Training vs Testing Performance
                st.subheader("Training vs Testing Performance")
                metrics_df = pd.DataFrame({
                    'Metric': ['R²', 'RMSE', 'MAE'],
                    'Training': [
                        metrics['train_r2'],
                        metrics['train_rmse'],
                        metrics['train_mae']
                    ],
                    'Testing': [
                        metrics['test_r2'],
                        metrics['test_rmse'],
                        metrics['test_mae']
                    ]
                })
                st.dataframe(metrics_df)

                # Feature Importance Section
                st.subheader("Feature Importance Analysis")
                
                # Create DataFrame from feature importance
                importance_df = pd.DataFrame({
                    'Feature': list(feature_importance.keys()),
                    'Importance': list(feature_importance.values())
                }).sort_values('Importance', ascending=True)

                # Horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color='rgb(26, 118, 255)',
                        line=dict(color='rgba(26, 118, 255, 1.0)', width=1)
                    )
                ))

                fig.update_layout(
                    title='Feature Importance',
                    xaxis_title='Importance Score',
                    yaxis_title='Features',
                    template='plotly_white',
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Feature Importance Details
                with st.expander("Feature Importance Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Top Features")
                        top_features = importance_df.tail(3).copy()
                        top_features['Importance (%)'] = top_features['Importance'] * 100
                        st.dataframe(
                            top_features[['Feature', 'Importance (%)']].round(2)
                        )
                    
                    with col2:
                        st.markdown("### Feature Importance Distribution")
                        fig = px.pie(
                            importance_df,
                            values='Importance',
                            names='Feature',
                            title='Feature Importance Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Model metrics and feature importance not available. Please train the model first.")

except Exception as e:
    st.error(f"Error in analytics: {str(e)}")