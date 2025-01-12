import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from config.config import Config
from utils.styling import load_css

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

# Load CSS
load_css()

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Prediction Form
with st.form("prediction_form"):
    st.subheader("Enter House Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lstat = st.slider(
            "Lower Status Population (%)",
            min_value=0.0,
            max_value=40.0,
            value=10.0,
            help=Config.FEATURE_DESCRIPTIONS['LSTAT']
        )
        
        rm = st.slider(
            "Average Rooms",
            min_value=3.0,
            max_value=9.0,
            value=6.0,
            help=Config.FEATURE_DESCRIPTIONS['RM']
        )
        
        crim = st.number_input(
            "Crime Rate",
            min_value=0.0,
            max_value=100.0,
            value=0.1,
            step=0.01,
            help=Config.FEATURE_DESCRIPTIONS['CRIM']
        )
    
    with col2:
        ptratio = st.slider(
            "Pupil-Teacher Ratio",
            min_value=12.0,
            max_value=22.0,
            value=15.0,
            help=Config.FEATURE_DESCRIPTIONS['PTRATIO']
        )
        
        indus = st.slider(
            "Industrial Area (%)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            help=Config.FEATURE_DESCRIPTIONS['INDUS']
        )
        
        tax = st.slider(
            "Property Tax Rate",
            min_value=150.0,
            max_value=800.0,
            value=300.0,
            help=Config.FEATURE_DESCRIPTIONS['TAX']
        )
    
    with col3:
        nox = st.slider(
            "Nitric Oxide Concentration",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            help=Config.FEATURE_DESCRIPTIONS['NOX']
        )
        
        b = st.slider(
            "Black Population Ratio",
            min_value=0.0,
            max_value=400.0,
            value=300.0,
            help=Config.FEATURE_DESCRIPTIONS['B']
        )
    
    submitted = st.form_submit_button("🏠 Predict Price")

if submitted:
    input_data = {
        "LSTAT": lstat,
        "RM": rm,
        "CRIM": crim,
        "PTRATIO": ptratio,
        "INDUS": indus,
        "TAX": tax,
        "NOX": nox,
        "B": b
    }

# Update API endpoint URL untuk Docker
# API_URL = "http://fastapi:8000"  # Gunakan nama service dari docker-compose
    
    try:
        with st.spinner('Making prediction...'):
            response = requests.post(
                "http://localhost:8000/predict",
                json=input_data
            )
            
            #--- Jika pake docker ---
            # # Update prediction request
            # response = requests.post(
            #     f"{API_URL}/predict",
            #     json=input_data
            # )
            
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                
                # Store prediction
                st.session_state.predictions.append({
                    "prediction": prediction,
                    **input_data
                })
                
                st.success(f"### Predicted House Price: ${prediction:,.2f}")
                
                # Display feature values
                st.subheader("Feature Values Used")
                feature_df = pd.DataFrame([input_data]).T
                feature_df.columns = ['Value']
                st.dataframe(feature_df)
                
            else:
                st.error(f"Error making prediction: {response.text}")
                
    except requests.exceptions.ConnectionError:
        st.error("Error connecting to the prediction service. Please make sure the API is running.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display prediction history
if st.session_state.predictions:
    st.header("Prediction History")
    
    df_pred = pd.DataFrame(st.session_state.predictions)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Recent Predictions")
        for idx, pred in enumerate(df_pred.tail(5).iloc[::-1].to_dict('records')):
            with st.expander(f"Prediction {len(df_pred) - idx}", expanded=idx == 0):
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Price", f"${pred['prediction']:,.2f}")
                with cols[1]:
                    st.metric("Rooms", f"{pred['RM']:.1f}")
                with cols[2]:
                    st.metric("Crime Rate", f"{pred['CRIM']:.4f}")
                with cols[3]:
                    st.metric("Tax Rate", f"{pred['TAX']:.1f}")
    
    with col2:
        if st.button("Clear Prediction History"):
            st.session_state.predictions = []
            st.experimental_rerun()
    
    # Visualization section
    st.header("Prediction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot of predictions vs rooms
        fig1 = px.scatter(
            df_pred,
            x='RM',
            y='prediction',
            title='Predicted Price vs Number of Rooms',
            labels={
                'RM': 'Number of Rooms',
                'prediction': 'Predicted Price ($)'
            }
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Create manual room ranges for grouping
        df_pred['Room_Range'] = pd.cut(
            df_pred['RM'],
            bins=[2, 4, 5, 6, 7, 8, 9],
            labels=['2-4', '4-5', '5-6', '6-7', '7-8', '8-9']
        )
        
        fig2 = px.box(
            df_pred,
            x='Room_Range',
            y='prediction',
            title='Price Distribution by Room Ranges',
            labels={
                'Room_Range': 'Room Ranges',
                'prediction': 'Predicted Price ($)'
            }
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Statistics
    st.subheader("Prediction Statistics")
    stats_cols = st.columns(4)
    
    with stats_cols[0]:
        st.metric("Average Price", f"${df_pred['prediction'].mean():,.2f}")
    with stats_cols[1]:
        st.metric("Highest Price", f"${df_pred['prediction'].max():,.2f}")
    with stats_cols[2]:
        st.metric("Lowest Price", f"${df_pred['prediction'].min():,.2f}")
    with stats_cols[3]:
        st.metric("Total Predictions", len(df_pred))
    
    # Download predictions
    if not df_pred.empty:
        st.download_button(
            label="Download Prediction History",
            data=df_pred.to_csv(index=False).encode('utf-8'),
            file_name="house_price_predictions.csv",
            mime="text/csv"
        )

else:
    st.info("No predictions made yet. Use the form above to make predictions.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)