import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import torch

from .simulation import generate_eeg_data, simulate_cognitive_load
from .preprocessing import extract_features, segment_eeg
from .inference import predict_overload, generate_alert_with_cortex
from .database import initialize_snowflake_connection, store_measurement, get_historical_measurements, get_overload_statistics

def create_dashboard(snowflake_conn, model, cortex_api_key=None):
    st.title("Cogniguard: Cognitive Overload Detection")
    
    tab1, tab2, tab3 = st.tabs(["Real-time Monitoring", "Historical Data", "Statistics"])
    
    with tab1:
        st.header("Real-time Cognitive Load")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_monitoring = st.button("Start Monitoring")
            monitoring_duration = st.slider("Monitoring Duration (seconds)", 30, 600, 120)
        
        with col2:
            alert_threshold = st.slider("Overload Alert Threshold", 0.5, 0.9, 0.7)
        
        if start_monitoring:
            progress_text = "Monitoring cognitive load..."
            progress_bar = st.progress(0)
            
            eeg_chart = st.line_chart()
            probability_chart = st.line_chart()
            
            alert_placeholder = st.empty()
            
            start_time = time.time()
            end_time = start_time + monitoring_duration
            
            while time.time() < end_time:
                elapsed = time.time() - start_time
                progress = elapsed / monitoring_duration
                progress_bar.progress(min(progress, 1.0))
                
                eeg_data = generate_eeg_data(duration=1)
                
                cognitive_load = 0.3 + 0.6 * np.sin(elapsed / 20) ** 2
                eeg_with_load = simulate_cognitive_load(eeg_data, load_level=cognitive_load)
                
                eeg_chart.add_rows({f"Channel {i+1}": eeg_with_load[-50:, i].mean() for i in range(min(4, eeg_with_load.shape[1]))})
                
                prediction = predict_overload(model, eeg_with_load, threshold=alert_threshold)
                probability_chart.add_rows({"Overload Probability": prediction['overload_probability']})
                
                if prediction['is_overloaded']:
                    alert = generate_alert_with_cortex(prediction, cortex_api_key)
                    if alert and 'error' not in alert:
                        alert_message = alert.get('message', 'Take a break, you might be experiencing cognitive overload.')
                        alert_placeholder.error(f"🚨 ALERT: {alert_message}")
                        
                        store_measurement(
                            snowflake_conn, 
                            prediction, 
                            eeg_features=extract_features(segment_eeg(eeg_with_load)), 
                            alert_message=alert_message
                        )
                else:
                    store_measurement(
                        snowflake_conn, 
                        prediction, 
                        eeg_features=extract_features(segment_eeg(eeg_with_load))
                    )
                
                time.sleep(0.1)
            
            progress_bar.progress(1.0)
            st.success("Monitoring complete!")
    
    with tab2:
        st.header("Historical Cognitive Load Data")
        
        days_ago = st.slider("Show data from last X days", 1, 30, 7)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_ago)
        
        historical_data = get_historical_measurements(
            snowflake_conn, 
            start_time=start_date, 
            end_time=end_date, 
            limit=1000
        )
        
        if not historical_data.empty:
            st.subheader("Overload Probability Over Time")
            
            fig = px.line(
                historical_data, 
                x='TIMESTAMP', 
                y='OVERLOAD_PROBABILITY',
                color_discrete_sequence=['blue'],
                title="Cognitive Overload Probability"
            )
            
            fig.add_hline(
                y=0.7, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Alert Threshold"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Overload Events")
            
            overload_events = historical_data[historical_data['IS_OVERLOADED']]
            if not overload_events.empty:
                st.dataframe(
                    overload_events[['TIMESTAMP', 'OVERLOAD_PROBABILITY', 'ALERT_MESSAGE']], 
                    use_container_width=True
                )
            else:
                st.info("No overload events in the selected time period.")
        else:
            st.info("No historical data available yet.")
    
    with tab3:
        st.header("Cognitive Load Statistics")
        
        stats_period = st.selectbox("Statistics Period", [7, 14, 30, 90], index=1)
        
        stats_data = get_overload_statistics(snowflake_conn, period_days=stats_period)
        
        if not stats_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Daily Overload Count")
                
                daily_fig = px.bar(
                    stats_data,
                    x='DAY',
                    y='OVERLOAD_COUNT',
                    title="Number of Overload Events per Day"
                )
                
                st.plotly_chart(daily_fig, use_container_width=True)
            
            with col2:
                st.subheader("Average Overload Probability")
                
                avg_fig = px.line(
                    stats_data,
                    x='DAY',
                    y='AVG_PROBABILITY',
                    title="Average Daily Overload Probability"
                )
                
                st.plotly_chart(avg_fig, use_container_width=True)
            
            st.subheader("Overload Ratio")
            
            stats_data['OVERLOAD_RATIO'] = stats_data['OVERLOAD_COUNT'] / stats_data['TOTAL_MEASUREMENTS']
            
            ratio_fig = px.area(
                stats_data,
                x='DAY',
                y='OVERLOAD_RATIO',
                title="Ratio of Overload Events to Total Measurements"
            )
            
            st.plotly_chart(ratio_fig, use_container_width=True)
        else:
            st.info("No statistics available yet.")

def main():
    st.set_page_config(
        page_title="Cogniguard Dashboard", 
        page_icon="🧠", 
        layout="wide"
    )
    
    # For demo purposes
    cortex_api_key = st.sidebar.text_input("Cortex API Key (Optional)", type="password")
    
    # Snowflake credentials (optional)
    st.sidebar.header("Snowflake Connection (Optional)")
    use_snowflake = st.sidebar.checkbox("Use Snowflake Connection", value=False)
    
    if use_snowflake:
        snowflake_user = st.sidebar.text_input("Snowflake User")
        snowflake_password = st.sidebar.text_input("Snowflake Password", type="password")
        snowflake_account = st.sidebar.text_input("Snowflake Account")
        snowflake_warehouse = st.sidebar.text_input("Snowflake Warehouse")
        snowflake_database = st.sidebar.text_input("Snowflake Database", value="COGNIGUARD_DB")
        snowflake_schema = st.sidebar.text_input("Snowflake Schema", value="PUBLIC")
        
        snowflake_conn = initialize_snowflake_connection(
            user=snowflake_user,
            password=snowflake_password,
            account=snowflake_account,
            warehouse=snowflake_warehouse,
            database=snowflake_database,
            schema=snowflake_schema
        )
    else:
        snowflake_conn = initialize_snowflake_connection()
    
    from .model import CNNLSTM
    from .simulation import generate_simulated_dataset
    
    st.sidebar.header("Model Options")
    load_pretrained = st.sidebar.checkbox("Load Pretrained Model", value=False)
    
    if load_pretrained:
        try:
            model_path = st.sidebar.text_input("Model Path", value="models/cogniguard_model.pt")
            model = CNNLSTM()
            model.load_state_dict(torch.load(model_path))
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            st.sidebar.info("Training new model...")
            X_train, y_train = generate_simulated_dataset(num_samples=200)
            X_val, y_val = generate_simulated_dataset(num_samples=50)
            
            from .model import train_model
            model = train_model(X_train, y_train, X_val, y_val, epochs=10)
    else:
        # Train a model with simulated data
        with st.sidebar.spinner("Training model..."):
            X_train, y_train = generate_simulated_dataset(num_samples=200)
            X_val, y_val = generate_simulated_dataset(num_samples=50)
            
            from .model import train_model
            model = train_model(X_train, y_train, X_val, y_val, epochs=10)
        
        st.sidebar.success("Model trained successfully!")
    
    create_dashboard(snowflake_conn, model, cortex_api_key)

if __name__ == "__main__":
    main() 