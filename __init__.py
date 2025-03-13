"""
Cogniguard: Cognitive Overload Detection System
===============================================

Modules:
    - simulation: EEG data simulation
    - preprocessing: Signal processing pipeline
    - model: CNN-LSTM neural network model
    - inference: Real-time inference and Cortex API
    - database: Snowflake integration
    - dashboard: Streamlit dashboard interface
    - app: Main application
"""

from .simulation import generate_eeg_data, simulate_cognitive_load, generate_simulated_dataset
from .preprocessing import preprocess_eeg, segment_eeg, extract_features
from .model import CNNLSTM, train_model, transfer_learning
from .inference import predict_overload, generate_alert_with_cortex, run_real_time_monitoring
from .database import initialize_snowflake_connection, store_measurement, get_historical_measurements, get_overload_statistics

__version__ = '0.1.0' 