import torch
import numpy as np
import time
import argparse
import logging
import os
import streamlit as st
import sys

from .simulation import generate_eeg_data, simulate_cognitive_load, generate_simulated_dataset
from .preprocessing import preprocess_eeg, segment_eeg, extract_features
from .model import CNNLSTM, train_model
from .inference import predict_overload, generate_alert_with_cortex, run_real_time_monitoring
from .database import initialize_snowflake_connection, store_measurement

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("cogniguard.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("Cogniguard")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cogniguard: Cognitive Overload Detection')
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'monitor', 'dashboard'],
                        help='Operation mode: train, monitor, or dashboard')
    
    parser.add_argument('--model_path', type=str, default='models/cogniguard_model.pt',
                        help='Path to save/load the model')
    
    parser.add_argument('--duration', type=int, default=300,
                        help='Duration for monitoring (seconds)')
    
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Threshold for overload detection')
    
    parser.add_argument('--cortex_api_key', type=str, default=None,
                        help='Cortex API key for alert generation')
    
    parser.add_argument('--snowflake_user', type=str, default=None,
                        help='Snowflake username')
    
    parser.add_argument('--snowflake_password', type=str, default=None,
                        help='Snowflake password')
    
    parser.add_argument('--snowflake_account', type=str, default=None,
                        help='Snowflake account identifier')
    
    parser.add_argument('--snowflake_warehouse', type=str, default=None,
                        help='Snowflake warehouse')
    
    parser.add_argument('--snowflake_database', type=str, default='COGNIGUARD_DB',
                        help='Snowflake database name')
    
    parser.add_argument('--snowflake_schema', type=str, default='PUBLIC',
                        help='Snowflake schema name')
    
    return parser.parse_args()

def ensure_model_directory(model_path):
    """Ensure the directory for the model exists"""
    directory = os.path.dirname(model_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def train_and_save_model(args, logger):
    """Train a new model and save it"""
    ensure_model_directory(args.model_path)
    
    logger.info("Generating simulated training data...")
    X_train, y_train = generate_simulated_dataset(num_samples=1000)
    X_val, y_val = generate_simulated_dataset(num_samples=200)
    
    logger.info("Training model...")
    model = train_model(X_train, y_train, X_val, y_val, epochs=50)
    
    torch.save(model.state_dict(), args.model_path)
    logger.info(f"Model saved to {args.model_path}")
    
    return model

def load_model(args, logger):
    """Load an existing model or train a new one"""
    model = CNNLSTM()
    
    try:
        model.load_state_dict(torch.load(args.model_path))
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {args.model_path}: {str(e)}")
        logger.info("Training new model...")
        model = train_and_save_model(args, logger)
    
    return model

def run_monitoring(args, logger, snowflake_conn):
    """Run real-time monitoring"""
    logger.info("Loading model...")
    model = load_model(args, logger)
    
    logger.info(f"Starting monitoring for {args.duration} seconds...")
    overload_events = run_real_time_monitoring(model, args.duration, 2, args.cortex_api_key)
    
    logger.info(f"Monitoring complete. Detected {len(overload_events)} overload events.")
    return overload_events

def run_dashboard():
    """Run the Streamlit dashboard"""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(current_dir, "dashboard.py")
    
    # Run the Streamlit dashboard
    os.system(f"streamlit run {dashboard_path}")

def main():
    """Main function"""
    args = parse_arguments()
    logger = setup_logging()
    
    # Initialize Snowflake connection
    logger.info("Initializing Snowflake connection...")
    snowflake_conn = initialize_snowflake_connection(
        user=args.snowflake_user,
        password=args.snowflake_password,
        account=args.snowflake_account,
        warehouse=args.snowflake_warehouse,
        database=args.snowflake_database,
        schema=args.snowflake_schema
    )
    
    if args.mode == 'train':
        train_and_save_model(args, logger)
    elif args.mode == 'monitor':
        run_monitoring(args, logger, snowflake_conn)
    elif args.mode == 'dashboard':
        run_dashboard()
    else:
        logger.error(f"Invalid mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 