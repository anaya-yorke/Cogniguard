import requests
import json
import torch
import numpy as np
import time
from .simulation import generate_eeg_data, simulate_cognitive_load
from .preprocessing import preprocess_eeg, segment_eeg, extract_features

def predict_overload(model, eeg_data, threshold=0.7):
    model.eval()
    
    processed_data = preprocess_eeg(eeg_data)
    segmented_data = segment_eeg(processed_data)
    
    predictions = []
    
    with torch.no_grad():
        for segment in segmented_data:
            segment_tensor = torch.FloatTensor(segment).unsqueeze(0)
            prediction = model(segment_tensor).item()
            predictions.append(prediction)
    
    avg_prediction = np.mean(predictions) if predictions else 0.0
    is_overloaded = avg_prediction > threshold
    
    return {
        'is_overloaded': bool(is_overloaded),
        'overload_probability': float(avg_prediction),
        'timestamp': time.time()
    }

def generate_alert_with_cortex(prediction_result, api_key=None):
    if not prediction_result['is_overloaded']:
        return None
    
    cortex_api_url = "https://api.cortex.ai/v1/alerts"
    
    payload = {
        "overload_probability": prediction_result['overload_probability'],
        "timestamp": prediction_result['timestamp'],
        "prompt": "Create a helpful alert for a user experiencing cognitive overload"
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "Bearer YOUR_CORTEX_API_KEY"
    }
    
    try:
        response = requests.post(cortex_api_url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            return response.json()
        else:
            # For demo purposes, create a response when API call fails
            return {"message": "Take a break, you might be experiencing cognitive overload."}
    except requests.exceptions.RequestException:
        # If there's a connection error, we'll still provide a fallback
        return {"message": "Take a break, you might be experiencing cognitive overload."}

def run_real_time_monitoring(model, monitoring_duration=300, sample_interval=2, api_key=None):
    print("Starting real-time cognitive load monitoring...")
    
    start_time = time.time()
    end_time = start_time + monitoring_duration
    
    overload_events = []
    
    while time.time() < end_time:
        eeg_data = generate_eeg_data(duration=5)
        
        cognitive_load = np.random.random()
        eeg_with_load = simulate_cognitive_load(eeg_data, load_level=cognitive_load)
        
        prediction = predict_overload(model, eeg_with_load)
        
        if prediction['is_overloaded']:
            alert = generate_alert_with_cortex(prediction, api_key)
            if alert and 'error' not in alert:
                overload_events.append({
                    'timestamp': prediction['timestamp'],
                    'probability': prediction['overload_probability'],
                    'alert_message': alert.get('message', 'Take a break, you might be experiencing cognitive overload.')
                })
                print(f"Overload detected! Probability: {prediction['overload_probability']:.2f}")
                print(f"Alert: {alert.get('message', 'Take a break')}")
        
        time.sleep(sample_interval)
    
    return overload_events 