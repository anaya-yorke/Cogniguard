import numpy as np
from scipy import signal

def preprocess_eeg(eeg_data, sampling_rate=250):
    # Apply bandpass filter (1-45 Hz)
    sos = signal.butter(4, [1, 45], 'bandpass', fs=sampling_rate, output='sos')
    filtered = signal.sosfilt(sos, eeg_data, axis=0)
    
    # Normalize data (per channel)
    normalized = np.zeros_like(filtered)
    for i in range(filtered.shape[1]):
        normalized[:, i] = (filtered[:, i] - filtered[:, i].mean()) / (filtered[:, i].std() + 1e-8)
    
    return normalized

def segment_eeg(eeg_data, window_size=250, step_size=125):
    # Segment the continuous EEG into overlapping windows
    segments = []
    for i in range(0, len(eeg_data) - window_size + 1, step_size):
        segments.append(eeg_data[i:i+window_size])
    
    return np.array(segments)

def extract_features(eeg_segments):
    # Extract time-domain and frequency-domain features
    features = []
    
    for segment in eeg_segments:
        segment_features = []
        
        for channel in range(segment.shape[1]):
            # Time domain features
            mean = np.mean(segment[:, channel])
            std = np.std(segment[:, channel])
            max_val = np.max(segment[:, channel])
            min_val = np.min(segment[:, channel])
            
            # Frequency domain features (power in different bands)
            f, psd = signal.welch(segment[:, channel], fs=250, nperseg=250)
            
            # Delta (1-4 Hz)
            delta_idx = np.logical_and(f >= 1, f <= 4)
            delta_power = np.sum(psd[delta_idx])
            
            # Theta (4-8 Hz)
            theta_idx = np.logical_and(f >= 4, f <= 8)
            theta_power = np.sum(psd[theta_idx])
            
            # Alpha (8-12 Hz)
            alpha_idx = np.logical_and(f >= 8, f <= 12)
            alpha_power = np.sum(psd[alpha_idx])
            
            # Beta (12-30 Hz)
            beta_idx = np.logical_and(f >= 12, f <= 30)
            beta_power = np.sum(psd[beta_idx])
            
            # Gamma (30-45 Hz)
            gamma_idx = np.logical_and(f >= 30, f <= 45)
            gamma_power = np.sum(psd[gamma_idx])
            
            # Ratios between bands (cognitive load indicators)
            theta_beta_ratio = theta_power / (beta_power + 1e-8)
            alpha_beta_ratio = alpha_power / (beta_power + 1e-8)
            
            channel_features = [mean, std, max_val, min_val, 
                               delta_power, theta_power, alpha_power, beta_power, gamma_power,
                               theta_beta_ratio, alpha_beta_ratio]
            
            segment_features.extend(channel_features)
        
        features.append(segment_features)
    
    return np.array(features) 