import numpy as np
from scipy import signal

def generate_eeg_data(duration=10, sampling_rate=250, num_channels=8):
    num_samples = int(duration * sampling_rate)
    
    # Base signal generation (alpha, beta, theta waves)
    t = np.linspace(0, duration, num_samples)
    
    channels = []
    for _ in range(num_channels):
        # Create mixture of brain waves
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)  # Alpha: 8-12 Hz
        beta = 0.2 * np.sin(2 * np.pi * 20 * t)   # Beta: 13-30 Hz
        theta = 0.3 * np.sin(2 * np.pi * 5 * t)   # Theta: 4-7 Hz
        
        # Add noise
        noise = 0.2 * np.random.normal(0, 1, num_samples)
        
        # Combine signals
        eeg = alpha + beta + theta + noise
        channels.append(eeg)
        
    return np.array(channels).T  # Shape: (samples, channels)

def simulate_cognitive_load(eeg_data, load_level=0.5):
    # Simulate cognitive load by modifying beta/alpha ratio
    # Higher load = more beta, less alpha
    load_adjusted = eeg_data.copy()
    
    for i in range(eeg_data.shape[1]):
        # Apply bandpass filters to isolate components
        fs = 250  # Sampling frequency (Hz)
        
        # Extract alpha (8-12 Hz)
        alpha_band = signal.butter(4, [8, 12], 'bandpass', fs=fs, output='sos')
        alpha_component = signal.sosfilt(alpha_band, eeg_data[:, i])
        
        # Extract beta (13-30 Hz)
        beta_band = signal.butter(4, [13, 30], 'bandpass', fs=fs, output='sos')
        beta_component = signal.sosfilt(beta_band, eeg_data[:, i])
        
        # Modify ratio based on cognitive load
        alpha_scale = 1 - load_level
        beta_scale = 0.5 + load_level
        
        # Reconstruct the signal
        load_adjusted[:, i] = eeg_data[:, i] - alpha_component - beta_component + \
                              (alpha_component * alpha_scale) + (beta_component * beta_scale)
    
    return load_adjusted

def generate_simulated_dataset(num_samples=100, duration=2):
    X = []
    y = []
    
    for _ in range(num_samples):
        # Random cognitive load between 0 (none) and 1 (maximum)
        load = np.random.random()
        
        # Generate base EEG
        eeg_base = generate_eeg_data(duration=duration)
        
        # Apply cognitive load
        eeg_with_load = simulate_cognitive_load(eeg_base, load_level=load)
        
        X.append(eeg_with_load)
        
        # Binary classification: 1 if load > 0.7
        y.append(1 if load > 0.7 else 0)
    
    return np.array(X), np.array(y) 