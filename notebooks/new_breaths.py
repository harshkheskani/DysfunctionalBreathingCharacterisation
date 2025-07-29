import numpy as np
import pandas as pd
from scipy.signal import find_peaks, detrend, butter, filtfilt
import warnings

import pandas as pd
import numpy as np
from scipy.signal import detrend, find_peaks, butter, filtfilt, savgol_filter
from skimage.filters import threshold_otsu



def _calibrate_orientation_thresholds(signal, sampling_rate, fallback_low=0.15, fallback_high=0.5):
    """
    Automatically determines orientation thresholds using Otsu's method on the signal's DC offset.
    
    This function analyzes the distribution of the signal's DC component, assuming a
    bimodal distribution corresponding to supine (high DC offset) and lateral (low DC offset) positions.
    
    Returns:
    - (low_threshold, high_threshold): A tuple of floats for distinguishing orientations.
    """

    try:
        # 1. Calculate the moving DC offset using a 15-second window
        window_len = int(15 * sampling_rate)
        if len(signal) < window_len * 2: # Need enough data for meaningful stats
            raise ValueError("Signal too short for robust calibration.")

        # Use convolution for an efficient moving average to find the DC component
        dc_series = np.convolve(signal, np.ones(window_len) / window_len, mode='valid')
        abs_dc_series = np.abs(dc_series)
        
        # Ensure there is variance in the signal for Otsu to work
        if np.std(abs_dc_series) < 1e-4:
             raise ValueError("No significant change in DC offset; calibration not possible.")

        # 2. Apply Otsu's method to find the optimal split point
        # Otsu's method works on integer histograms, so we scale the data to an integer range.
        # This preserves precision while allowing the algorithm to work.
        scaler = 10000
        scaled_data = (abs_dc_series * scaler).astype(int)
        
        otsu_threshold_scaled = threshold_otsu(scaled_data)
        otsu_threshold = otsu_threshold_scaled / scaler
        
        # 3. Define the low and high thresholds based on this optimal split point
        # Heuristic: The low threshold is the split point itself. The high threshold
        # is set to distinguish the "clearly supine" state from the transitional state.
        low_threshold = otsu_threshold
        
        # Define the high threshold relative to the distribution of values above the split point.
        # A good heuristic is the 25th percentile of the "supine" cluster.
        supine_values = abs_dc_series[abs_dc_series > low_threshold]
        if len(supine_values) > 10: # Ensure there are enough samples
            high_threshold = np.percentile(supine_values, 25)
        else:
            # Fallback if there aren't many "supine" samples
            high_threshold = low_threshold + 0.1 

        # 4. Sanity checks to ensure thresholds are reasonable
        if low_threshold >= high_threshold or high_threshold > 2.0 or low_threshold < 0.01:
            raise ValueError("Calibrated thresholds are not plausible.")

        print(f"INFO: Auto-calibrated orientation thresholds: low={low_threshold:.3f}, high={high_threshold:.3f}")
        return low_threshold, high_threshold

    except Exception as e:
        print(f"WARNING: Orientation calibration failed ({e}). Falling back to default thresholds.")
        return fallback_low, fallback_high
    


import numpy as np
import pandas as pd
from scipy.signal import find_peaks, detrend, butter, filtfilt
import warnings

def adaptive_breath_detection(df, adaptation_window_minutes=5, 
                             sensitivity='medium', method='peaks'):
    # Input validation
    required_columns = ['breathingSignal', 'timestamp']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # Clean signal
    signal_series = df['breathingSignal'].copy().replace([np.inf, -np.inf], np.nan)
    valid_signal = signal_series.dropna()
    
    if len(valid_signal) < 200:
        raise ValueError(f"Insufficient valid samples: {len(valid_signal)}")
    
    signal = valid_signal.values
    valid_indices = valid_signal.index
    timestamps = df.loc[valid_indices, 'timestamp'].values
    
    # Get activity data if available
    activity_level = None
    if 'activityLevel' in df.columns:
        activity_level = df.loc[valid_indices, 'activityLevel'].values
    
    # Calculate sampling rate
    time_diffs = pd.Series(timestamps).diff().dropna()
    avg_sample_period = time_diffs.apply(lambda x: x.total_seconds()).median()
    if pd.isna(avg_sample_period) or avg_sample_period <= 0:
        avg_sample_period = 0.02  # 50 Hz fallback
    sampling_rate = 1 / avg_sample_period
    
    # Calculate adaptation window size
    adaptation_window_samples = int(adaptation_window_minutes * 60 * sampling_rate)
    adaptation_window_samples = min(adaptation_window_samples, len(signal) // 4)
    
    # Sensitivity settings
    sensitivity_params = {
        'low': {'base_height': 0.6, 'base_prominence': 0.5},
        'medium': {'base_height': 0.5, 'base_prominence': 0.4},
        'high': {'base_height': 0.4, 'base_prominence': 0.3}
    }
    params = sensitivity_params.get(sensitivity, sensitivity_params['medium'])
    
    # Process signal in overlapping windows
    low_orientation_thresh, high_orientation_thresh = _calibrate_orientation_thresholds(
        signal, sampling_rate, fallback_low=0.15, fallback_high=0.5
    )
    
    # ... (rest of the setup for the sliding window analysis)
    all_breath_events = []
    adaptation_history = []
    position_changes = []
    step_size = adaptation_window_samples // 4
    window_start = 0
    window_count = 0  # Signal change threshold for recalculation
    
    while window_start + adaptation_window_samples <= len(signal):
        window_end = window_start + adaptation_window_samples
        window_signal = signal[window_start:window_end]
        window_timestamps = timestamps[window_start:window_end]
        window_activity = activity_level[window_start:window_end] if activity_level is not None else None
        window_count += 1
        
        # Analyze window characteristics
        detrended_signal = detrend(window_signal, type='constant')
        signal_mean = np.mean(window_signal)
        signal_std = np.std(detrended_signal)
        signal_abs_mean = abs(signal_mean)
        
        # Determine orientation and preprocessing
        if signal_abs_mean > 0.5:
            gravity_influence = 'high'
            min_amplitude = 0.015
            base_height_factor = 0.25
            base_prominence_factor = 0.2
        elif signal_abs_mean > 0.15:
            gravity_influence = 'medium'
            min_amplitude = 0.008
            base_height_factor = 0.15
            base_prominence_factor = 0.12
        else:
            gravity_influence = 'low'
            min_amplitude = 0.005
            base_height_factor = 0.1
            base_prominence_factor = 0.08
        
        # Activity-based adjustment
        if window_activity is not None:
            avg_activity = np.mean(window_activity)
            if avg_activity > 0.1:
                activity_factor = 1.3  # Less sensitive during movement
            elif avg_activity > 0.03:
                activity_factor = 1.0  # Normal
            else:
                activity_factor = 0.8  # More sensitive during rest
        else:
            activity_factor = 1.0
        
        # Signal quality adjustment
        signal_diff = np.diff(detrended_signal)
        noise_estimate = np.std(signal_diff) * np.sqrt(2)
        snr = signal_std / (noise_estimate + 1e-10)
        
        if snr < 2:
            quality_factor = 1.4  # Less sensitive for noisy signals
        elif snr < 4:
            quality_factor = 1.0
        else:
            quality_factor = 0.8  # More sensitive for clean signals
        
        # Sleep factor (assuming sleep data - slightly more sensitive)
        sleep_factor = 0.9
        
        # Combine factors
        combined_factor = (sleep_factor * activity_factor * quality_factor * 
                          params['base_height'] * base_height_factor)
        
        # Calculate adaptive thresholds
        height_threshold = max(signal_std * combined_factor, min_amplitude * 0.3)
        prominence_threshold = max(signal_std * combined_factor * 0.8, min_amplitude * 0.15)
        
        # Adaptive minimum distance
        base_distance_sec = 1.5
        if gravity_influence == 'high':
            base_distance_sec = 1.8
        elif gravity_influence == 'low':
            base_distance_sec = 1.2
        min_distance = max(5, int(base_distance_sec * sampling_rate))
        
        # Apply preprocessing
        processed_signal = detrend(window_signal, type='constant')
        
        # High-pass filter for high gravity influence
        if gravity_influence == 'high':
            try:
                cutoff = 0.05
                nyquist = sampling_rate / 2
                normalized_cutoff = cutoff / nyquist
                if normalized_cutoff < 1.0:
                    b, a = butter(2, normalized_cutoff, btype='high')
                    processed_signal = filtfilt(b, a, processed_signal)
            except:
                pass  # Continue without filtering if it fails
        
        # Detect peaks and troughs
        try:
            peaks, _ = find_peaks(
                processed_signal,
                height=height_threshold,
                distance=min_distance,
                prominence=prominence_threshold,
                width=2
            )
            
            troughs, _ = find_peaks(
                -processed_signal,
                height=height_threshold,
                distance=min_distance,
                prominence=prominence_threshold,
                width=2
            )
            
            # Validate detected events
            validation_threshold = height_threshold * 1.5
            
            valid_peaks = [p for p in peaks if processed_signal[p] > validation_threshold]
            valid_troughs = [t for t in troughs if abs(processed_signal[t]) > validation_threshold]
            
            # Add breath events (adjust indices to global signal)
            for peak_idx in valid_peaks:
                global_idx = window_start + peak_idx
                if global_idx < len(timestamps):
                    all_breath_events.append({
                        'type': 'Inhalation',
                        'index': valid_indices[global_idx],
                        'timestamp': timestamps[global_idx],
                        'amplitude': processed_signal[peak_idx],
                        'raw_amplitude': signal[global_idx],
                        'event_type': 'peak'
                    })
            
            for trough_idx in valid_troughs:
                global_idx = window_start + trough_idx
                if global_idx < len(timestamps):
                    all_breath_events.append({
                        'type': 'Exhalation',
                        'index': valid_indices[global_idx],
                        'timestamp': timestamps[global_idx],
                        'amplitude': processed_signal[trough_idx],
                        'raw_amplitude': signal[global_idx],
                        'event_type': 'trough'
                    })
        
        except:
            # Continue to next window if detection fails
            pass
        
        window_start += step_size
    
    # Remove duplicate events from overlapping windows
    if not all_breath_events:
        return pd.DataFrame(), {
            'total_events': 0, 'breathing_cycles': 0, 'events_per_minute': 0,
            'breaths_per_minute': 0, 'duration_minutes': 0, 'error': 'No events detected'
        }
    
    # Sort and filter events
    all_breath_events.sort(key=lambda x: x['timestamp'])
    
    filtered_events = []
    last_timestamp = None
    min_event_spacing = pd.Timedelta(seconds=0.5)
    
    for event in all_breath_events:
        if last_timestamp is None or (event['timestamp'] - last_timestamp) > min_event_spacing:
            filtered_events.append(event)
            last_timestamp = event['timestamp']
    
    # Create output DataFrame
    breath_df = pd.DataFrame(filtered_events)
    if not breath_df.empty:
        breath_df = breath_df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate statistics
    total_events = len(filtered_events)
    inhalations = len([e for e in filtered_events if e['type'] == 'Inhalation'])
    exhalations = len([e for e in filtered_events if e['type'] == 'Exhalation'])
    breathing_cycles = min(inhalations, exhalations)
    
    duration_minutes = (timestamps[-1] - timestamps[0]) / pd.Timedelta(minutes=1)
    if duration_minutes <= 0:
        duration_minutes = len(timestamps) * avg_sample_period / 60
    
    events_per_minute = total_events / duration_minutes if duration_minutes > 0 else 0
    breaths_per_minute = breathing_cycles / duration_minutes if duration_minutes > 0 else 0
    
    stats = {
        'total_events': total_events,
        'breathing_cycles': breathing_cycles,
        'events_per_minute': events_per_minute,
        'breaths_per_minute': breaths_per_minute,
        'duration_minutes': duration_minutes,
        'inhalations': inhalations,
        'exhalations': exhalations,
        'method': method,
        'sensitivity': sensitivity,
        'sampling_rate': sampling_rate,
        'adaptation_window_minutes': adaptation_window_minutes,
        'error': None
    }
    
    return breath_df, stats

# # Basic breath detection
breath_df, stats = adaptive_breath_detection(respeck_df, adaptation_window_minutes=2)
print(f"Breathing rate: {stats['breaths_per_minute']:.1f} breaths/min")
