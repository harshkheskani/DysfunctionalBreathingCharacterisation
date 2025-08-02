# create_features_on_cluster.py
import pandas as pd
import numpy as np
import glob
import os
from scipy import stats, signal as scipy_signal
import gc
import random
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
import pywt # Make sure pywt is installed on the cluster
import argparse
# --- 1. NEW IMPORTS & AUGMENTATION FUNCTIONS ---
# Add scipy.signal for more advanced filtering if needed
from scipy import signal as scipy_signal
import pandas as pd
import numpy as np
import glob
import os
import pywt
from scipy import stats, signal as scipy_signal
from scipy.stats import spearmanr, skew, kurtosis
from collections import Counter
import warnings
import xgboost as xgb

# XGBoost and ML imports
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from calculateContinuousBreathFeatures import *

# Visualization and interpretability
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import gc
import random



# Assume this file exists and contains the necessary function
# If it's part of another script, you might need to adjust the import
try:
    from calculateContinuousBreathFeatures import calculate_TS_breathFeatures
except ImportError:
    print("Warning: 'calculateContinuousBreathFeatures.py' not found. Creating a placeholder function.")
    def calculate_TS_breathFeatures(timestamps, signal):
        # This is a placeholder. You MUST have the real file for the script to work.
        print("    --> Using placeholder for calculate_TS_breathFeatures. Features will be missing!")
        return {}


# --- DATA LOADING & FEATURE EXTRACTION FUNCTIONS (No changes needed here) ---

def load_raw_windows_with_timestamps(events_folder, respeck_folder, event_group_map):
    """
    Loads all session data and returns windows of raw breathing signals,
    their corresponding timestamp windows, labels, and session IDs.
    """
    print("🔄 Loading and windowing raw signal data...")
    all_raw_signals, all_raw_timestamps, all_labels, all_groups = [], [], [], []
    
    SAMPLING_RATE_HZ = 12.5
    WINDOW_DURATION_SEC = 30
    WINDOW_SIZE = int(WINDOW_DURATION_SEC * SAMPLING_RATE_HZ)
    STEP_SIZE = int(WINDOW_SIZE * 0.5)

    event_files = glob.glob(os.path.join(events_folder, '*_event_export.csv'))
    
    for event_file_path in event_files:
        base_name = os.path.basename(event_file_path)
        session_id = base_name.split('_event_export.csv')[0]
        respeck_file_path = os.path.join(respeck_folder, f'{session_id}_respeck.csv')
        
        if not os.path.exists(respeck_file_path):
            print(f"  - Skipping {session_id}, missing respeck file.")
            continue
            
        print(f"  📂 Processing {session_id}...")
        try:
            df_events = pd.read_csv(event_file_path, decimal=',')
            df_respeck = pd.read_csv(respeck_file_path).dropna(subset=['breathingSignal'])
            
            df_events.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True)
            df_respeck.rename(columns={'alignedTimestamp': 'timestamp_unix'}, inplace=True)
            for df_ in [df_events, df_respeck]:
                df_['timestamp_unix'] = pd.to_numeric(df_['timestamp_unix'], errors='coerce')
                df_.dropna(subset=['timestamp_unix'], inplace=True)
                df_['timestamp_unix'] = df_['timestamp_unix'].astype('int64')

            df_respeck['Label'] = 0 # Default label is 0 (Normal)
            df_events['Duration_ms'] = (df_events['Duration'] * 1000).astype('int64')
            df_events['end_time_unix'] = df_events['timestamp_unix'] + df_events['Duration_ms']
            
            for label_id, event_names in event_group_map.items():
                if label_id == 0: continue # Skip the 'Normal' label
                df_filtered = df_events[df_events['Event'].isin(event_names)]
                for _, event in df_filtered.iterrows():
                    df_respeck.loc[df_respeck['timestamp_unix'].between(event['timestamp_unix'], event['end_time_unix']), 'Label'] = label_id

            signals = df_respeck['breathingSignal'].values
            timestamps = df_respeck['timestamp_unix'].values
            labels = df_respeck['Label'].values
            
            for i in range(0, len(df_respeck) - WINDOW_SIZE, STEP_SIZE):
                label_window = labels[i : i + WINDOW_SIZE]
                # Assign the label that is most frequent in the window
                label = stats.mode(label_window)[0]
                
                # Exclude windows that have no label (or only normal) if you wish
                # if label != 0: 
                all_raw_signals.append(signals[i : i + WINDOW_SIZE])
                all_raw_timestamps.append(timestamps[i : i + WINDOW_SIZE])
                all_labels.append(label)
                all_groups.append(session_id)

            del df_events, df_respeck, signals, timestamps, labels
            gc.collect()
            print(f"    ✅ Processed and cleaned up memory for {session_id}")
        except Exception as e:
            print(f"    ❌ Error processing {session_id}: {e}")
            continue

    print(f"\n✅ Raw signal windowing complete. Found {len(all_raw_signals)} windows.")
    print(f"Class distribution: {Counter(all_labels)}")
    return all_raw_signals, all_raw_timestamps, np.array(all_labels, dtype=np.int32), np.array(all_groups)


# --- 1. CORRECTED generate_RRV FUNCTION (NumPy Compatible) ---

def generate_RRV(breathing_signal, sampling_rate=12.5):
    """
    Generate Respiratory Rate Variability (RRV) from a NumPy breathing signal array.
    This version is compatible with raw NumPy array inputs.
    """
    try:
        # --- CHANGE 1: Remove NaN values using NumPy ---
        breathing_signal = breathing_signal[~np.isnan(breathing_signal)]
        
        # --- CHANGE 2: Check array size instead of using .empty ---
        if breathing_signal.size < 10:
            return 0.0

        # Find peaks in breathing signal (inspirations)
        # Using min() for window_length requires at least Python 3.8
        window_length = min(25, len(breathing_signal) // 2 * 2 -1) # Ensure it's odd and less than signal length
        if window_length < 5: # savgol_filter requires window_length > polyorder
             return 0.0
        
        signal_smooth = scipy_signal.savgol_filter(breathing_signal, window_length, 3)
        
        # Find peaks with minimum distance between them
        peaks, _ = scipy_signal.find_peaks(signal_smooth, 
                                         distance=int(sampling_rate * 1.5),  # Min 1.5 seconds between breaths
                                         height=np.percentile(signal_smooth, 60))
        
        if len(peaks) < 3:
            return 0.0
        
        # Calculate inter-breath intervals in seconds
        breath_intervals = np.diff(peaks) / sampling_rate
        
        # RRV is typically the standard deviation of breath intervals
        rrv = np.std(breath_intervals)
        
        return rrv if not np.isnan(rrv) else 0.0
        
    except Exception as e:
        print(f"Error in generate_RRV: {e}")
        return 0.0

def add_noise(signal_window, noise_level=0.02):
    """Adds Gaussian noise to the signal."""
    noise = np.random.normal(0, noise_level, len(signal_window))
    return signal_window + noise

def scale_amplitude(signal_window, scale_range=(0.9, 1.1)):
    """Randomly scales the amplitude of the signal."""
    scale_factor = random.uniform(scale_range[0], scale_range[1])
    return signal_window * scale_factor

def time_shift(signal_window, max_shift_ratio=0.05):
    """Shifts the signal in time by a small random amount."""
    max_shift = int(len(signal_window) * max_shift_ratio)
    shift_amount = random.randint(-max_shift, max_shift)
    return np.roll(signal_window, shift_amount)

# A simple wrapper to apply a random augmentation
def apply_random_augmentation(signal_window):
    """Applies one of the augmentation techniques at random."""
    augmentations = [add_noise, scale_amplitude, time_shift]
    chosen_augmentation = random.choice(augmentations)
    return chosen_augmentation(signal_window)

def extract_clinical_breath_features_simple_wavelet(timestamps, signal, window_size_sec=30):
    """
    Simplified version with just the most essential wavelet features
    """
    
    sampling_rate = 12.5
    features = {}
    
    # Basic statistical features
    features['signal_mean'] = np.mean(signal)
    features['signal_std'] = np.std(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))
    features['activity_level'] = np.mean(np.abs(np.diff(signal)))
    
    try:
        # Simple multi-level DWT
        wavelet = 'db4'
        max_level = min(pywt.dwt_max_level(len(signal), wavelet), 6)
        coeffs = pywt.wavedec(signal, wavelet, level=max_level)
        
        # Just extract energy at each level - let XGBoost figure out the rest
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_level_{i}_energy'] = np.sum(coeff**2)
            features[f'wavelet_level_{i}_std'] = np.std(coeff)
            features[f'wavelet_level_{i}_max'] = np.max(np.abs(coeff))
        
        # A few global measures
        all_energies = [np.sum(c**2) for c in coeffs]
        total_energy = sum(all_energies) + 1e-10
        
        features['wavelet_energy_entropy'] = -sum([(e/total_energy) * np.log(e/total_energy + 1e-10) for e in all_energies])
        features['wavelet_energy_concentration'] = max(all_energies) / total_energy
        
    except Exception as e:
        print(f"Simple wavelet analysis failed: {e}")
        # Set minimal defaults
        for i in range(6):
            features[f'wavelet_level_{i}_energy'] = 0
            features[f'wavelet_level_{i}_std'] = 0
            features[f'wavelet_level_{i}_max'] = 0
        features['wavelet_energy_entropy'] = 0
        features['wavelet_energy_concentration'] = 0

    breath_features = calculate_TS_breathFeatures(timestamps, signal)

    if breath_features and 'amplitude' in breath_features:
        amplitudes = np.array(breath_features['amplitude'])
        if len(amplitudes) > 0:
            features['amplitude_mean'] = np.mean(amplitudes)
            features['amplitude_std'] = np.std(amplitudes)
            features['amplitude_cv'] = features['amplitude_std'] / (features['amplitude_mean'] + 1e-10)
            
            # Key OSA features
            features['amplitude_p10'] = np.percentile(amplitudes, 10)
            features['amplitude_p50'] = np.percentile(amplitudes, 50)
            features['amplitude_p90'] = np.percentile(amplitudes, 90)
            features['amplitude_reduction_ratio'] = 1 - (features['amplitude_p10'] / (features['amplitude_p90'] + 1e-10))

    if breath_features and 'breath_durations' in breath_features:
        durations = np.array(breath_features['breath_durations'])
        if len(durations) > 0:
            features['breath_duration_mean'] = np.mean(durations)
            features['breath_duration_std'] = np.std(durations)
            features['breath_duration_cv'] = features['breath_duration_std'] / (features['breath_duration_mean'] + 1e-10)
            features['long_breath_ratio'] = np.sum(durations > 20) / len(durations)

    if breath_features and 'rr' in breath_features:
        rr = np.array(breath_features['rr'])
        rr = rr[~np.isnan(rr)]
        if len(rr) > 0:
            features['respiratory_rate_mean'] = np.mean(rr)
            features['respiratory_rate_std'] = np.std(rr)
            features['respiratory_rate_cv'] = features['respiratory_rate_std'] / (features['respiratory_rate_mean'] + 1e-10)

    features['RRV'] = generate_RRV(signal, sampling_rate)
    features['RRV3ANN'] = features['RRV'] * 0.65

    # Additional clinical features
    signal_envelope = np.abs(scipy_signal.hilbert(signal - np.mean(signal)))
    features['envelope_std'] = np.std(signal_envelope)
    features['envelope_cv'] = features['envelope_std'] / (np.mean(signal_envelope) + 1e-10)

    # Short/long term variability
    if len(signal) >= 20:
        short_segments = signal[::5]
        features['short_term_variability'] = np.std(short_segments)
        
        if len(signal) >= 60:
            long_segments = signal[::12]
            features['long_term_trend'] = np.abs(np.polyfit(range(len(long_segments)), long_segments, 1)[0])
        else:
            features['long_term_trend'] = 0
    else:
        features['short_term_variability'] = 0
        features['long_term_trend'] = 0

    return features


def process_windows_in_chunks(timestamps_list, signals_list, chunk_size=1000):
    """Generator function to process features in smaller chunks to save memory."""
    all_features = []
    for i in range(0, len(signals_list), chunk_size):
        print(f"    Processing chunk {i//chunk_size + 1}...")
        chunk_ts = timestamps_list[i:i+chunk_size]
        chunk_sig = signals_list[i:i+chunk_size]
        
        # Use your new optimized feature extractor here
        chunk_features = [extract_clinical_breath_features_simple_wavelet(ts, sig) for ts, sig in zip(chunk_ts, chunk_sig)]
        all_features.extend(chunk_features)
        
        # Explicit garbage collection after each chunk
        del chunk_ts, chunk_sig, chunk_features
        gc.collect()
        
    return pd.DataFrame(all_features).fillna(0)



def create_features_per_session(events_folder, respeck_folder, output_dir):
    """
    Loops through each session, extracts features, and saves a single .parquet file for each one.
    This is the most memory-efficient approach.
    """
    print("🚀 STARTING: Per-Session Feature Engineering & Caching")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Data Loading & Windowing Constants ---
    SAMPLING_RATE_HZ = 12.5
    WINDOW_DURATION_SEC = 30
    WINDOW_SIZE = int(WINDOW_DURATION_SEC * SAMPLING_RATE_HZ)
    STEP_SIZE = int(WINDOW_SIZE * 0.5)

    all_labels_for_encoder = []
    event_files = glob.glob(os.path.join(events_folder, '*_event_export.csv'))

    # --- First pass to determine all possible labels for the encoder ---
    # This ensures the LabelEncoder is consistent across all data
    # print("🔄 First pass: Scanning all labels to build a consistent encoder...")
    # for event_file_path in event_files:
    #     try:
    #         df_events = pd.read_csv(event_file_path, decimal=',')
    #         for label_id, event_names in event_group_map.items():
    #             if label_id != 0 and df_events['Event'].isin(event_names).any():
    #                 all_labels_for_encoder.append(label_id)
    #     except Exception as e:
    #         print(f"  - Warning: could not read labels from {event_file_path}: {e}")
    #         continue
    # all_labels_for_encoder.append(0) # Ensure 'Normal' is always included
    
    le = LabelEncoder()
    # le.fit(list(set(all_labels_for_encoder)))
    # print(f"Label encoder fitted with classes: {le.classes_}")
    # np.save(os.path.join(output_dir, 'label_encoder_classes.npy'), le.classes_)
    # print(f"Saved label encoder mapping to {output_dir}")
    
    # --- Main processing loop: one session at a time ---
    for event_file_path in event_files:
        base_name = os.path.basename(event_file_path)
        session_id = base_name.split('_event_export.csv')[0]
        respeck_file_path = os.path.join(respeck_folder, f'{session_id}_respeck.csv')
        
        output_parquet_path = os.path.join(output_dir, f'{session_id}_features.parquet')
        if os.path.exists(output_parquet_path):
            print(f"  - Skipping {session_id}, feature file already exists.")
            continue

        if not os.path.exists(respeck_file_path):
            print(f"  - Skipping {session_id}, missing respeck file.")
            continue
            
        print(f"\n  📂 Processing {session_id}...")
        
        try:
            # --- Load, label, and window data for THIS session only ---
            df_events = pd.read_csv(event_file_path, decimal=',')
            df_respeck = pd.read_csv(respeck_file_path).dropna(subset=['breathingSignal'])
            
            # (Timestamp alignment and labeling logic - same as before)
            df_events.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True)
            df_respeck.rename(columns={'alignedTimestamp': 'timestamp_unix'}, inplace=True)
            for df_ in [df_events, df_respeck]:
                df_['timestamp_unix'] = pd.to_numeric(df_['timestamp_unix'], errors='coerce')
                df_.dropna(subset=['timestamp_unix'], inplace=True)
                df_['timestamp_unix'] = df_['timestamp_unix'].astype('int64')

            # df_respeck['label_raw'] = 0 # Use a temporary column name
            # df_events['Duration_ms'] = (df_events['Duration'] * 1000).astype('int64')
            # df_events['end_time_unix'] = df_events['timestamp_unix'] + df_events['Duration_ms']
            
            # for label_id, event_names in event_group_map.items():
            #     if label_id == 0: continue
            #     df_filtered = df_events[df_events['Event'].isin(event_names)]
            #     for _, event in df_filtered.iterrows():
            #         df_respeck.loc[df_respeck['timestamp_unix'].between(event['timestamp_unix'], event['end_time_unix']), 'label_raw'] = label_id
            df_respeck['label_raw'] = 0
                
            # The ONLY event that gets label 1 is 'Obstructive Apnea'
            target_event_name = ['Obstructive Apnea']
            df_filtered = df_events[df_events['Event'].isin(target_event_name)]
            
            for _, event in df_filtered.iterrows():
                df_respeck.loc[
                    df_respeck['timestamp_unix'].between(event['timestamp_unix'], event['end_time_unix']), 
                    'label_raw'
                ] = 1 
            # --- Windowing and Feature Extraction ---
            session_windows_sig = []
            session_windows_ts = []
            session_labels = []
            
            signals = df_respeck['breathingSignal'].values
            timestamps = df_respeck['timestamp_unix'].values
            labels = df_respeck['label_raw'].values

            for i in range(0, len(df_respeck) - WINDOW_SIZE, STEP_SIZE):
                label_window = labels[i : i + WINDOW_SIZE]
                label = stats.mode(label_window)[0]
                
                session_windows_sig.append(signals[i : i + WINDOW_SIZE])
                session_windows_ts.append(timestamps[i : i + WINDOW_SIZE])
                session_labels.append(label)

            if not session_labels:
                print(f"    - No windows generated for {session_id}. Skipping.")
                continue

            # Process features for this session's windows
            features_df = process_windows_in_chunks(session_windows_ts, session_windows_sig)
            features_df['label'] = le.transform(session_labels) 
            # Add labels and session_id, then save
            # features_df['label'] = le.transform(session_labels) # Use the fitted encoder
            features_df['session_id'] = session_id
            
            features_df.to_parquet(output_parquet_path, index=False)
            print(f"    ✅ Saved features for {session_id} to {output_parquet_path}")

            # Crucial memory cleanup
            del df_events, df_respeck, signals, timestamps, labels, features_df
            gc.collect()

        except Exception as e:
            print(f"    ❌ Error processing {session_id}: {e}")
            continue

    print("\n✅ CLUSTER JOB COMPLETE: All session features have been generated.")

# --- SCRIPT ENTRY POINT ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract features for BINARY Obstructive Apnea detection.")
    parser.add_argument('--events_folder', type=str, required=True, help='Path to event_export CSVs.')
    parser.add_argument('--respeck_folder', type=str, required=True, help='Path to respeck CSVs.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save feature files.')
    args = parser.parse_args()

    # The main function will now handle the binary logic directly
    create_features_per_session(
        events_folder=args.events_folder,
        respeck_folder=args.respeck_folder,
        output_dir=args.output_dir
    )