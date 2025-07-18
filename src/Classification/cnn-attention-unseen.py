# ==============================================================================
# 0. Preamble & Imports
# ==============================================================================
# This script trains and evaluates an attention-based multi-scale CNN for sleep
# apnea detection. It uses a fixed training set (patient with 9 nights) plus
# one additional night for training, and tests on the remaining night.
#
# To run, ensure all dependencies are installed and execute from the command line:
# python attention_cnn_cluster.py --data_dir /path/to/data --base_output_dir /path/to/output
#
# NOTE: The script will generate and display plots. For use on a cluster without
# a display, you can save the plots by uncommenting the `plt.savefig()` lines
# and commenting out `plt.show()`.
#
# Dependencies:
# pandas, numpy, scikit-learn, imbalanced-learn, torch, seaborn, matplotlib

import pandas as pd
import numpy as np
import glob
import os
from scipy import stats
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
import math
import argparse

# ==============================================================================
# 2. Model & Helper Class Definitions
# ==============================================================================

### NEW: Improved CNN with Attention and Multi-Scale Convolutions ###


class ChannelAttention(nn.Module):
    """Channel attention mechanism to focus on important features"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _ = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pooling  
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1) * x

class SpatialAttention(nn.Module):
    """Spatial attention to focus on important time segments"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat)) * x

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class MultiScaleConv(nn.Module):
    """Multi-scale convolution to capture patterns at different temporal scales"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Different kernel sizes to capture different temporal patterns
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )
        
        # 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )
        
        # Combine features
        self.combine = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate all branches
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.combine(out)
        
        return out

class ImprovedCNN(nn.Module):
    """
    Improved CNN for sleep apnea detection with:
    - Multi-scale convolutions for different temporal patterns
    - Residual connections for better gradient flow
    - Channel and spatial attention mechanisms
    - Proper regularization
    """
    def __init__(self, n_features, n_outputs, n_timesteps):
        super().__init__()
        
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_timesteps = n_timesteps
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-scale feature extraction
        self.multiscale1 = MultiScaleConv(64, 128)
        self.pool1 = nn.MaxPool1d(2)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(128, 128, kernel_size=5, dropout=0.3)
        self.res_block2 = ResidualBlock(128, 256, kernel_size=5, stride=2, dropout=0.3)
        
        # Second multi-scale layer
        self.multiscale2 = MultiScaleConv(256, 256)
        self.pool2 = nn.MaxPool1d(2)
        
        # More residual blocks
        self.res_block3 = ResidualBlock(256, 256, kernel_size=3, dropout=0.4)
        self.res_block4 = ResidualBlock(256, 512, kernel_size=3, stride=2, dropout=0.4)
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(512)
        self.spatial_attention = SpatialAttention()
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Calculate feature size after convolutions
        self.feature_size = 512
        
        # Classifier with proper regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_outputs)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: (batch, timesteps, features)
        # Convert to: (batch, features, timesteps) for Conv1d
        x = x.permute(0, 2, 1)
        
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # Multi-scale feature extraction
        x = self.multiscale1(x)
        x = self.pool1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Second multi-scale layer
        x = self.multiscale2(x)
        x = self.pool2(x)
        
        # More residual blocks
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Apply attention mechanisms
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def add_signal_features(df):
    """Adds rolling window features to the dataframe."""
    print("Engineering new signal-based features...")
    ROLLING_WINDOW_SIZE = 25
    df['breathing_signal_rolling_mean'] = df.groupby('SessionID')['breathingSignal'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
    )
    df['breathing_signal_rolling_std'] = df.groupby('SessionID')['breathingSignal'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).std()
    )
    df['accel_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    print(f"New features added: {['breathing_signal_rolling_mean', 'breathing_signal_rolling_std', 'accel_magnitude']}\n")
    return df

def load_session_data(session_id, events_folder, respeck_folder, nasal_folder, event_group_to_label):
    """Load and process data for a single session."""
    event_file_path = os.path.join(events_folder, f'{session_id}_event_export.csv')
    respeck_file_path = os.path.join(respeck_folder, f'{session_id}_respeck.csv')
    nasal_file_path = os.path.join(nasal_folder, f'{session_id}_nasal.csv')
    
    if not all(os.path.exists(p) for p in [event_file_path, respeck_file_path, nasal_file_path]):
        print(f"  - WARNING: Skipping session '{session_id}'. A corresponding file is missing.")
        return None
    
    print(f"  - Processing session: {session_id}")
    
    df_events = pd.read_csv(event_file_path, decimal=',')
    df_nasal = pd.read_csv(nasal_file_path)
    df_respeck = pd.read_csv(respeck_file_path)

    df_events.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True)
    df_nasal.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True, errors='ignore')
    df_respeck.rename(columns={'alignedTimestamp': 'timestamp_unix'}, inplace=True)

    for df_ in [df_events, df_nasal, df_respeck]:
        df_['timestamp_unix'] = pd.to_numeric(df_['timestamp_unix'], errors='coerce')
        df_.dropna(subset=['timestamp_unix'], inplace=True)
        df_['timestamp_unix'] = df_['timestamp_unix'].astype('int64')

    start_time = max(df_nasal['timestamp_unix'].min(), df_respeck['timestamp_unix'].min())
    end_time = min(df_nasal['timestamp_unix'].max(), df_respeck['timestamp_unix'].max())
    
    df_respeck = df_respeck[(df_respeck['timestamp_unix'] >= start_time) & (df_respeck['timestamp_unix'] <= end_time)].copy()

    if df_respeck.empty:
        print(f"  - WARNING: Skipping session '{session_id}'. No Respeck data in the overlapping range.")
        return None

    print(f"  - Applying precise interval-based labels...")
    df_respeck['Label'] = 0
    df_events['Duration_ms'] = (df_events['Duration'] * 1000).astype('int64')
    df_events['end_time_unix'] = df_events['timestamp_unix'] + df_events['Duration_ms']
    
    for label_id, event_names_in_group in event_group_to_label.items():
        df_filtered_events = df_events[df_events['Event'].isin(event_names_in_group)]
        for _, event in df_filtered_events.iterrows():
            start_event = event['timestamp_unix']
            end_event = event['end_time_unix']
            df_respeck.loc[df_respeck['timestamp_unix'].between(start_event, end_event), 'Label'] = label_id

    df_respeck['SessionID'] = session_id
    return df_respeck

def balance_data(X_data, y_data, random_state):
    """Resamples a dataset to balance its classes using SMOTE or RandomOverSampler."""
    if len(X_data) == 0:
        return X_data, y_data

    print(f"  - Balancing data with initial shape {X_data.shape} and distribution {Counter(y_data)}")
    
    # Reshape for sampler
    nsamples, n_timesteps, n_features = X_data.shape
    X_reshaped = X_data.reshape((nsamples, n_timesteps * n_features))
    
    class_counts = Counter(y_data)
    # Filter out class 0 and any classes not present
    minority_classes = {label: count for label, count in class_counts.items() if label != 0 and count > 0}
    
    if not minority_classes:
        print("  - No minority classes to balance. Returning original data.")
        return X_data, y_data
        
    min_class_count = min(minority_classes.values())
    k = min_class_count - 1

    if k < 1:
        print(f"  - Smallest minority class has {min_class_count} samples. Using RandomOverSampler.")
        sampler = RandomOverSampler(random_state=random_state)
    else:
        sampler = SMOTE(random_state=random_state, k_neighbors=k)
    
    X_resampled, y_resampled = sampler.fit_resample(X_reshaped, y_data)
    print(f"  - Resampled distribution: {Counter(y_resampled)}")
    
    # Reshape back to original format
    X_resampled = X_resampled.reshape(-1, n_timesteps, n_features)
    
    return X_resampled, y_resampled

# ==============================================================================
# 3. Main Execution Block
# ==============================================================================
def main():
    """Main function to run the data processing, training, and evaluation pipeline."""

    parser = argparse.ArgumentParser(description="Train an attention-based CNN for sleep apnea detection.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing /03/ folder with patient data.')
    parser.add_argument('--base_output_dir', type=str, required=True,
                        help='Base directory to save results (e.g., plots, checkpoints).')
    args = parser.parse_args()

    os.makedirs(args.base_output_dir, exist_ok=True)
    
    # --- Configuration & Constants ---
    print("--- 1. Setting up configuration and constants ---")
    
    # Define the three patient directories
    MAIN_PATIENT_DIR = os.path.join(args.data_dir)   # Patient with 9 nights
    ADDITIONAL_PATIENT_1_DIR = os.path.join(args.data_dir, 'CSR005')  # Additional patient 1
    ADDITIONAL_PATIENT_2_DIR = os.path.join(args.data_dir, 'CSR003')  # Additional patient 2

    EVENT_GROUP_TO_LABEL = {
        1: ['Obstructive Apnea'],
        2: ['Hypopnea', 'Central Hypopnea', 'Obstructive Hypopnea'],
        3: ['Central Apnea', 'Mixed Apnea'],
        4: ['Desaturation']
    }
    LABEL_TO_EVENT_GROUP_NAME = {
        0: 'Normal',
        1: 'Obstructive Apnea',
        2: 'Hypopnea Events',
        3: 'Central/Mixed Apnea',
        4: 'Desaturation'
    }
    N_OUTPUTS = len(EVENT_GROUP_TO_LABEL) + 1
    CLASS_NAMES = [LABEL_TO_EVENT_GROUP_NAME[i] for i in range(N_OUTPUTS)]

    SAMPLING_RATE_HZ = 12.5
    WINDOW_DURATION_SEC = 30
    WINDOW_SIZE = int(WINDOW_DURATION_SEC * SAMPLING_RATE_HZ)
    OVERLAP_PERCENTAGE = 0.80
    STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENTAGE))
    
    RANDOM_STATE = 42
    EPOCHS = 100
    BATCH_SIZE = 64
    FEATURE_COLUMNS = ['breathingSignal', 'activityLevel', 'breathingRate', 'x', 'y', 'z']

    # --- Data Loading ---
    print("\n--- 2. Loading and preprocessing data ---")
    
    # Load main patient data (9 nights)
    print("Loading main patient data (9 nights)...")
    main_events_folder = os.path.join(MAIN_PATIENT_DIR, 'event_exports')
    main_respeck_folder = os.path.join(MAIN_PATIENT_DIR, 'respeck')
    main_nasal_folder = os.path.join(MAIN_PATIENT_DIR, 'nasal_files')
    
    main_event_files = glob.glob(os.path.join(main_events_folder, '*_event_export.csv'))
    main_sessions_df_list = []
    
    for event_file_path in main_event_files:
        base_name = os.path.basename(event_file_path)
        session_id = base_name.split('_event_export.csv')[0]
        session_data = load_session_data(session_id, main_events_folder, main_respeck_folder, 
                                       main_nasal_folder, EVENT_GROUP_TO_LABEL)
        if session_data is not None:
            main_sessions_df_list.append(session_data)
    
    print(f"Loaded {len(main_sessions_df_list)} sessions from main patient.")
    
    # Load additional patients data
    print("\nLoading additional patient data...")
    additional_sessions_data = {}
    
    for patient_name, patient_dir in [('csr005', ADDITIONAL_PATIENT_1_DIR), ('csr003', ADDITIONAL_PATIENT_2_DIR)]:
        events_folder = os.path.join(patient_dir, 'event_exports')
        respeck_folder = os.path.join(patient_dir, 'respeck')
        nasal_folder = os.path.join(patient_dir, 'nasal_files')
        
        if not os.path.exists(events_folder):
            print(f"  - WARNING: Events folder not found for {patient_name}: {events_folder}")
            continue
            
        event_files = glob.glob(os.path.join(events_folder, '*_event_export.csv'))
        patient_sessions = []
        
        for event_file_path in event_files:
            base_name = os.path.basename(event_file_path)
            session_id = base_name.split('_event_export.csv')[0]
            session_data = load_session_data(session_id, events_folder, respeck_folder, 
                                           nasal_folder, EVENT_GROUP_TO_LABEL)
            if session_data is not None:
                # Add patient identifier to session ID to avoid conflicts
                session_data['SessionID'] = f"{patient_name}_{session_id}"
                patient_sessions.append(session_data)
        
        additional_sessions_data[patient_name] = patient_sessions
        print(f"  - Loaded {len(patient_sessions)} sessions from {patient_name}")

    if not main_sessions_df_list:
        raise ValueError("No main patient data was loaded.")
    
    # Combine main patient data
    main_df = pd.concat(main_sessions_df_list, ignore_index=True)
    
    print(f"\nMain patient data shape: {main_df.shape}")
    print(f"Main patient class distribution: \n{main_df['Label'].value_counts(normalize=True)}")

    # --- Feature Engineering & Imputation ---
    print("\n--- 3. Engineering features, imputing missing values, and normalizing ---")
    
    def process_dataframe(df):
        """Process a dataframe with feature engineering and imputation."""
        df = add_signal_features(df)
        
        # Fill std NaNs that can occur at the start of a group
        df['breathing_signal_rolling_std'].bfill(inplace=True)

        print("Checking for and imputing missing values (NaNs)...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                print(f"  - Found {df[col].isnull().sum()} NaNs in '{col}'. Applying forward-fill and backward-fill.")
                df[col].ffill(inplace=True)
                df[col].bfill(inplace=True)
        
        final_nan_count = df[FEATURE_COLUMNS].isnull().sum().sum()
        if final_nan_count > 0:
            print(f"\nWARNING: {final_nan_count} NaNs still remain in feature columns after imputation.")
        else:
            print("\nImputation complete. No NaNs remain in feature columns.")
        
        return df

    # Process main patient data
    main_df = process_dataframe(main_df)
    
    # Process additional patients data
    for patient_name in additional_sessions_data:
        for i, session_df in enumerate(additional_sessions_data[patient_name]):
            additional_sessions_data[patient_name][i] = process_dataframe(session_df)

    # --- Per-Session Normalization ---
    print("\nApplying per-session (per-subject) normalization...")
    
    def normalize_dataframe(df):
        """Apply per-session normalization to a dataframe."""
        df_normalized = df.copy()
        for session_id in df['SessionID'].unique():
            session_mask = df['SessionID'] == session_id
            session_features = df.loc[session_mask, FEATURE_COLUMNS]
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(session_features)
            df_normalized.loc[session_mask, FEATURE_COLUMNS] = scaled_features
        return df_normalized
    
    main_df_normalized = normalize_dataframe(main_df)
    
    # Normalize additional patients data
    for patient_name in additional_sessions_data:
        for i, session_df in enumerate(additional_sessions_data[patient_name]):
            additional_sessions_data[patient_name][i] = normalize_dataframe(session_df)

    print("Normalization complete.")

    # --- Windowing Function ---
    def create_windows(df, window_size, step_size):
        """Create time-series windows from dataframe."""
        X, y, groups = [], [], []
        for session_id, session_df in df.groupby('SessionID'):
            for i in range(0, len(session_df) - window_size, step_size):
                window_df = session_df.iloc[i : i + window_size]
                features = window_df[FEATURE_COLUMNS].values
                label = stats.mode(window_df['Label'])[0]
                X.append(features)
                y.append(label)
                groups.append(session_id)
        return np.asarray(X), np.asarray(y), np.asarray(groups)

    # --- Device Setup ---
    print("\n--- 4. Setting up PyTorch device ---")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # --- Cross-Validation with Fixed Training Set ---
    print("\n--- 5. Starting Cross-Validation with Fixed Training Set ---")
    print("Training Strategy:")
    print("- Fixed training set: Main patient (9 nights)")
    print("- Additional training: One additional patient")
    print("- Test set: The other additional patient")
    print()

    # Create windows for main patient data
    print("Creating windows for main patient data...")
    main_X, main_y, main_groups = create_windows(main_df_normalized, WINDOW_SIZE, STEP_SIZE)
    print(f"Main patient windows - X: {main_X.shape}, y: {main_y.shape}")
    print(f"Main patient class distribution: {Counter(main_y)}")

    # Create windows for additional patients
    additional_patients_windows = {}
    for patient_name, sessions_list in additional_sessions_data.items():
        if sessions_list:  # Only process if sessions exist
            combined_df = pd.concat(sessions_list, ignore_index=True)
            X, y, groups = create_windows(combined_df, WINDOW_SIZE, STEP_SIZE)
            additional_patients_windows[patient_name] = (X, y, groups)
            print(f"{patient_name} windows - X: {X.shape}, y: {y.shape}")
            print(f"{patient_name} class distribution: {Counter(y)}")

    # Get patient names for cross-validation
    patient_names = list(additional_patients_windows.keys())
    if len(patient_names) != 2:
        raise ValueError(f"Expected 2 additional patients, got {len(patient_names)}: {patient_names}")

    all_fold_predictions = []
    all_fold_true_labels = []
    
    # Perform 2-fold cross-validation
    for fold in range(2):
        train_patient = patient_names[fold]
        test_patient = patient_names[1 - fold]
        
        print(f"\n--- FOLD {fold + 1}/2 ---")
        print(f"Training on: Main patient + {train_patient}")
        print(f"Testing on: {test_patient}")
        
        # Combine main patient data with additional training patient
        train_additional_X, train_additional_y, train_additional_groups = additional_patients_windows[train_patient]
        X_train = np.vstack([main_X, train_additional_X])
        y_train = np.hstack([main_y, train_additional_y])
        
        # Use test patient for testing
        X_test, y_test, test_groups = additional_patients_windows[test_patient]
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Training class distribution: {Counter(y_train)}")
        print(f"Test class distribution: {Counter(y_test)}")
        
        # Apply SMOTE to training data
        nsamples, n_timesteps, n_features = X_train.shape
        # X_train_reshaped = X_train.reshape((nsamples, n_timesteps * n_features))
        
        # class_counts = Counter(y_train)
        # minority_classes = {label: count for label, count in class_counts.items() if label != 0}
        
        # if not minority_classes:
        #     print("  - No minority classes in training data. Skipping resampling.")
        #     X_train_resampled, y_train_resampled = X_train_reshaped, y_train
        # else:
        #     min_class_count = min(minority_classes.values())
        #     k = min_class_count - 1
        #     if k < 1:
        #         print(f"  - Smallest minority class has {min_class_count} samples. Using RandomOverSampler.")
        #         sampler = RandomOverSampler(random_state=RANDOM_STATE)
        #     else:
        #         print(f"  - Using SMOTE with k_neighbors={k}")
        #         sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            
        #     X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_reshaped, y_train)
        #     print(f"  - Resampled training distribution: {Counter(y_train_resampled)}")

        # --- THIS IS THE NEW RESAMPLING LOGIC ---
        print("\nApplying new sophisticated resampling strategy...")

        # 1. Balance the main patient's data separately
        main_X_resampled, main_y_resampled = balance_data(main_X, main_y, RANDOM_STATE)

        # 2. Balance the additional training patient's data separately
        train_additional_X, train_additional_y, _ = additional_patients_windows[train_patient]
        add_X_resampled, add_y_resampled = balance_data(train_additional_X, train_additional_y, RANDOM_STATE)

        # 3. Combine the TWO BALANCED datasets
        print("\nCombining the two balanced datasets...")
        X_train_resampled = np.vstack([main_X_resampled, add_X_resampled])
        y_train_resampled = np.hstack([main_y_resampled, add_y_resampled])

        print(f"Final combined and balanced training set shape: {X_train_resampled.shape}")
        print(f"Final training class distribution: {Counter(y_train_resampled)}")
        # --- END OF NEW RESAMPLING LOGIC ---
        
        # Reshape back to original format
        # X_train_resampled = X_train_resampled.reshape(-1, n_timesteps, n_features)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train_resampled).float()
        y_train_tensor = torch.from_numpy(y_train_resampled).long()
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).long()
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = ImprovedCNN(n_features=n_features, n_outputs=N_OUTPUTS, n_timesteps=n_timesteps).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
        criterion = nn.CrossEntropyLoss()
        checkpoint_path = os.path.join(args.base_output_dir, f'fixed_training_checkpoint_fold_{fold}.pt')
        early_stopping = EarlyStopping(patience=8, verbose=False, path=checkpoint_path)
        
        max_grad_norm = 1.0
        
        # Training loop
        print(f"  - Starting training for fold {fold + 1}...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                running_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(test_loader)
            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss, model)
            
            if early_stopping.early_stop:
                print(f"  - Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        fold_preds = []
        fold_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                fold_preds.extend(predicted.cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())
        
        # Store results
        all_fold_predictions.extend(fold_preds)
        all_fold_true_labels.extend(fold_labels)
        
        print(f"  - Fold {fold + 1} complete. Test accuracy: {100 * np.mean(np.array(fold_preds) == np.array(fold_labels)):.2f}%")
        
        # Print fold-specific results
        print(f"\nFold {fold + 1} Results:")
        print(f"Training on: Main patient + {train_patient}")
        print(f"Testing on: {test_patient}")
        print(classification_report(fold_labels, fold_preds, labels=range(N_OUTPUTS), 
                                  target_names=CLASS_NAMES, zero_division=0))
        
        # Create fold-specific confusion matrix
        cm = confusion_matrix(fold_labels, fold_preds, labels=range(N_OUTPUTS))
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'Fold {fold + 1} - Normalized Confusion Matrix\n'
                 f'Train: Main + {train_patient}, Test: {test_patient}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_path = os.path.join(args.base_output_dir, f"confusion_matrix_fold_{fold + 1}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    # --- Final Aggregated Evaluation ---
    print("\n====================================================")
    print("Fixed Training Set Cross-Validation Complete")
    print("====================================================")
    print("Training Strategy Summary:")
    print("- Fixed training set: Main patient (9 nights)")
    print("- Fold 1: Train on Main + csr005, Test on csr003")
    print("- Fold 2: Train on Main + csr003, Test on csr005")
    print("====================================================")
    
    overall_accuracy = 100 * np.mean(np.array(all_fold_predictions) == np.array(all_fold_true_labels))
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print()
    
    print("Aggregated Results Across Both Folds:")
    print(classification_report(all_fold_true_labels, all_fold_predictions, 
                              labels=range(N_OUTPUTS), target_names=CLASS_NAMES, zero_division=0))
    
    # Create aggregated confusion matrix
    cm = confusion_matrix(all_fold_true_labels, all_fold_predictions, labels=range(N_OUTPUTS))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Aggregated Normalized Confusion Matrix\n'
             'Fixed Training Set Cross-Validation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    aggregated_plot_path = os.path.join(args.base_output_dir, "confusion_matrix_aggregated_fixed_training.png")
    plt.savefig(aggregated_plot_path, bbox_inches='tight')
    plt.close()
    
    # Additional analysis
    print("\n--- Additional Analysis ---")
    print(f"Total test samples: {len(all_fold_true_labels)}")
    print(f"Class distribution in test data: {Counter(all_fold_true_labels)}")
    print(f"Class distribution in predictions: {Counter(all_fold_predictions)}")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = np.array(all_fold_true_labels) == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(np.array(all_fold_predictions)[class_mask] == i)
            print(f"  {class_name}: {class_accuracy:.2%} ({np.sum(class_mask)} samples)")
        else:
            print(f"  {class_name}: No samples in test set")
    
    print("\n====================================================")
    print("Analysis Complete!")
    print("====================================================")


# ==============================================================================
# 4. Script Entry Point
# ==============================================================================
if __name__ == "__main__":
    main()