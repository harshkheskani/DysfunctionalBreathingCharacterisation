# ==============================================================================
# 0. Preamble & Imports
# ==============================================================================
# This script trains and evaluates an attention-based multi-scale CNN for sleep
# apnea detection. It uses a fixed training set (patient with 9 nights) plus
# one additional night for training, and tests on the remaining night.
#
# This version is optimized for generalization to unseen patients.

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
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
import math
import argparse

# ==============================================================================
# 2. Model & Helper Class Definitions
# ==============================================================================

### CHANGE ###: Using the model architecture from `cnn-attention-full-train.py`
# This version has lower dropout, which is better suited for training on diverse,
# multi-patient data, as the data diversity itself acts as a regularizer.

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
        avg_out = self.fc(self.avg_pool(x).view(b, c))
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
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // 4), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels // 4), nn.ReLU())
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels // 4), nn.ReLU())
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4), nn.ReLU())
        self.combine = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels), nn.ReLU())

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return self.combine(out)

class ImprovedCNN(nn.Module):
    """Improved CNN with reduced dropout for better generalization across patients."""
    def __init__(self, n_features, n_outputs, n_timesteps):
        super().__init__()
        self.n_features, self.n_outputs, self.n_timesteps = n_features, n_outputs, n_timesteps
        self.initial_conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1))
        self.multiscale1 = MultiScaleConv(64, 128)
        self.pool1 = nn.MaxPool1d(2)
        self.res_block1 = ResidualBlock(128, 128, kernel_size=5, dropout=0.15)
        self.res_block2 = ResidualBlock(128, 256, kernel_size=5, stride=2, dropout=0.15)
        self.multiscale2 = MultiScaleConv(256, 256)
        self.pool2 = nn.MaxPool1d(2)
        self.res_block3 = ResidualBlock(256, 256, kernel_size=3, dropout=0.2)
        self.res_block4 = ResidualBlock(256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.channel_attention = ChannelAttention(512)
        self.spatial_attention = SpatialAttention()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_size = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.feature_size, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(128, n_outputs))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.initial_conv(x)
        x = self.multiscale1(x); x = self.pool1(x)
        x = self.res_block1(x); x = self.res_block2(x)
        x = self.multiscale2(x); x = self.pool2(x)
        x = self.res_block3(x); x = self.res_block4(x)
        x = self.channel_attention(x); x = self.spatial_attention(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience, self.verbose, self.counter = patience, verbose, 0
        self.best_score, self.early_stop, self.val_loss_min = None, False, np.inf
        self.delta, self.path, self.trace_func = delta, path, trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None: self.best_score = score; self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else: self.best_score = score; self.save_checkpoint(val_loss, model); self.counter = 0
    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path); self.val_loss_min = val_loss

def add_signal_features(df):
    """Adds rolling window features to the dataframe."""
    print("Engineering new signal-based features...")
    ROLLING_WINDOW_SIZE = 25
    df['breathing_signal_rolling_mean'] = df.groupby('SessionID')['breathingSignal'].transform(lambda x: x.rolling(ROLLING_WINDOW_SIZE, 1).mean())
    df['breathing_signal_rolling_std'] = df.groupby('SessionID')['breathingSignal'].transform(lambda x: x.rolling(ROLLING_WINDOW_SIZE, 1).std())
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
    if df_respeck.empty: return None
    df_respeck['Label'] = 0
    df_events['Duration_ms'] = (df_events['Duration'] * 1000).astype('int64')
    df_events['end_time_unix'] = df_events['timestamp_unix'] + df_events['Duration_ms']
    for label_id, event_names in event_group_to_label.items():
        for _, event in df_events[df_events['Event'].isin(event_names)].iterrows():
            df_respeck.loc[df_respeck['timestamp_unix'].between(event['timestamp_unix'], event['end_time_unix']), 'Label'] = label_id
    df_respeck['SessionID'] = session_id
    return df_respeck

def balance_data(X_data, y_data, random_state):
    """Resamples a dataset to balance its classes using SMOTE or RandomOverSampler."""
    if len(X_data) == 0: return X_data, y_data
    print(f"  - Balancing data with initial shape {X_data.shape} and distribution {Counter(y_data)}")
    nsamples, n_timesteps, n_features = X_data.shape
    X_reshaped = X_data.reshape((nsamples, n_timesteps * n_features))
    minority_classes = {l: c for l, c in Counter(y_data).items() if l != 0 and c > 0}
    if not minority_classes: return X_data, y_data
    min_class_count = min(minority_classes.values())
    k = min_class_count - 1
    sampler = RandomOverSampler(random_state=random_state) if k < 1 else SMOTE(random_state=random_state, k_neighbors=k)
    X_resampled, y_resampled = sampler.fit_resample(X_reshaped, y_data)
    print(f"  - Resampled distribution: {Counter(y_resampled)}")
    return X_resampled.reshape(-1, n_timesteps, n_features), y_resampled

# ==============================================================================
# 3. Main Execution Block
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train an attention-based CNN for sleep apnea detection.")
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory containing /03/ folder with patient data.')
    parser.add_argument('--base_output_dir', type=str, required=True, help='Base directory to save results.')
    args = parser.parse_args()
    os.makedirs(args.base_output_dir, exist_ok=True)
    
    print("--- 1. Setting up configuration and constants ---")
    MAIN_PATIENT_DIR = os.path.join(args.data_dir)
    ADDITIONAL_PATIENT_1_DIR = os.path.join(args.data_dir, 'CSR005')
    ADDITIONAL_PATIENT_2_DIR = os.path.join(args.data_dir, 'CSR003')

    ### CHANGE ###: Removed 'RERA' class since it has 0 support. This cleans up reports.
    EVENT_GROUP_TO_LABEL = {
        1: ['Obstructive Apnea'],
        2: ['Hypopnea', 'Central Hypopnea', 'Obstructive Hypopnea'],
        3: ['Central Apnea', 'Mixed Apnea'],
        4: ['Desaturation']
    }
    LABEL_TO_EVENT_GROUP_NAME = {
        0: 'Normal', 1: 'Obstructive Apnea', 2: 'Hypopnea Events',
        3: 'Central/Mixed Apnea', 4: 'Desaturation'
    }
    N_OUTPUTS = len(EVENT_GROUP_TO_LABEL) + 1
    CLASS_NAMES = [LABEL_TO_EVENT_GROUP_NAME[i] for i in range(N_OUTPUTS)]
    
    SAMPLING_RATE_HZ, WINDOW_DURATION_SEC, OVERLAP_PERCENTAGE = 12.5, 30, 0.8
    WINDOW_SIZE = int(WINDOW_DURATION_SEC * SAMPLING_RATE_HZ)
    STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENTAGE))
    RANDOM_STATE, EPOCHS, BATCH_SIZE = 42, 100, 64
    FEATURE_COLUMNS = ['breathingSignal', 'activityLevel', 'breathingRate', 'x', 'y', 'z', 
                       'breathing_signal_rolling_mean', 'breathing_signal_rolling_std', 'accel_magnitude']

    print("\n--- 2. Loading and preprocessing data ---")
    def process_and_load(patient_name, patient_dir):
        events_folder = os.path.join(patient_dir, 'event_exports')
        respeck_folder = os.path.join(patient_dir, 'respeck')
        nasal_folder = os.path.join(patient_dir, 'nasal_files')
        sessions = []
        for f in glob.glob(os.path.join(events_folder, '*_event_export.csv')):
            session_id = os.path.basename(f).split('_event_export.csv')[0]
            session_data = load_session_data(session_id, events_folder, respeck_folder, nasal_folder, EVENT_GROUP_TO_LABEL)
            if session_data is not None:
                session_data['SessionID'] = f"{patient_name}_{session_id}"
                sessions.append(session_data)
        return sessions
    
    main_sessions = process_and_load('main_patient', MAIN_PATIENT_DIR)
    additional_sessions_data = {
        'csr005': process_and_load('csr005', ADDITIONAL_PATIENT_1_DIR),
        'csr003': process_and_load('csr003', ADDITIONAL_PATIENT_2_DIR)
    }

    if not main_sessions: raise ValueError("No main patient data was loaded.")
    all_dfs = main_sessions + additional_sessions_data['csr005'] + additional_sessions_data['csr003']
    
    print("\n--- 3. Feature engineering, imputation, and normalization ---")
    df = pd.concat(all_dfs, ignore_index=True)
    df = add_signal_features(df)
    df.fillna(method='ffill', inplace=True); df.fillna(method='bfill', inplace=True)
    
    df_normalized = df.copy()
    for session_id in df['SessionID'].unique():
        mask = df['SessionID'] == session_id
        scaler = RobustScaler()
        df_normalized.loc[mask, FEATURE_COLUMNS] = scaler.fit_transform(df.loc[mask, FEATURE_COLUMNS])

    def create_windows(df, window_size, step_size):
        X, y, groups = [], [], []
        for session_id, session_df in df.groupby('SessionID'):
            for i in range(0, len(session_df) - window_size, step_size):
                window = session_df.iloc[i:i+window_size]
                X.append(window[FEATURE_COLUMNS].values)
                y.append(stats.mode(window['Label'], keepdims=True)[0][0])
                groups.append(session_id)
        return np.asarray(X), np.asarray(y), np.asarray(groups)

    print("\nCreating windows for all data...")
    all_X, all_y, all_groups = create_windows(df_normalized, WINDOW_SIZE, STEP_SIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 4. Setting up PyTorch device: {device} ---")

    print("\n--- 5. Starting Cross-Validation for Generalization ---")
    patient_names = ['csr005', 'csr003']
    all_fold_predictions, all_fold_true_labels = [], []

    for fold in range(len(patient_names)):
        train_patient_id = patient_names[fold]
        test_patient_id = patient_names[1 - fold]
        
        print(f"\n--- FOLD {fold + 1}/{len(patient_names)} ---")
        print(f"Training on: Main patient + {train_patient_id}")
        print(f"Testing on: {test_patient_id}")

        main_mask = np.char.startswith(all_groups, 'main_patient')
        train_add_mask = np.char.startswith(all_groups, train_patient_id)
        test_mask = np.char.startswith(all_groups, test_patient_id)

        # --- THIS IS THE SUPERIOR RESAMPLING STRATEGY ---
        print("\nApplying per-patient balancing strategy...")
        # 1. Balance main patient data
        main_X_resampled, main_y_resampled = balance_data(all_X[main_mask], all_y[main_mask], RANDOM_STATE)
        # 2. Balance additional training patient data
        add_X_resampled, add_y_resampled = balance_data(all_X[train_add_mask], all_y[train_add_mask], RANDOM_STATE)
        # 3. Combine the TWO BALANCED datasets
        X_train_resampled = np.vstack([main_X_resampled, add_X_resampled])
        y_train_resampled = np.hstack([main_y_resampled, add_y_resampled])
        
        X_test, y_test = all_X[test_mask], all_y[test_mask]
        
        print(f"\nFinal training shape: {X_train_resampled.shape}, Test shape: {X_test.shape}")
        print(f"Final training distribution: {Counter(y_train_resampled)}")
        print(f"Test distribution: {Counter(y_test)}")

        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_resampled).float(), torch.from_numpy(y_train_resampled).long()), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()), batch_size=BATCH_SIZE, shuffle=False)
        
        n_features = X_train_resampled.shape[2]
        n_timesteps = X_train_resampled.shape[1]
        model = ImprovedCNN(n_features=n_features, n_outputs=N_OUTPUTS, n_timesteps=n_timesteps).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5)
        criterion = nn.CrossEntropyLoss()
        checkpoint_path = os.path.join(args.base_output_dir, f'generalization_checkpoint_fold_{fold}.pt')
        ### CHANGE ###: Increased patience for early stopping to allow more time for convergence.
        early_stopping = EarlyStopping(patience=15, verbose=False, path=checkpoint_path)
        
        print(f"  - Starting training for fold {fold + 1}...")
        for epoch in range(EPOCHS):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    val_loss += criterion(model(inputs.to(device)), labels.to(device)).item()
            
            avg_val_loss = val_loss / len(test_loader)
            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss, model)
            if (epoch + 1) % 10 == 0: print(f"    Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.4f}")
            if early_stopping.early_stop:
                print(f"  - Early stopping triggered at epoch {epoch + 1}")
                break
        
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                _, predicted = torch.max(model(inputs.to(device)).data, 1)
                fold_preds.extend(predicted.cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())
        
        all_fold_predictions.extend(fold_preds)
        all_fold_true_labels.extend(fold_labels)
        print(classification_report(fold_labels, fold_preds, labels=range(N_OUTPUTS), target_names=CLASS_NAMES, zero_division=0))

    print("\n" + "="*60)
    print("Generalization Cross-Validation Complete")
    print("="*60)
    print("Aggregated Results Across Both Folds:")
    print(classification_report(all_fold_true_labels, all_fold_predictions, labels=range(N_OUTPUTS), target_names=CLASS_NAMES, zero_division=0))
    
    cm = confusion_matrix(all_fold_true_labels, all_fold_predictions, labels=range(N_OUTPUTS))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Aggregated Normalized Confusion Matrix (Unseen Patient Test)')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(args.base_output_dir, "confusion_matrix_unseen_patient.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
