# ==============================================================================
# 0. Preamble & Imports
# ==============================================================================
# This script trains and evaluates an attention-based multi-scale CNN for sleep
# apnea detection. It performs an initial train-test split evaluation followed by
# a more robust Leave-One-Night-Out (LONO) cross-validation.
#
# To run, ensure all dependencies are installed and execute from the command line:
# python multiCNN_attention.py
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

# ==============================================================================
# 2. Model & Helper Class Definitions
# ==============================================================================

### NEW: Improved CNN with Attention and Multi-Scale Convolutions ###


class OSA_CNN_LSTM_Hybrid(nn.Module):
    """
    Hybrid model that combines CNN features with LSTM temporal modeling
    and includes both global pooling and LSTM-based final representation.
    """
    def __init__(self, n_features, n_outputs, n_timesteps):
        super(OSA_CNN_LSTM_Hybrid, self).__init__()
        
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_timesteps = n_timesteps
        
        # CNN feature extraction (similar to attention model)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-scale convolutions
        self.multiscale_conv = MultiScaleConv(64, 128)
        self.pool1 = nn.MaxPool1d(2)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(128, 128, kernel_size=5, dropout=0.15)
        self.res_block2 = ResidualBlock(128, 256, kernel_size=3, dropout=0.15)
        self.pool2 = nn.MaxPool1d(2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Global pooling branch (from CNN)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 256),  # LSTM features + CNN global features
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
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
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        # Input: (batch, timesteps, features)
        x = x.permute(0, 2, 1)  # (batch, features, timesteps)
        
        # CNN feature extraction
        x = self.initial_conv(x)
        x = self.multiscale_conv(x)
        x = self.pool1(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.pool2(x)
        
        # Global pooling branch
        global_features = self.global_pool(x).squeeze(-1)  # (batch, 256)
        
        # LSTM branch
        x_lstm = x.permute(0, 2, 1)  # (batch, sequence_length, features)
        lstm_out, (hidden, _) = self.lstm(x_lstm)
        
        # Use final hidden state from both directions
        lstm_features = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, 256)
        
        # Feature fusion
        fused_features = torch.cat([global_features, lstm_features], dim=1)
        fused_features = self.fusion(fused_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output


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

# ==============================================================================
# 3. Main Execution Block
# ==============================================================================
def main():
    """Main function to run the data processing, training, and evaluation pipeline."""

    parser = argparse.ArgumentParser(description="Train an attention-based CNN for sleep apnea detection.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing event_exports, respeck, and nasal_files folders.')
    parser.add_argument('--base_output_dir', type=str, required=True,
                        help='Base directory to save results (e.g., plots, checkpoints).')
    args = parser.parse_args()
    # --- Configuration & Constants ---
    print("--- 1. Setting up configuration and constants ---")
    EVENTS_FOLDER = os.path.join(args.data_dir, 'event_exports')
    RESPECK_FOLDER = os.path.join(args.data_dir, 'respeck')
    NASAL_FOLDER = os.path.join(args.data_dir, 'nasal_files')

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
    all_sessions_df_list = []
    event_files = glob.glob(os.path.join(EVENTS_FOLDER, '*_event_export.csv'))

    if not event_files:
        raise FileNotFoundError(f"No event files found in '{EVENTS_FOLDER}'.")

    print(f"Found {len(event_files)} event files. Processing each one...")

    for event_file_path in event_files:
        base_name = os.path.basename(event_file_path)
        session_id = base_name.split('_event_export.csv')[0]
        respeck_file_path = os.path.join(RESPECK_FOLDER, f'{session_id}_respeck.csv')
        nasal_file_path = os.path.join(NASAL_FOLDER, f'{session_id}_nasal.csv')

        if not all(os.path.exists(p) for p in [respeck_file_path, nasal_file_path]):
            print(f"  - WARNING: Skipping session '{session_id}'. A corresponding file is missing.")
            continue
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
            continue

        print(f"  - Applying precise interval-based labels...")
        df_respeck['Label'] = 0
        df_events['Duration_ms'] = (df_events['Duration'] * 1000).astype('int64')
        df_events['end_time_unix'] = df_events['timestamp_unix'] + df_events['Duration_ms']
        
        for label_id, event_names_in_group in EVENT_GROUP_TO_LABEL.items():
            df_filtered_events = df_events[df_events['Event'].isin(event_names_in_group)]
            for _, event in df_filtered_events.iterrows():
                start_event = event['timestamp_unix']
                end_event = event['end_time_unix']
                df_respeck.loc[df_respeck['timestamp_unix'].between(start_event, end_event), 'Label'] = label_id

        df_respeck['SessionID'] = session_id
        all_sessions_df_list.append(df_respeck)

    if not all_sessions_df_list:
        raise ValueError("Processing failed. No data was loaded.")

    df = pd.concat(all_sessions_df_list, ignore_index=True)

    print("\n----------------------------------------------------")
    print("Data loading with PURE signals complete.")
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Final class distribution in raw data: \n{df['Label'].value_counts(normalize=True)}")

    # --- Feature Engineering & Imputation ---
    print("\n--- 3. Engineering features, imputing missing values, and normalizing ---")
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
        print(f"\nWARNING: {final_nan_count} NaNs still remain in feature columns after imputation. Please investigate.")
    else:
        print("\nImputation complete. No NaNs remain in feature columns.")

    # --- Per-Session Normalization ---
    print("\nApplying per-session (per-subject) normalization...")
    df_normalized = df.copy()
    for session_id in df['SessionID'].unique():
        session_mask = df['SessionID'] == session_id
        session_features = df.loc[session_mask, FEATURE_COLUMNS]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(session_features)
        df_normalized.loc[session_mask, FEATURE_COLUMNS] = scaled_features
    print("Normalization complete.")

    # --- Windowing Data ---
    print("\n--- 4. Creating time-series windows ---")
    print(f"Number of classes: {N_OUTPUTS}")
    print(f"Class names: {CLASS_NAMES}")

    X, y, groups = [], [], []
    print("\nStarting the windowing process on normalized data...")
    for session_id, session_df in df_normalized.groupby('SessionID'):
        for i in range(0, len(session_df) - WINDOW_SIZE, STEP_SIZE):
            window_df = session_df.iloc[i : i + WINDOW_SIZE]
            features = window_df[FEATURE_COLUMNS].values
            label = stats.mode(window_df['Label'])[0]
            X.append(features)
            y.append(label)
            groups.append(session_id)

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    print("\nData windowing complete.")
    print("----------------------------------------------------")
    print(f"Shape of X (features): {X.shape}")
    print(f"Shape of y (labels):   {y.shape}")
    print(f"Shape of groups (IDs): {groups.shape}")
    print(f"Final class distribution across all windows: {Counter(y)}")
    
    # --- Device Setup ---
    print("\n--- 5. Setting up PyTorch device ---")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # --- Initial Train-Test Split & SMOTE ---
    print("\n--- 6. Performing initial train-test split for preliminary check ---")
    unique_session_ids = np.unique(groups)
    train_ids, test_ids = train_test_split(
        unique_session_ids, 
        test_size=2,
        random_state=RANDOM_STATE
    )
    train_mask = np.isin(groups, train_ids)
    test_mask = np.isin(groups, test_ids)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print("Train-test split complete.")
    print(f"Training set class distribution: {Counter(y_train)}")
    print(f"Testing set class distribution:  {Counter(y_test)}")

    nsamples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape((nsamples, n_timesteps * n_features))

    print("\nBalancing the training data using SMOTE...")
    print(f"  - Original training distribution: {Counter(y_train)}")
    
    class_counts = Counter(y_train)
    min_class_count = min(count for label, count in class_counts.items() if label != 0)
    k = min_class_count - 1

    if k < 1:
        print(f"Warning: Smallest minority class has {min_class_count} samples. Falling back to RandomOverSampler.")
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    else:
        print(f"Smallest minority class has {min_class_count} samples. Setting k_neighbors for SMOTE to {k}.")
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)

    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_reshaped, y_train)
    print(f"  - Resampled training distribution: {Counter(y_train_resampled)}")
    
    X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], n_timesteps, n_features))
    X_train_tensor = torch.from_numpy(X_train_resampled).float()
    y_train_tensor = torch.from_numpy(y_train_resampled).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("\nPyTorch DataLoaders created successfully.")

    # --- Initial Model Training & Evaluation ---
    print("\n--- 7. Training and Evaluating on the initial split ---")
    model = OSA_CNN_LSTM_Hybrid(n_features=n_features, n_outputs=N_OUTPUTS, n_timesteps=n_timesteps).to(device)
    print("\nPyTorch OSA_CNN_LSTM_Hybrid model created and moved to device.")

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=20, verbose=False, path='checkpoint.pt')

    print("\nStarting PyTorch model training with Early Stopping and LR Scheduler...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    print("\nModel training complete. Loading best model weights...")
    model.load_state_dict(torch.load('checkpoint.pt'))
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print('\nClassification Report (Initial Split)')
    print('---------------------------------------')
    print(classification_report(all_labels, all_preds, labels=range(N_OUTPUTS), target_names=CLASS_NAMES, zero_division=0))
    
    print('\nConfusion Matrix (Initial Split)')
    print('--------------------------------')
    cm = confusion_matrix(all_labels, all_preds, labels=range(N_OUTPUTS))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix (Initial Split)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha="right")
    plt.savefig("confusion_matrix_initial_split.png", bbox_inches='tight')
    # plt.show()

    # --- Leave-One-Night-Out Cross-Validation ---
    print("\n--- 8. Starting Leave-One-Night-Out Cross-Validation ---")
    all_fold_predictions, all_fold_true_labels = [], []
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    max_grad_norm = 1.0

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_night = np.unique(groups[test_idx])[0]
        print(f"--- FOLD {fold + 1}/{n_folds} (Testing on Night: {test_night}) ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"  - Original training distribution: {Counter(y_train)}")
        nsamples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape((nsamples, n_timesteps * n_features))
        
        class_counts = Counter(y_train)
        minority_classes = {label: count for label, count in class_counts.items() if label != 0}
        
        if not minority_classes:
            print("  - No minority classes in this fold's training data. Skipping resampling.")
            X_train_resampled, y_train_resampled = X_train_reshaped, y_train
        else:
            min_class_count = min(minority_classes.values())
            k = min_class_count - 1
            if k < 1:
                print(f"  - Warning: Smallest minority class has {min_class_count} samples. Using RandomOverSampler.")
                sampler = RandomOverSampler(random_state=RANDOM_STATE)
            else:
                print(f"  - Smallest minority class has {min_class_count} samples. Setting k_neighbors for SMOTE to {k}.")
                sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_reshaped, y_train)

        print(f"  - Resampled training distribution: {Counter(y_train_resampled)}")
        X_train_resampled = X_train_resampled.reshape(-1, n_timesteps, n_features)

        X_train_tensor = torch.from_numpy(X_train_resampled).float()
        y_train_tensor = torch.from_numpy(y_train_resampled).long()
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).long()
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        model = OSA_CNN_LSTM_Hybrid(n_features=n_features, n_outputs=N_OUTPUTS, n_timesteps=n_timesteps).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=20, verbose=False, path=f'lono_checkpoint_fold_lstm_hybrid_{fold}.pt')

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
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"  - Early stopping triggered at epoch {epoch + 1}.")
                break

        print(f"  - Training complete for fold {fold + 1}.")
        
        model.load_state_dict(torch.load(f'lono_checkpoint_fold_attn_{fold}.pt'))
        model.eval()
        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                fold_preds.extend(predicted.cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())
        
        all_fold_predictions.extend(fold_preds)
        all_fold_true_labels.extend(fold_labels)
        print(f"  - Evaluation complete for fold {fold + 1}.\n")

    # --- FINAL AGGREGATED LONO EVALUATION ---
    print("\n====================================================")
    print("Leave-One-Night-Out Cross-Validation Complete.")
    print("Aggregated Results Across All Folds:")
    print("====================================================")
    
    print(classification_report(all_fold_true_labels, all_fold_predictions, labels=range(N_OUTPUTS), target_names=CLASS_NAMES, zero_division=0))
    
    cm = confusion_matrix(all_fold_true_labels, all_fold_predictions, labels=range(N_OUTPUTS))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Aggregated Normalized Confusion Matrix (LONO)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha="right")
    plt.savefig("confusion_matrix_lono_aggregated.png", bbox_inches='tight')
    # plt.show()


# ==============================================================================
# 4. Script Entry Point
# ==============================================================================
if __name__ == "__main__":
    main()