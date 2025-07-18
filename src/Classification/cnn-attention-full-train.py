
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
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import pickle

class ProductionEarlyStopping:
    """
    Early stopping for production training based on k-fold validation results.
    Uses training loss plateau detection with guidance from cross-validation.
    """
    def __init__(self, target_epochs=80, patience=15, min_delta=0.001, min_epochs=30):
        self.target_epochs = target_epochs  # Based on k-fold CV results
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = float('inf')
        self.epoch = 0
        self.should_stop = False
        
    def __call__(self, loss):
        self.epoch += 1
        
        # Don't stop before minimum epochs
        if self.epoch < self.min_epochs:
            return False
            
        # Primary stopping: reached target epochs from k-fold
        if self.epoch >= self.target_epochs:
            print(f"Reached target epochs ({self.target_epochs}) based on k-fold CV results. Stopping training.")
            self.should_stop = True
            return True
            
        # Secondary stopping: training loss plateau
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            
        # If we're past 70% of target epochs and loss is plateauing, stop
        if self.epoch > (self.target_epochs * 0.7) and self.counter >= self.patience:
            print(f"Training loss plateaued for {self.patience} epochs after epoch {self.epoch - self.patience}. Stopping training.")
            self.should_stop = True
            return True
            
        return False

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
        self.res_block1 = ResidualBlock(128, 128, kernel_size=5, dropout=0.15)
        self.res_block2 = ResidualBlock(128, 256, kernel_size=5, stride=2, dropout=0.15)
        
        # Second multi-scale layer
        self.multiscale2 = MultiScaleConv(256, 256)
        self.pool2 = nn.MaxPool1d(2)
        
        # More residual blocks
        self.res_block3 = ResidualBlock(256, 256, kernel_size=3, dropout=0.2)
        self.res_block4 = ResidualBlock(256, 512, kernel_size=3, stride=2, dropout=0.2)
        
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
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
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

def save_model_artifacts(model, scaler_info, config, output_dir):
    """Save the final model and associated artifacts for deployment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model state dict
    model_path = os.path.join(output_dir, f'final_sleep_apnea_model_{timestamp}.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save model architecture and configuration
    config_path = os.path.join(output_dir, f'model_config_{timestamp}.pkl')
    model_info = {
        'model_architecture': 'ImprovedCNN',
        'n_features': model.n_features,
        'n_outputs': model.n_outputs,
        'n_timesteps': model.n_timesteps,
        'feature_columns': config['feature_columns'],
        'class_names': config['class_names'],
        'window_size': config['window_size'],
        'step_size': config['step_size'],
        'sampling_rate': config['sampling_rate'],
        'scaler_info': scaler_info,
        'timestamp': timestamp
    }
    
    with open(config_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"Final model saved to: {model_path}")
    print(f"Model configuration saved to: {config_path}")
    return model_path, config_path

def train_final_model_with_early_stopping(final_model, final_train_loader, device, args):
    """
    Train final model with early stopping based on k-fold results.
    """
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-6)
    final_criterion = nn.CrossEntropyLoss()
    max_grad_norm = 1.0
    
    # Initialize early stopping based on your k-fold results
    # Average stopping epoch from your k-fold: ~80 epochs
    early_stopping = ProductionEarlyStopping(
        target_epochs=80,  # Based on your k-fold average
        patience=15,       # Allow some patience for plateau detection
        min_delta=0.001,   # Minimum improvement to reset counter
        min_epochs=40      # Don't stop before this many epochs
    )
    
    training_losses = []
    training_accuracies = []
    
    print(f"Starting final model training (max {args.epochs} epochs)...")
    print(f"Early stopping target: {early_stopping.target_epochs} epochs (based on k-fold CV)")
    print("Note: No validation set is used as we're training on all available data.")
    
    for epoch in range(args.epochs):
        final_model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (inputs, labels) in enumerate(final_train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            final_optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = final_criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_grad_norm)
            final_optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(final_train_loader)
        train_accuracy = 100 * correct_predictions / total_predictions
        training_losses.append(avg_train_loss)
        training_accuracies.append(train_accuracy)
        
        # Update learning rate
        final_scheduler.step(avg_train_loss)
        
        # Check early stopping
        if early_stopping(avg_train_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = final_optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{args.epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy:.2f}%, "
                  f"LR: {current_lr:.2e}")
    
    final_epoch = len(training_losses)
    print(f"\nFinal model training complete! Trained for {final_epoch} epochs.")
    
    return training_losses, training_accuracies, final_epoch

# ==============================================================================
# 3. Main Execution Block
# ==============================================================================
def main():
    """Main function to train the final production model."""

    parser = argparse.ArgumentParser(description="Train final production model for sleep apnea detection.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing event_exports, respeck, and nasal_files folders.')
    parser.add_argument('--base_output_dir', type=str, required=True,
                        help='Base directory to save results (e.g., plots, checkpoints).')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs for final model (default: 80).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64).')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for training (default: 1e-4).')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.base_output_dir, exist_ok=True)
    
    # --- Configuration & Constants ---
    print("="*60)
    print("FINAL SLEEP APNEA MODEL TRAINING")
    print("="*60)
    print("Training production model on all available data...")
    print(f"Output directory: {args.base_output_dir}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    
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
    FEATURE_COLUMNS = ['breathingSignal', 'activityLevel', 'breathingRate', 'x', 'y', 'z']

    # --- Data Loading ---
    print(f"\n--- 1. Loading and preprocessing data ---")
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

    print(f"\nData loading complete.")
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Number of sessions: {df['SessionID'].nunique()}")
    print(f"Class distribution in raw data:")
    for i, class_name in enumerate(CLASS_NAMES):
        count = sum(df['Label'] == i)
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count:,} samples ({percentage:.1f}%)")

    # --- Feature Engineering & Imputation ---
    print(f"\n--- 2. Feature engineering and preprocessing ---")
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
        print("Imputation complete. No NaNs remain in feature columns.")

    # --- Per-Session Normalization ---
    print("\nApplying per-session (per-subject) normalization...")
    df_normalized = df.copy()
    scaler_info = {}  # Store scaler parameters for each session
    
    for session_id in df['SessionID'].unique():
        session_mask = df['SessionID'] == session_id
        session_features = df.loc[session_mask, FEATURE_COLUMNS]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(session_features)
        df_normalized.loc[session_mask, FEATURE_COLUMNS] = scaled_features
        
        # Store scaler parameters for potential future use
        scaler_info[session_id] = {
            'mean': scaler.mean_,
            'scale': scaler.scale_
        }
    
    print("Normalization complete.")

    # --- Windowing Data ---
    print(f"\n--- 3. Creating time-series windows ---")
    print(f"Window size: {WINDOW_SIZE} samples ({WINDOW_DURATION_SEC}s)")
    print(f"Step size: {STEP_SIZE} samples ({OVERLAP_PERCENTAGE*100}% overlap)")
    print(f"Sampling rate: {SAMPLING_RATE_HZ} Hz")

    X, y, groups = [], [], []
    print("Starting the windowing process...")
    for session_id, session_df in df_normalized.groupby('SessionID'):
        session_windows = 0
        for i in range(0, len(session_df) - WINDOW_SIZE, STEP_SIZE):
            window_df = session_df.iloc[i : i + WINDOW_SIZE]
            features = window_df[FEATURE_COLUMNS].values
            label = stats.mode(window_df['Label'])[0]
            X.append(features)
            y.append(label)
            groups.append(session_id)
            session_windows += 1
        print(f"  - Session {session_id}: {session_windows} windows")

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    print(f"\nWindowing complete.")
    print(f"Total windows: {len(X):,}")
    print(f"Window shape: {X.shape}")
    print(f"Final windowed class distribution:")
    for i, class_name in enumerate(CLASS_NAMES):
        count = sum(y == i)
        percentage = (count / len(y)) * 100
        print(f"  {class_name}: {count:,} windows ({percentage:.1f}%)")
    
    # --- Device Setup ---
    print(f"\n--- 4. Setting up PyTorch device ---")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # --- FINAL MODEL TRAINING ON ALL DATA ---
    print(f"\n--- 5. TRAINING FINAL MODEL ON ALL DATA ---")
    print("Training the final production model on all available data...")
    
    # Prepare all data for final training
    nsamples, n_timesteps, n_features = X.shape
    X_all_reshaped = X.reshape((nsamples, n_timesteps * n_features))
    
    print(f"Original distribution across all data: {Counter(y)}")
    
    # Apply SMOTE to all data
    class_counts = Counter(y)
    min_class_count = min(count for label, count in class_counts.items() if label != 0)
    k = min_class_count - 1

    if k < 1:
        print(f"Warning: Smallest minority class has {min_class_count} samples. Using RandomOverSampler.")
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    else:
        print(f"Smallest minority class has {min_class_count} samples. Setting k_neighbors for SMOTE to {k}.")
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    
    X_final_resampled, y_final_resampled = sampler.fit_resample(X_all_reshaped, y)
    print(f"Final resampled distribution: {Counter(y_final_resampled)}")
    
    X_final_resampled = X_final_resampled.reshape((X_final_resampled.shape[0], n_timesteps, n_features))
    
    # Create final training dataset
    X_final_tensor = torch.from_numpy(X_final_resampled).float()
    y_final_tensor = torch.from_numpy(y_final_resampled).long()
    
    final_dataset = TensorDataset(X_final_tensor, y_final_tensor)
    final_train_loader = DataLoader(final_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize final model
    final_model = ImprovedCNN(n_features=n_features, n_outputs=N_OUTPUTS, n_timesteps=n_timesteps).to(device)
    total_params = sum(p.numel() for p in final_model.parameters())
    trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"\nFinal model initialized:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    training_losses, training_accuracies, final_epoch = train_final_model_with_early_stopping(
        final_model, final_train_loader, device, args
    )
    
    # Final model training configuration
    # final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    # final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-6)
    # final_criterion = nn.CrossEntropyLoss()
    # max_grad_norm = 1.0
    
    # # Train final model
    # print(f"\nStarting final model training for {args.epochs} epochs...")
    # print("Note: No validation set is used as we're training on all available data.")
    
    # training_losses = []
    # training_accuracies = []
    
    # for epoch in range(args.epochs):
    #     final_model.train()
    #     running_loss = 0.0
    #     correct_predictions = 0
    #     total_predictions = 0
        
    #     for batch_idx, (inputs, labels) in enumerate(final_train_loader):
    #         inputs, labels = inputs.to(device), labels.to(device)
            
    #         final_optimizer.zero_grad()
    #         outputs = final_model(inputs)
    #         loss = final_criterion(outputs, labels)
    #         loss.backward()
            
    #         # Gradient clipping
    #         torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_grad_norm)
    #         final_optimizer.step()
            
    #         running_loss += loss.item()
            
    #         # Calculate training accuracy
    #         _, predicted = torch.max(outputs.data, 1)
    #         total_predictions += labels.size(0)
    #         correct_predictions += (predicted == labels).sum().item()
        
    #     avg_train_loss = running_loss / len(final_train_loader)
    #     train_accuracy = 100 * correct_predictions / total_predictions
    #     training_losses.append(avg_train_loss)
    #     training_accuracies.append(train_accuracy)
        
    #     # Update learning rate (using training loss since we don't have validation)
    #     final_scheduler.step(avg_train_loss)
        
    #     if (epoch + 1) % 10 == 0 or epoch == 0:
    #         current_lr = final_optimizer.param_groups[0]['lr']
    #         print(f"Epoch [{epoch+1}/{args.epochs}], "
    #               f"Train Loss: {avg_train_loss:.4f}, "
    #               f"Train Accuracy: {train_accuracy:.2f}%, "
    #               f"LR: {current_lr:.2e}")
    
    # print("\nFinal model training complete!")
    
    # Save final model and artifacts
    print(f"\n--- 6. Saving final model and configuration ---")
    
    config = {
        'feature_columns': FEATURE_COLUMNS,
        'class_names': CLASS_NAMES,
        'window_size': WINDOW_SIZE,
        'step_size': STEP_SIZE,
        'sampling_rate': SAMPLING_RATE_HZ,
        'n_outputs': N_OUTPUTS,
        'window_duration_sec': WINDOW_DURATION_SEC,
        'overlap_percentage': OVERLAP_PERCENTAGE,
        'event_group_to_label': EVENT_GROUP_TO_LABEL,
        'label_to_event_group_name': LABEL_TO_EVENT_GROUP_NAME,
        'training_epochs': final_epoch,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'actual_training_epochs': final_epoch,
        'target_epochs_from_cv': 80,
        'early_stopping_triggered': final_epoch < args.epochs,
        'final_train_loss': training_losses[-1],
        'final_train_accuracy': training_accuracies[-1]
    }
    
    model_path, config_path = save_model_artifacts(final_model, scaler_info, config, args.base_output_dir)
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training loss
    ax1.plot(training_losses, 'b-', linewidth=2)
    ax1.set_title('Final Model Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Training accuracy
    ax2.plot(training_accuracies, 'g-', linewidth=2)
    ax2.set_title('Final Model Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.base_output_dir, "final_model_training_curves.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create a summary plot of class distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original class distribution
    original_counts = [sum(y == i) for i in range(N_OUTPUTS)]
    ax1.bar(CLASS_NAMES, original_counts, color='skyblue', alpha=0.7)
    ax1.set_title('Original Class Distribution')
    ax1.set_ylabel('Number of Windows')
    ax1.tick_params(axis='x', rotation=45)
    
    # Resampled class distribution
    resampled_counts = [sum(y_final_resampled == i) for i in range(N_OUTPUTS)]
    ax2.bar(CLASS_NAMES, resampled_counts, color='lightcoral', alpha=0.7)
    ax2.set_title('Resampled Class Distribution (SMOTE)')
    ax2.set_ylabel('Number of Windows')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.base_output_dir, "class_distributions.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Generate final model summary
    print("\n" + "="*80)
    print("FINAL MODEL TRAINING SUMMARY")
    print("="*80)
    print(f"Model Architecture: ImprovedCNN with Attention & Multi-Scale Convolutions")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Training Epochs: {final_epoch}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Final Training Loss: {training_losses[-1]:.4f}")
    print(f"Final Training Accuracy: {training_accuracies[-1]:.2f}%")
    print(f"Training Data Shape: {X_final_resampled.shape}")
    print(f"Original Windows: {len(X):,}")
    print(f"Resampled Windows: {len(X_final_resampled):,}")
    print(f"Device Used: {device}")
    
    print(f"\nData Configuration:")
    print(f"  - Features Used: {FEATURE_COLUMNS}")
    print(f"  - Number of Classes: {N_OUTPUTS}")
    print(f"  - Window Size: {WINDOW_SIZE} samples ({WINDOW_DURATION_SEC}s)")
    print(f"  - Step Size: {STEP_SIZE} samples ({OVERLAP_PERCENTAGE*100}% overlap)")
    print(f"  - Sampling Rate: {SAMPLING_RATE_HZ} Hz")
    print(f"  - Sessions Processed: {len(scaler_info)}")
    
    print(f"\nModel Files:")
    print(f"  - Model Weights: {model_path}")
    print(f"  - Configuration: {config_path}")
    print(f"  - Training Curves: {os.path.join(args.base_output_dir, 'final_model_training_curves.png')}")
    print(f"  - Class Distributions: {os.path.join(args.base_output_dir, 'class_distributions.png')}")
    
    print(f"\nFinal Class Distribution in Training Data:")
    for i, class_name in enumerate(CLASS_NAMES):
        original_count = sum(y == i)
        resampled_count = sum(y_final_resampled == i)
        original_pct = (original_count / len(y)) * 100
        resampled_pct = (resampled_count / len(y_final_resampled)) * 100
        print(f"  {class_name}:")
        print(f"    Original: {original_count:,} windows ({original_pct:.1f}%)")
        print(f"    Resampled: {resampled_count:,} windows ({resampled_pct:.1f}%)")
    
    print(f"\nModel Architecture Details:")
    print(f"  - Multi-scale convolutions with kernels: 3, 7, 15, 1")
    print(f"  - Residual blocks with skip connections")
    print(f"  - Channel attention mechanism")
    print(f"  - Spatial attention mechanism")
    print(f"  - Batch normalization and dropout regularization")
    print(f"  - Global adaptive pooling")
    print(f"  - Fully connected classifier with 3 hidden layers")
    
    print(f"\nTraining Configuration:")
    print(f"  - Optimizer: Adam with weight decay (1e-3)")
    print(f"  - Learning Rate Scheduler: ReduceLROnPlateau")
    print(f"  - Gradient Clipping: Max norm = {max_grad_norm}")
    print(f"  - Loss Function: CrossEntropyLoss")
    print(f"  - Data Augmentation: SMOTE oversampling")
    print(f"  - Per-session normalization applied")
    
    # Save training metrics
    metrics_path = os.path.join(args.base_output_dir, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(metrics_path, 'w') as f:
        f.write("FINAL MODEL TRAINING METRICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Training Epochs: {final_epoch}\n")
        f.write(f"Final Training Loss: {training_losses[-1]:.6f}\n")
        f.write(f"Final Training Accuracy: {training_accuracies[-1]:.2f}%\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Training Data Shape: {X_final_resampled.shape}\n\n")
        
        f.write("Epoch-by-Epoch Metrics:\n")
        f.write("Epoch\tTrain_Loss\tTrain_Acc\n")
        for epoch, (loss, acc) in enumerate(zip(training_losses, training_accuracies)):
            f.write(f"{epoch+1}\t{loss:.6f}\t{acc:.2f}\n")
    
    print(f"  - Training Metrics: {metrics_path}")
    
    print(f"\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"\nActual training epochs: {final_epoch} (target was {args.epochs})")
    print(f"Early stopping {'was' if final_epoch < args.epochs else 'was not'} triggered")
    print("="*80)
    print(f"Your final production model is ready for deployment!")
    print(f"Use the saved model weights and configuration for inference on new data.")
    print(f"The model achieved {training_accuracies[-1]:.2f}% accuracy on the training data.")
    print(f"Based on your LONO results, this model should generalize well to new patients.")
    print("="*80)


# ==============================================================================
# 4. Script Entry Point
# ==============================================================================
if __name__ == "__main__":
    main()