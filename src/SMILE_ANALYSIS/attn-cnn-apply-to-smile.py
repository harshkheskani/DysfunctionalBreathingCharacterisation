# apply_to_smile.py

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import pytz 

warnings.filterwarnings('ignore')

# --- Advanced Model Architecture Components ---
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

class ApneaDetector:
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        print(f"Using device: {self.device}")
        self.scotland_tz = pytz.timezone('Europe/London')
        
        # Load configuration first
        self.config = self._load_config()
        
        # Set up parameters from config
        self.FEATURE_COLUMNS = self.config['feature_columns']
        self.WINDOW_SIZE = self.config['window_size']
        self.STEP_SIZE = self.config['step_size']
        self.CLASS_NAMES = self.config['class_names']
        self.scaler_info = self.config['scaler_info']
        
        # Find the obstructive apnea label
        self.APNEA_LABEL = None
        for idx, class_name in enumerate(self.CLASS_NAMES):
            if 'obstructive' in class_name.lower() or 'apnea' in class_name.lower():
                self.APNEA_LABEL = idx
                break
        
        if self.APNEA_LABEL is None:
            print("Warning: Could not find obstructive apnea class in class names")
            print(f"Available classes: {self.CLASS_NAMES}")
            self.APNEA_LABEL = 1  # Default fallback
        
        print(f"Target class: '{self.CLASS_NAMES[self.APNEA_LABEL]}' (label: {self.APNEA_LABEL})")
        
        # Load model
        self.model = self._load_model()

    def _load_config(self):
        print(f"Loading configuration from {self.config_path}...")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        
        with open(self.config_path, 'rb') as f:
            config = pickle.load(f)
        
        print("Configuration loaded successfully.")
        return config

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        
        # Create model instance using config parameters
        model = ImprovedCNN(
            n_features=self.config['n_features'],
            n_outputs=self.config['n_outputs'],
            n_timesteps=self.config['n_timesteps']
        )
        
        # Load state dict
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        print("Model loaded successfully and set to evaluation mode.")
        return model

    def _filter_sleep_window(self, df):
        """Filters DataFrame to include data only between 9 PM and 10 AM Scotland time."""
        timestamp_col = 'timestamp_unix'
        if timestamp_col not in df.columns:
            print(f"  - Warning: '{timestamp_col}' not found. Cannot filter by sleep window. Processing all data.")
            return df

        print("Filtering data for sleep window (9pm - 10am Scotland time)...")
        
        df['datetime_scotland'] = pd.to_datetime(df[timestamp_col], unit='ms', utc=True).dt.tz_convert(self.scotland_tz)
        
        hour = df['datetime_scotland'].dt.hour
        
        # The sleep window is from 9 PM (21:00) until 9:59 AM the next day.
        is_in_sleep_window = (hour >= 21) | (hour < 10)
        
        filtered_df = df[is_in_sleep_window].copy()
        
        print(f"  - Original data points: {len(df)}")
        print(f"  - Sleep window data points: {len(filtered_df)}")
        
        filtered_df.drop(columns=['datetime_scotland'], inplace=True)
        return filtered_df

    def _load_and_prep_single_file(self, file_path):
        """Loads and prepares a single Respeck CSV file without scaling."""
        try:
            df = pd.read_csv(file_path, na_values=['Na'])
            
            df.rename(columns={'interpolatedPhoneTimestamp': 'timestamp_unix'}, inplace=True)
            
            # Ensure timestamp is numeric before any filtering
            df['timestamp_unix'] = pd.to_numeric(df['timestamp_unix'], errors='coerce')
            df.dropna(subset=['timestamp_unix'], inplace=True)
            df['timestamp_unix'] = df['timestamp_unix'].astype('int64')

            if not all(col in df.columns for col in self.FEATURE_COLUMNS):
                print(f"  - Missing one or more feature columns in {file_path}. Skipping.")
                return None
            
            # Ensure feature columns are numeric
            for col in self.FEATURE_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df
        except Exception as e:
            print(f"  - Error processing file {file_path}: {e}")
            return None
    
    def _create_windows_with_timestamps(self, df):
        windows, timestamps = [], []
        
        if len(df) < self.WINDOW_SIZE:
            return np.array([]), []

        for i in range(0, len(df) - self.WINDOW_SIZE, self.STEP_SIZE):
            window_df = df.iloc[i : i + self.WINDOW_SIZE]
            windows.append(window_df[self.FEATURE_COLUMNS].values)
            timestamps.append(window_df['timestamp_unix'].iloc[0])
            
        return np.asarray(windows), timestamps

    def _run_inference_in_batches(self, windows_np, batch_size=1024):
        """Runs model inference in batches to manage memory."""
        all_predictions = []
        windows_tensor = torch.from_numpy(windows_np).float()
        
        with torch.no_grad():
            for i in range(0, len(windows_tensor), batch_size):
                batch = windows_tensor[i:i + batch_size].to(self.device)
                outputs = self.model(batch)
                _, predictions = torch.max(outputs, 1)
                all_predictions.extend(predictions.cpu().numpy())
        
        return np.array(all_predictions)

    def _compute_robust_scaler_params(self):
        """Compute robust scaler parameters using median and percentiles."""
        all_means = []
        all_scales = []
        
        for patient_id, scaler_params in self.scaler_info.items():
            all_means.append(scaler_params['mean'])
            all_scales.append(scaler_params['scale'])
        
        all_means = np.array(all_means)
        all_scales = np.array(all_scales)
        
        # Use median for robustness against outliers
        robust_mean = np.median(all_means, axis=0)
        robust_scale = np.median(all_scales, axis=0)
        
        # Compute additional robust statistics
        mean_q25 = np.percentile(all_means, 25, axis=0)
        mean_q75 = np.percentile(all_means, 75, axis=0)
        scale_q25 = np.percentile(all_scales, 25, axis=0)
        scale_q75 = np.percentile(all_scales, 75, axis=0)
        
        print(f"  - Feature columns: {self.FEATURE_COLUMNS}")
        print(f"  - Robust median mean: {robust_mean}")
        print(f"  - Mean Q25-Q75 range: {mean_q25} to {mean_q75}")
        print(f"  - Robust median scale: {robust_scale}")
        print(f"  - Scale Q25-Q75 range: {scale_q25} to {scale_q75}")
        
        # Check for outliers by computing how far each patient is from median
        print("\n  - Patient deviation analysis:")
        for i, (patient_id, scaler_params) in enumerate(self.scaler_info.items()):
            mean_diff = np.abs(scaler_params['mean'] - robust_mean)
            scale_diff = np.abs(scaler_params['scale'] - robust_scale)
            mean_max_diff = np.max(mean_diff)
            scale_max_diff = np.max(scale_diff)
            print(f"    {patient_id}: max_mean_diff={mean_max_diff:.3f}, max_scale_diff={scale_max_diff:.3f}")
        
        return robust_mean, robust_scale

    def _apply_scaler(self, df):
        """Apply robust scaler parameters to normalize the data."""
        print(f"  - Scaler info type: {type(self.scaler_info)}")
        
        # Handle per-patient scaler info (multiple scalers)
        if isinstance(self.scaler_info, dict) and all(isinstance(v, dict) for v in self.scaler_info.values()):
            print(f"  - Found per-patient scalers for {len(self.scaler_info)} patients")
            print("  - Computing robust scaler parameters...")
            
            # Use robust statistics instead of simple mean
            robust_mean, robust_scale = self._compute_robust_scaler_params()
            
            # Create scaler with robust parameters
            scaler = StandardScaler()
            scaler.mean_ = robust_mean
            scaler.scale_ = robust_scale
            scaler.var_ = robust_scale ** 2  # Approximate variance from scale
            scaler.n_features_in_ = len(self.FEATURE_COLUMNS)
            scaler.feature_names_in_ = np.array(self.FEATURE_COLUMNS)
            
        elif hasattr(self.scaler_info, 'transform'):
            print("  - Using saved scaler object directly")
            scaler = self.scaler_info
            
        else:
            raise ValueError(f"Unexpected scaler_info structure: {type(self.scaler_info)}")
        
        # Transform the data
        print("  - Applying robust normalization...")
        df[self.FEATURE_COLUMNS] = scaler.transform(df[self.FEATURE_COLUMNS])
        return df

    def process_patient_folder(self, patient_folder_path):
        respeck_folder = os.path.join(patient_folder_path, 'Respeck')
        if not os.path.isdir(respeck_folder):
            print(f"  - No 'Respeck' subfolder found for this patient.")
            return 0, []

        respeck_files = glob.glob(os.path.join(respeck_folder, '*.csv'))
        
        if not respeck_files:
            print(f"  - No '*.csv' files found in {respeck_folder}.")
            return 0, []

        # Load and concatenate all data for the patient
        all_patient_dfs = []
        for file_path in respeck_files:
            prepped_df = self._load_and_prep_single_file(file_path)
            if prepped_df is not None:
                all_patient_dfs.append(prepped_df)
        
        if not all_patient_dfs:
            print("  - No valid data could be loaded for this patient.")
            return 0, []
        
        full_df = pd.concat(all_patient_dfs, ignore_index=True)
        full_df.sort_values('timestamp_unix', inplace=True)
        
        # Preprocess the combined DataFrame
        full_df = self._filter_sleep_window(full_df)
        if full_df.empty:
            print("  - No data left after filtering for sleep window.")
            return 0, []
            
        # Handle missing values
        for col in self.FEATURE_COLUMNS:
            if full_df[col].isnull().sum() > 0:
                full_df[col].ffill(inplace=True)
                full_df[col].bfill(inplace=True)

        # Apply normalization using saved scaler parameters
        print("  - Applying normalization using saved scaler parameters...")
        full_df = self._apply_scaler(full_df)
        print("  - Normalization complete.")
        
        # Create windows from the fully preprocessed data
        windows_np, timestamps_ms = self._create_windows_with_timestamps(full_df)
        
        if len(windows_np) == 0:
            print("  - Not enough data to create windows from the sleep period.")
            return 0, []
            
        print(f"  - Created {len(windows_np)} windows for inference.")
        
        # Run inference
        predictions_np = self._run_inference_in_batches(windows_np)
        
        apnea_timestamps = []
        for i, pred in enumerate(predictions_np):
            if pred == self.APNEA_LABEL:
                apnea_timestamps.append(timestamps_ms[i])
                
        return len(apnea_timestamps), apnea_timestamps


def main():
    # Update these paths to match your new model files
    MODEL_PATH = "../results/cd2046784/final_sleep_apnea_model_20250718_200301.pt"  # Update timestamp
    CONFIG_PATH = "../results/cd2046784/model_config_20250718_200301.pkl"  # Update timestamp
    DATA_ROOT_FOLDER = "../data/Smile" 

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return

    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Configuration file not found at '{CONFIG_PATH}'")
        return

    if not os.path.isdir(DATA_ROOT_FOLDER):
        print(f"Error: Data folder not found at '{DATA_ROOT_FOLDER}'")
        return

    detector = ApneaDetector(MODEL_PATH, CONFIG_PATH)
    all_results = {}

    patient_folders = sorted([d for d in os.listdir(DATA_ROOT_FOLDER) if os.path.isdir(os.path.join(DATA_ROOT_FOLDER, d))])
    
    for patient_id in patient_folders:
        print(f"\n--- Processing Patient: {patient_id} ---")
        patient_path = os.path.join(DATA_ROOT_FOLDER, patient_id)
        
        apnea_count, apnea_timestamps = detector.process_patient_folder(patient_path)
        all_results[patient_id] = {
            'count': apnea_count,
            'timestamps_ms': apnea_timestamps
        }
        print(f"  - Found {apnea_count} '{detector.CLASS_NAMES[detector.APNEA_LABEL]}' events.")

    # Final Report and CSV Export
    print("\n" + "="*50)
    print("           FINAL APNEA DETECTION SUMMARY")
    print("="*50)

    for patient_id, result in all_results.items():
        print(f"\nPatient: {patient_id}")
        print(f"  Total '{detector.CLASS_NAMES[detector.APNEA_LABEL]}' events detected: {result['count']}")

        if result['count'] > 0:
            patient_prefix = patient_id.split(' ')[0]
            csv_filename = f"{patient_prefix}_osa.csv"

            timestamps_ms = result['timestamps_ms']
            dt_index = pd.to_datetime(timestamps_ms, unit='ms', utc=True).tz_convert('Europe/London')

            event_df = pd.DataFrame({
                'date': dt_index.strftime('%Y-%m-%d'),
                'time': dt_index.strftime('%H:%M:%S')
            })

            event_df.to_csv(csv_filename, index=False)
            print(f"  - Saved {result['count']} event timestamps to '{csv_filename}'")

if __name__ == "__main__":
    main()