# apply_to_smile.py

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
import pytz 

warnings.filterwarnings('ignore')

# --- Parameters from the training notebook (multiCNN.ipynb) ---
FEATURE_COLUMNS = [
    'breathingSignal', 'activityLevel', 'breathingRate', 'x', 'y', 'z'
]
WINDOW_SIZE = 375
STEP_SIZE = 75
LABEL_TO_EVENT_GROUP_NAME = {
    0: 'Normal',
    1: 'Obstructive Apnea',
    2: 'Hypopnea Events',
    3: 'Central/Mixed Apnea',
    4: 'Desaturation'
}
APNEA_LABEL = 1

# --- Model Definition (copied exactly from training notebook) ---
class OSA_CNN_MultiClass(nn.Module):
    def __init__(self, n_features, n_outputs, n_timesteps):
        super(OSA_CNN_MultiClass, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_features, n_timesteps)
            x = self.relu1(self.bn1(self.conv1(dummy_input)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.conv3(x))
            dummy_output = self.pool1(x)
            flattened_size = dummy_output.shape[1] * dummy_output.shape[2]
        self.fc1 = nn.Linear(flattened_size, 100)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(100, n_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.dropout1(self.relu3(self.conv3(x)))
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

class ApneaDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        print(f"Using device: {self.device}")
        self.scotland_tz = pytz.timezone('Europe/London')
        self.model = self._load_model()

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model = checkpoint['model_architecture']
        model.load_state_dict(checkpoint['model_state_dict'])
        
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
        # This means we keep hours that are 21 or greater OR less than 10.
        is_in_sleep_window = (hour >= 21) | (hour < 10)
        
        filtered_df = df[is_in_sleep_window].copy()
        
        print(f"  - Original data points: {len(df)}")
        print(f"  - Sleep window data points: {len(filtered_df)}")
        
        filtered_df.drop(columns=['datetime_scotland'], inplace=True)
        return filtered_df

    # <<< MODIFIED: This function now only loads and does initial prep. NO SCALING here.
    def _load_and_prep_single_file(self, file_path):
        """Loads and prepares a single Respeck CSV file without scaling."""
        try:
            # <<< FIX: Tell pandas to recognize 'Na' as a missing value upon loading.
            df = pd.read_csv(file_path, na_values=['Na'])
            
            df.rename(columns={'interpolatedPhoneTimestamp': 'timestamp_unix'}, inplace=True)
            
            # Ensure timestamp is numeric before any filtering
            df['timestamp_unix'] = pd.to_numeric(df['timestamp_unix'], errors='coerce')
            df.dropna(subset=['timestamp_unix'], inplace=True)
            df['timestamp_unix'] = df['timestamp_unix'].astype('int64')

            if not all(col in df.columns for col in FEATURE_COLUMNS):
                print(f"  - Missing one or more feature columns in {file_path}. Skipping.")
                return None
            
            # <<< NEW: After loading with na_values, ensure feature columns are numeric
            for col in FEATURE_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df
        except Exception as e:
            print(f"  - Error processing file {file_path}: {e}")
            return None
    
    def _create_windows_with_timestamps(self, df):
        windows, timestamps = [], []
        
        if len(df) < WINDOW_SIZE:
            return np.array([]), []

        for i in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
            window_df = df.iloc[i : i + WINDOW_SIZE]
            windows.append(window_df[FEATURE_COLUMNS].values)
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

    # <<< MODIFIED: Main logic refactored to ensure correct normalization strategy.
    def process_patient_folder(self, patient_folder_path):
        respeck_folder = os.path.join(patient_folder_path, 'Respeck')
        if not os.path.isdir(respeck_folder):
            print(f"  - No 'Respeck' subfolder found for this patient.")
            return 0, []

        respeck_files = glob.glob(os.path.join(respeck_folder, '*.csv'))
        
        if not respeck_files:
            print(f"  - No '*.csv' files found in {respeck_folder}.")
            return 0, []

        # --- STEP 1: Load and concatenate all data for the patient FIRST ---
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
        
        # --- STEP 2: Preprocess the COMBINED DataFrame ---
        full_df = self._filter_sleep_window(full_df)
        if full_df.empty:
            print("  - No data left after filtering for sleep window.")
            return 0, []
            
        for col in FEATURE_COLUMNS:
            if full_df[col].isnull().sum() > 0:
                full_df[col].ffill(inplace=True)
                full_df[col].bfill(inplace=True)

        # <<< CRITICAL FIX: Apply StandardScaler to the entire patient dataset at once.
        print("  - Applying per-patient normalization (the correct strategy)...")
        scaler = StandardScaler()
        full_df[FEATURE_COLUMNS] = scaler.fit_transform(full_df[FEATURE_COLUMNS])
        print("  - Normalization complete.")
        
        # --- STEP 3: Create windows from the fully preprocessed data ---
        windows_np, timestamps_ms = self._create_windows_with_timestamps(full_df)
        
        if len(windows_np) == 0:
            print("  - Not enough data to create windows from the sleep period.")
            return 0, []
            
        print(f"  - Created {len(windows_np)} windows for inference.")
        
        # --- STEP 4: Run inference ---
        predictions_np = self._run_inference_in_batches(windows_np)
        
        apnea_timestamps = []
        for i, pred in enumerate(predictions_np):
            if pred == APNEA_LABEL:
                apnea_timestamps.append(timestamps_ms[i])
                
        return len(apnea_timestamps), apnea_timestamps


def main():
    MODEL_PATH = "../Classification/cnn/osa_cnn_full_train.pt"
    DATA_ROOT_FOLDER = "../../data/Smile" 

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return

    if not os.path.isdir(DATA_ROOT_FOLDER):
        print(f"Error: Data folder not found at '{DATA_ROOT_FOLDER}'")
        return

    detector = ApneaDetector(MODEL_PATH)
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
        print(f"  - Found {apnea_count} '{LABEL_TO_EVENT_GROUP_NAME[APNEA_LABEL]}' events.")

    # --- Final Report and CSV Export ---
    print("\n" + "="*50)
    print("           FINAL APNEA DETECTION SUMMARY")
    print("="*50)

    for patient_id, result in all_results.items():
        print(f"\nPatient: {patient_id}")
        print(f"  Total '{LABEL_TO_EVENT_GROUP_NAME[APNEA_LABEL]}' events detected: {result['count']}")

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