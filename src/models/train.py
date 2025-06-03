import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os 

# --- Configuration ---
BASE_PATH_RESPECK = '../../data/bishkek_csr/02_prepped/respeck'
BASE_PATH_EVENTS = '../../data/bishkek_csr/02_prepped/event_exports'
BASE_PATH_NASAL = '../../data/bishkek_csr/02_prepped/nasal_files'
DATES = [
    "04-04-2025", "05-04-2025", "08-05-2025", "10-05-2025",
    "11-05-2025", "16-04-2025", "24-04-2025", "25-04-2025", "26-04-2025"
]

RESPECK_TIMESTAMP_COL = 'interpolatedTimestamp' 
EVENT_TIMESTAMP_COL = 'UnixTimestamp'
EVENT_NAME_COL = 'Event'                
EVENT_DURATION_COL = 'Duration'

######### DEFINE THE PUTO FEATURES ######################
FEATURES_TO_USE = ['x', 'y', 'z', 'breathingSignal', 'breathingRate']
NUM_FEATURES = len(FEATURES_TO_USE)
SAMPLING_RATE_HZ = 12.5 
WINDOW_SIZE_S = 15
STRIDE_S = 5
WINDOW_LENGTH_SAMPLES = int(WINDOW_SIZE_S * SAMPLING_RATE_HZ)
STRIDE_SAMPLES = int(STRIDE_S * SAMPLING_RATE_HZ)

# --- Helper Functions ---
def parse_duration(duration_str):
    try:
        return float(str(duration_str).replace(',', '.'))
    except ValueError:
        print(f"Warning: Could not parse duration '{duration_str}'. Returning 0.")
        return 0.0

# --- Data Aggregation Loop ---
master_windows_list = []
master_labels_list = []

print("Processing files...")
for date_str in DATES:
    respeck_file = os.path.join(BASE_PATH_RESPECK, f"{date_str}_respeck.csv")
    event_file = os.path.join(BASE_PATH_EVENTS, f"{date_str}_event_export.csv")

    print(f"  Loading data for {date_str}...")
    try:
        respeck_df = pd.read_csv(respeck_file)
        classification_df = pd.read_csv(event_file)
    except FileNotFoundError:
        print(f"    Warning: Files for date {date_str} not found. Skipping.")
        continue
    
    # Basic validation of required columns
    if not all(col in respeck_df.columns for col in FEATURES_TO_USE + [RESPECK_TIMESTAMP_COL]):
        print(f"    Warning: Missing required columns in {respeck_file}. Skipping.")
        continue
    if not all(col in classification_df.columns for col in [EVENT_NAME_COL, EVENT_TIMESTAMP_COL, EVENT_DURATION_COL]):
        print(f"    Warning: Missing required columns in {event_file}. Skipping.")
        continue

    # 1. Preprocess Data (per session)
    # Ensure timestamps are numeric. Respeck timestamps assumed to be in ms.
    # Event timestamps assumed to be in seconds (convert to ms).
    try:
        respeck_df[RESPECK_TIMESTAMP_COL] = pd.to_numeric(respeck_df[RESPECK_TIMESTAMP_COL])
        classification_df[EVENT_TIMESTAMP_COL] = pd.to_numeric(classification_df[EVENT_TIMESTAMP_COL]) * 1000 # s to ms
        classification_df['Duration_ms'] = classification_df[EVENT_DURATION_COL].apply(parse_duration) * 1000 # s to ms
        classification_df['EventEndTimestamp_ms'] = classification_df[EVENT_TIMESTAMP_COL] + classification_df['Duration_ms']
    except Exception as e:
        print(f"    Error processing timestamps/durations for {date_str}: {e}. Skipping.")
        continue
        
    # 2. Data Alignment and Windowing (per session, using RAW features)
    session_windows = []
    session_labels = []
    respeck_df['covered_by_event'] = False

    for _, event_row in classification_df.iterrows():
        event_start_ms = event_row[EVENT_TIMESTAMP_COL]
        event_end_ms = event_row['EventEndTimestamp_ms']
        event_label = event_row[EVENT_NAME_COL]

        event_respeck_data = respeck_df[
            (respeck_df[RESPECK_TIMESTAMP_COL] >= event_start_ms) &
            (respeck_df[RESPECK_TIMESTAMP_COL] < event_end_ms)
        ]
        if not event_respeck_data.empty:
            respeck_df.loc[event_respeck_data.index, 'covered_by_event'] = True
        
        data_segment = event_respeck_data[FEATURES_TO_USE].values
        for i in range(0, len(data_segment) - WINDOW_LENGTH_SAMPLES + 1, STRIDE_SAMPLES):
            window = data_segment[i : i + WINDOW_LENGTH_SAMPLES]
            if window.shape[0] == WINDOW_LENGTH_SAMPLES:
                session_windows.append(window)
                session_labels.append(event_label)
    
    # Create 'Normal' windows
    start_idx = 0
    while start_idx < len(respeck_df):
        while start_idx < len(respeck_df) and respeck_df.iloc[start_idx]['covered_by_event']:
            start_idx += 1
        if start_idx >= len(respeck_df): break
        end_idx = start_idx
        while end_idx < len(respeck_df) and not respeck_df.iloc[end_idx]['covered_by_event']:
            end_idx += 1
        
        normal_segment_data = respeck_df.iloc[start_idx:end_idx][FEATURES_TO_USE].values
        for i in range(0, len(normal_segment_data) - WINDOW_LENGTH_SAMPLES + 1, STRIDE_SAMPLES):
            window = normal_segment_data[i : i + WINDOW_LENGTH_SAMPLES]
            if window.shape[0] == WINDOW_LENGTH_SAMPLES:
                session_windows.append(window)
                session_labels.append('Normal') # Add 'Normal' label
        start_idx = end_idx
        
    master_windows_list.extend(session_windows)
    master_labels_list.extend(session_labels)
    print(f"    Processed {date_str}: Found {len(session_windows)} windows.")

if not master_windows_list:
    print("No windows were generated from any files. Check file paths, column names, and data content. Exiting.")
    exit()

# --- Post-Aggregation Processing ---
X_all_raw = np.array(master_windows_list, dtype=np.float32)
y_all_labels_str = np.array(master_labels_list)

print(f"\nTotal windows aggregated: {X_all_raw.shape[0]}")
if X_all_raw.shape[0] == 0:
    print("No data to train on. Exiting.")
    exit()

# Split data BEFORE scaling
X_train_raw, X_test_raw, y_train_labels_str, y_test_labels_str = train_test_split(
    X_all_raw, y_all_labels_str, test_size=0.2, random_state=42, stratify=y_all_labels_str
)

X_train_scaled = np.zeros_like(X_train_raw)
X_test_scaled = np.zeros_like(X_test_raw)
scalers = {} # Store one scaler per feature

for i in range(NUM_FEATURES):
    scalers[i] = StandardScaler()
    feature_data_train = X_train_raw[:, :, i].reshape(-1, 1)
    scalers[i].fit(feature_data_train)
    X_train_scaled[:, :, i] = scalers[i].transform(X_train_raw[:, :, i].reshape(-1,1)).reshape(X_train_raw.shape[0], X_train_raw.shape[1])
    X_test_scaled[:, :, i] = scalers[i].transform(X_test_raw[:, :, i].reshape(-1,1)).reshape(X_test_raw.shape[0], X_test_raw.shape[1])

# Encode labels to integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_labels_str)
y_test_encoded = label_encoder.transform(y_test_labels_str) # Use transform only for test
NUM_CLASSES = len(label_encoder.classes_)

# PyTorch Conv1D expects (batch, channels/features, length)
X_train_pytorch = X_train_scaled.transpose(0, 2, 1)
X_test_pytorch = X_test_scaled.transpose(0, 2, 1)

# Torch ->>> integraed 1D cnn
class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = TimeSeriesDataset(X_train_pytorch, y_train_encoded)
test_dataset = TimeSeriesDataset(X_test_pytorch, y_test_encoded)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- PyTorch CNN Model (same as before) ---
class CNN1D(nn.Module):
    def __init__(self, num_features, num_classes, window_length):
        super(CNN1D, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=10), nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=10), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.3)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5), nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.3)
        )
        # Calculate flattened size dynamically
        with torch.no_grad():
            self._dummy_input = torch.randn(1, num_features, window_length)
            self._flattened_size = self._get_flattened_size()

        self.fc_block = nn.Sequential(
            nn.Flatten(), nn.Linear(self._flattened_size, 100), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(100, num_classes)
        )
    
    def _get_flattened_size(self):
        x = self.conv_block1(self._dummy_input)
        x = self.conv_block2(x)
        return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_block(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN1D(NUM_FEATURES, NUM_CLASSES, WINDOW_LENGTH_SAMPLES).to(device)

print(f"\nModel loaded on: {device}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Class mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
print(model)

# --- Loss, Optimizer, Training, Evaluation (same as before) ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=20): # Consider making num_epochs a parameter
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0; correct_predictions = 0; total_predictions = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0; correct_predictions = 0; total_predictions = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct_predictions / total_predictions
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")






# --- Run Training and Evaluation ---
if len(train_dataset) > 0 and len(test_dataset) > 0:
    print("\nStarting model training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=20) # Adjust num_epochs
    print("\nEvaluating model...")
    evaluate_model(model, test_loader, criterion)
else:
    print("\nNot enough data to train/test the model after processing all files.")