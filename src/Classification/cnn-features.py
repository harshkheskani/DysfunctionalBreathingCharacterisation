# FILE: cnn-features.py (Final Corrected Version)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob
from collections import Counter
from scipy import stats
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. Model Definition (Copied from your training script - NO CHANGES)
# ==============================================================================
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

# ==============================================================================
# 2. Grad-CAM and Model Wrapper (Final Corrected Version)
# ==============================================================================
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class GradCAM1D(GradCAM):
    """ Custom GradCAM class to handle 1D Tensors. """
    def __init__(self, model, target_layers, reshape_transform=None):
        super(GradCAM1D, self).__init__(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    def get_cam_weights(self, input_tensor, target_layer, target_category, activations, grads):
        # For 1D conv layers, grads will be 4D after reshape_transform: (batch, channels, timesteps, 1)
        # We average over the spatial dimensions (timesteps and dummy height)
        return np.mean(grads, axis=(2, 3))

class CAMModelWrapper(nn.Module):
    """ Wrapper to handle 4D tensor from CAM library for a 3D-expecting model. """
    def __init__(self, model):
        super(CAMModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Input x is 4D: (batch, features, timesteps, dummy_height=1)
        # Squeeze the last dimension to make it 3D
        x_3d = x.squeeze(-1)
        # Permute it back to the (batch, timesteps, features) format the original model expects
        x_3d_permuted = x_3d.permute(0, 2, 1)
        return self.model(x_3d_permuted)

def reshape_transform_1d(tensor):
    """
    Reshape transform for 1D conv outputs to work with Grad-CAM.
    Converts 3D tensor (batch, channels, timesteps) to 4D (batch, channels, timesteps, 1)
    """
    if tensor.dim() == 3:
        # Add a dummy height dimension
        return tensor.unsqueeze(-1)
    elif tensor.dim() == 4:
        # Already in the right format
        return tensor
    else:
        raise ValueError(f"Unexpected tensor dimensions: {tensor.shape}")
# ==============================================================================
# 3. Data Loading and 4. Visualization Functions (No changes needed)
# ==============================================================================
def load_and_preprocess_data(data_dir, event_group_map, feature_columns, window_size, step_size):
    # This function is assumed to be correctly implemented as before
    print("--- 1. Loading and preprocessing all data ---")
    EVENTS_FOLDER = os.path.join(data_dir, 'event_exports')
    RESPECK_FOLDER = os.path.join(data_dir, 'respeck')
    all_sessions_df_list = []
    event_files = glob.glob(os.path.join(EVENTS_FOLDER, '*_event_export.csv'))
    for event_file_path in event_files:
        base_name = os.path.basename(event_file_path)
        session_id = base_name.split('_event_export.csv')[0]
        respeck_file_path = os.path.join(RESPECK_FOLDER, f'{session_id}_respeck.csv')
        if not os.path.exists(respeck_file_path): continue
        df_events = pd.read_csv(event_file_path, decimal=',')
        df_respeck = pd.read_csv(respeck_file_path)
        for df_, ts_col in [(df_events, 'UnixTimestamp'), (df_respeck, 'alignedTimestamp')]:
            df_.rename(columns={ts_col: 'timestamp_unix'}, inplace=True)
            df_['timestamp_unix'] = pd.to_numeric(df_['timestamp_unix'], errors='coerce')
            df_.dropna(subset=['timestamp_unix'], inplace=True)
            df_['timestamp_unix'] = df_['timestamp_unix'].astype('int64')
        df_respeck['Label'] = 0
        df_events['Duration_ms'] = (df_events['Duration'] * 1000).astype('int64')
        df_events['end_time_unix'] = df_events['timestamp_unix'] + df_events['Duration_ms']
        for label_id, event_names_in_group in event_group_map.items():
            df_filtered_events = df_events[df_events['Event'].isin(event_names_in_group)]
            for _, event in df_filtered_events.iterrows():
                df_respeck.loc[df_respeck['timestamp_unix'].between(event['timestamp_unix'], event['end_time_unix']), 'Label'] = label_id
        df_respeck['SessionID'] = session_id
        all_sessions_df_list.append(df_respeck)
    if not all_sessions_df_list: raise ValueError("No data loaded.")
    df = pd.concat(all_sessions_df_list, ignore_index=True)
    for col in feature_columns:
        if df[col].isnull().any(): df[col].ffill(inplace=True); df[col].bfill(inplace=True)
    df_normalized = df.copy()
    for session_id in df['SessionID'].unique():
        session_mask = df['SessionID'] == session_id
        scaler = StandardScaler()
        df_normalized.loc[session_mask, feature_columns] = scaler.fit_transform(df.loc[session_mask, feature_columns])
    X, y, groups = [], [], []
    for session_id, session_df in df_normalized.groupby('SessionID'):
        for i in range(0, len(session_df) - window_size, step_size):
            window_df = session_df.iloc[i : i + window_size]
            X.append(window_df[feature_columns].values)
            y.append(stats.mode(window_df['Label'])[0])
            groups.append(session_id)
    print("--- Data loading and windowing complete. ---")
    return np.asarray(X), np.asarray(y), np.asarray(groups)

def visualize_grad_cam_on_signal(signal_window, cam_heatmap, title=""):
    """
    Fixed visualization function with robust error handling
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Debug prints
    print(f"Debug - signal_window shape: {signal_window.shape}")
    print(f"Debug - signal_window type: {type(signal_window)}")
    print(f"Debug - cam_heatmap shape: {cam_heatmap.shape}")
    print(f"Debug - cam_heatmap type: {type(cam_heatmap)}")
    print(f"Debug - cam_heatmap dtype: {cam_heatmap.dtype}")
    
    # Ensure cam_heatmap is a proper 1D numpy array
    cam_heatmap = np.asarray(cam_heatmap)
    if cam_heatmap.ndim > 1:
        cam_heatmap = cam_heatmap.flatten()
    
    # Ensure signal_window is a proper numpy array
    signal_window = np.asarray(signal_window)
    
    # Extract breathing signal (first column)
    breathing_signal = signal_window[:, 0]
    timesteps = np.arange(len(breathing_signal))
    
    print(f"Debug - breathing_signal length: {len(breathing_signal)}")
    print(f"Debug - cam_heatmap length: {len(cam_heatmap)}")
    print(f"Debug - timesteps length: {len(timesteps)}")
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(16, 5))
    
    # Plot breathing signal
    color = 'tab:blue'
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Normalized Breathing Signal', color=color)
    ax1.plot(timesteps, breathing_signal, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Plot CAM heatmap
    ax2 = ax1.twinx()
    
    # Resize CAM to match signal length if needed
    if len(cam_heatmap) != len(timesteps):
        print(f"Debug - Resizing CAM from {len(cam_heatmap)} to {len(timesteps)}")
        
        # Create coordinate arrays for interpolation
        x_old = np.linspace(0, 1, len(cam_heatmap))
        x_new = np.linspace(0, 1, len(timesteps))
        
        # Ensure arrays are 1D and contain only finite values
        x_old = np.asarray(x_old).flatten()
        x_new = np.asarray(x_new).flatten()
        cam_heatmap_clean = np.asarray(cam_heatmap).flatten()
        
        # Check for any issues with the arrays
        print(f"Debug - x_old shape: {x_old.shape}, finite: {np.isfinite(x_old).all()}")
        print(f"Debug - x_new shape: {x_new.shape}, finite: {np.isfinite(x_new).all()}")
        print(f"Debug - cam_heatmap_clean shape: {cam_heatmap_clean.shape}, finite: {np.isfinite(cam_heatmap_clean).all()}")
        
        # Replace any non-finite values
        if not np.isfinite(cam_heatmap_clean).all():
            print("Warning: Non-finite values in CAM heatmap, replacing with zeros")
            cam_heatmap_clean = np.nan_to_num(cam_heatmap_clean, nan=0.0, posinf=1.0, neginf=0.0)
        
        try:
            resized_cam = np.interp(x_new, x_old, cam_heatmap_clean)
        except Exception as e:
            print(f"Interpolation failed: {e}")
            print("Falling back to simple repetition/truncation")
            if len(cam_heatmap_clean) > len(timesteps):
                resized_cam = cam_heatmap_clean[:len(timesteps)]
            else:
                # Repeat the pattern to match length
                repeat_factor = len(timesteps) // len(cam_heatmap_clean) + 1
                resized_cam = np.tile(cam_heatmap_clean, repeat_factor)[:len(timesteps)]
    else:
        resized_cam = cam_heatmap
    
    print(f"Debug - resized_cam shape: {resized_cam.shape}")
    print(f"Debug - resized_cam min/max: {resized_cam.min():.4f}/{resized_cam.max():.4f}")
    
    # Plot the heatmap
    color = 'tab:red'
    ax2.set_ylabel('Model Attention (Activation)', color=color)
    ax2.fill_between(timesteps, 0, resized_cam, color=color, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(1, resized_cam.max() * 1.1) if len(resized_cam) > 0 else 1)
    
    fig.tight_layout()
    plt.title(title, fontsize=16)
    plt.show()

# ==============================================================================
# 5. Main Analysis Block
# ==============================================================================
def main(args):
    # --- Configuration ---
    EVENT_GROUP_TO_LABEL = {
        1: ['Obstructive Apnea'], 2: ['Hypopnea', 'Central Hypopnea', 'Obstructive Hypopnea'],
        3: ['Central Apnea', 'Mixed Apnea'], 4: ['RERA'], 5: ['Desaturation']
    }
    LABEL_TO_EVENT_GROUP_NAME = {
        0: 'Normal', 1: 'Obstructive Apnea', 2: 'Hypopnea Events',
        3: 'Central/Mixed Apnea', 4: 'RERA', 5: 'Desaturation'
    }
    FEATURE_COLUMNS = ['breathingSignal', 'breathingRate']
    WINDOW_SIZE = 375
    STEP_SIZE = int(WINDOW_SIZE * 0.2)
    N_OUTPUTS = len(EVENT_GROUP_TO_LABEL) + 1
    N_FEATURES = len(FEATURE_COLUMNS)

    # --- Load Data and Isolate Test Set ---
    X, y, groups = load_and_preprocess_data(args.data_dir, EVENT_GROUP_TO_LABEL, FEATURE_COLUMNS, WINDOW_SIZE, STEP_SIZE)
    unique_groups = sorted(list(np.unique(groups)))
    if args.fold >= len(unique_groups): raise ValueError(f"Fold {args.fold} is out of bounds.")
    test_session_id = unique_groups[args.fold]
    test_mask = groups == test_session_id
    X_test, y_test = X[test_mask], y[test_mask]
    
    # --- Load Model ---
    device = torch.device("cpu") # Forcing CPU for simplicity, can be changed
    original_model = OSA_CNN_MultiClass(n_features=N_FEATURES, n_outputs=N_OUTPUTS, n_timesteps=WINDOW_SIZE)
    model_path = os.path.join(args.model_dir, f'lono_checkpoint_cnn_{args.fold}.pt')
    try:
        original_model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError: print(f"ERROR: Model checkpoint not found at '{model_path}'"); return
    original_model.to(device)
    original_model.eval()

    # --- Find Sample ---
    class_to_explain_name = LABEL_TO_EVENT_GROUP_NAME[args.class_id]
    try:
        sample_idx = np.where(y_test == args.class_id)[0][0]
    except IndexError: print(f"ERROR: No samples of class '{class_to_explain_name}' found."); return
    input_sample = X_test[sample_idx]
    input_tensor = torch.from_numpy(input_sample).float().unsqueeze(0).to(device)

    # --- Run Prediction ---
    with torch.no_grad():
        output = original_model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_name = LABEL_TO_EVENT_GROUP_NAME[predicted_idx.item()]
    print(f"Analyzing sample. True Class: '{class_to_explain_name}', Model Prediction: '{predicted_name}'")

    # --- Run Grad-CAM ---
    cam_model = CAMModelWrapper(original_model)
    target_layers = [cam_model.model.conv3]
    targets_for_gradcam = [ClassifierOutputTarget(predicted_idx.item())]
    
    cam_model = CAMModelWrapper(original_model)
    target_layers = [cam_model.model.conv3]
    targets_for_gradcam = [ClassifierOutputTarget(predicted_idx.item())]
        
        # Instantiate Grad-CAM with reshape transform
    cam = GradCAM1D(
        model=cam_model, 
        target_layers=target_layers,
        reshape_transform=reshape_transform_1d
    )


    # Prepare the input tensor for CAM
    input_tensor_cam = input_tensor.permute(0, 2, 1)  # (1, features, timesteps)
    input_tensor_cam_4d = input_tensor_cam.unsqueeze(-1)  # (1, features, timesteps, 1)

    # Generate Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor_cam_4d, targets=targets_for_gradcam)

    # Debug: Print shape and type of grayscale_cam
    print(f"Debug - grayscale_cam shape: {grayscale_cam.shape}")
    print(f"Debug - grayscale_cam type: {type(grayscale_cam)}")

    # Ensure we have a 1D array
    if grayscale_cam.ndim > 1:
        grayscale_cam = grayscale_cam.squeeze()

    # If still multidimensional, take the first sample
    if grayscale_cam.ndim > 1:
        grayscale_cam = grayscale_cam[0]

    # Ensure it's a numpy array
    if hasattr(grayscale_cam, 'numpy'):
        grayscale_cam = grayscale_cam.numpy()

    print(f"Debug - final grayscale_cam shape: {grayscale_cam.shape}")
    # Define a reshape_transform for the output of the conv layer, not the input
    def reshape_transform_1d_tonormal(x):
        """
        Reshape transform for 1D conv outputs to work with Grad-CAM.
        Handles both 3D and 4D tensors appropriately.
        """
        if x.dim() == 3:
            # Shape: (batch, channels, timesteps) -> (batch, channels, timesteps, 1)
            return x.unsqueeze(-1)
        elif x.dim() == 4:
            # Already in the right format
            return x
        else:
            raise ValueError(f"Unexpected tensor dimensions: {x.shape}")

    # Prepare the 4D input tensor for the CAM library
    input_tensor_cam = input_tensor.permute(0, 2, 1) # Shape: (1, 2, 375)
    input_tensor_cam_reshaped = input_tensor_cam.unsqueeze(-1) # Shape: (1, 2, 375, 1)

    # The library will now pass the 4D tensor to our wrapper, which will fix it
    grayscale_cam = cam(input_tensor=input_tensor_cam_reshaped, targets=targets_for_gradcam)
    grayscale_cam = grayscale_cam[0, :]
    
    # --- Visualize ---
    title = f"Grad-CAM for '{predicted_name}' Prediction\n(True Class was '{class_to_explain_name}')"
    visualize_grad_cam_on_signal(input_sample, grayscale_cam, title=title)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CNN attention using Grad-CAM for 1D signals.")
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory of the data.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where saved model checkpoints are located.')
    parser.add_argument('--fold', type=int, required=True, help='The cross-validation fold number to analyze.')
    parser.add_argument('--class_id', type=int, required=True, help='The integer ID of the class to explain.')
    args = parser.parse_args()
    main(args)