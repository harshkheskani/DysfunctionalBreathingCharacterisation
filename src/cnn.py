import pandas as pd
import numpy as np
import glob
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
import time
import json
from sklearn.model_selection import ParameterGrid

# ===================================================================
# 1. DATA PREPARATION FOR CNN
# ===================================================================

def load_and_prep_data(data_dir):
    """
    Loads RAW data, aligns time using PSG files, applies a bandpass filter,
    and applies ground truth labels. It DOES NOT use pre-computed features.
    """
    print("--- Starting Data Loading and Preprocessing for CNN ---")
    data_paths = {
        'events': os.path.join(data_dir, 'event_exports'),
        'respeck': os.path.join(data_dir, 'respeck'),
        'nasal': os.path.join(data_dir, 'nasal_files')
    }
    APNEA_EVENT_LABELS = ['Obstructive Apnea']
    all_sessions_df_list = []

    event_files = glob.glob(os.path.join(data_paths['events'], '*_event_export.csv'))

    for event_file_path in event_files:
        session_id = os.path.basename(event_file_path).split('_event_export.csv')[0]
        respeck_file = os.path.join(data_paths['respeck'], f'{session_id}_respeck.csv')
        nasal_file = os.path.join(data_paths['nasal'], f'{session_id}_nasal.csv')

        if not all(os.path.exists(p) for p in [respeck_file, nasal_file]):
            print(f"  - WARNING: Skipping session '{session_id}'. A required file is missing.")
            continue
        
        print(f"  - Processing session: {session_id}")
        df_events = pd.read_csv(event_file_path, decimal=',')
        df_respeck = pd.read_csv(respeck_file)
        df_nasal = pd.read_csv(nasal_file)

        # Standardize timestamp columns to 'timestamp_unix'
        for df, col_name in [(df_events, 'UnixTimestamp'), (df_respeck, 'alignedTimestamp'), (df_nasal, 'UnixTimestamp')]:
            if col_name in df.columns:
                df.rename(columns={col_name: 'timestamp_unix'}, inplace=True)
                df['timestamp_unix'] = pd.to_numeric(df['timestamp_unix'], errors='coerce')
                df.dropna(subset=['timestamp_unix'], inplace=True)
                df['timestamp_unix'] = df['timestamp_unix'].astype('int64')

        # Find and trim to overlapping time range
        start_time = max(df_nasal['timestamp_unix'].min(), df_respeck['timestamp_unix'].min())
        end_time = min(df_nasal['timestamp_unix'].max(), df_respeck['timestamp_unix'].max())
        df_session = df_respeck[(df_respeck['timestamp_unix'] >= start_time) & (df_respeck['timestamp_unix'] <= end_time)].copy()

        if df_session.empty:
            continue

        # Impute and Filter raw breathing signal
        if df_session['breathingSignal'].isnull().sum() > 0:
            df_session['breathingSignal'] = df_session['breathingSignal'].ffill().bfill()
        
        df_session = df_session.sort_values('timestamp_unix').reset_index(drop=True)
        time_diffs_ms = df_session['timestamp_unix'].diff().median()
        if not (pd.isna(time_diffs_ms) or time_diffs_ms == 0):
            fs = 1000.0 / time_diffs_ms
            b, a = butter(2, [0.1 / (0.5 * fs), 1.5 / (0.5 * fs)], btype='band')
            df_session['breathingSignal'] = filtfilt(b, a, df_session['breathingSignal'])
        
        # Labeling
        df_session['Label'] = 0
        df_events['end_time_unix'] = df_events['timestamp_unix'] + (df_events['Duration'] * 1000).astype('int64')
        df_apnea_events = df_events[df_events['Event'].isin(APNEA_EVENT_LABELS)]

        for _, event in df_apnea_events.iterrows():
            mask = df_session['timestamp_unix'].between(event['timestamp_unix'], event['end_time_unix'])
            df_session.loc[mask, 'Label'] = 1
        
        df_session['SessionID'] = session_id
        all_sessions_df_list.append(df_session)

    final_df = pd.concat(all_sessions_df_list, ignore_index=True)
    # Impute all other columns that might have NaNs (like x, y, z)
    final_df.ffill(inplace=True)
    final_df.bfill(inplace=True)
    print(f"--- Data Preprocessing Complete. Final DataFrame shape: {final_df.shape} ---")
    return final_df

def create_windows(df, config):
    print("--- Starting Windowing Process for CNN ---")
    X, y, groups = [], [], []
    for session_id, session_df in df.groupby(config['session_id_col']):
        for i in range(0, len(session_df) - config['window_size'], config['step_size']):
            window_df = session_df.iloc[i : i + config['window_size']]
            X.append(window_df[config['feature_cols']].values)
            y.append(1 if window_df[config['label_col']].sum() > 0 else 0)
            groups.append(session_id)
    
    # Transpose for PyTorch Conv1D: (N, Timesteps, Features) -> (N, Features, Timesteps)
    X = np.asarray(X, dtype=np.float32).transpose(0, 2, 1)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    
    print(f"--- Windowing Complete. Shape of X: {X.shape} ---")
    return X, y, groups

class OSA_CNN(nn.Module):
    def __init__(self, n_features, n_outputs=2):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        
        # Input: 375 -> After pool1: 187 -> After pool2: 93
        flattened_size = 128 * 93
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(flattened_size, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_outputs))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)

def train_and_evaluate_fold(model, train_loader, val_loader, criterion, optimizer, device, epochs, output_dir, fold_id):
    """
    Trains a model for one fold, saves the best version, and returns
    the predictions from that best model on the validation set.
    """
    best_val_f1 = -1
    best_epoch_preds = []
    train_losses, val_losses, val_f1_scores = [], [], []

    for epoch in range(epochs):
        model.train()
        # ... (training loop for one epoch is the same) ...
        for inputs, labels in train_loader:
            # ...
            optimizer.step()
        
        # --- Validation after each epoch ---
        # The evaluate_model function is perfect for this part
        val_true_labels, val_preds = evaluate_model(model, val_loader, device)
        
        val_f1 = f1_score(val_true_labels, val_preds, pos_label=1, zero_division=0)
        print(f"  Epoch {epoch+1}/{epochs}, Val F1: {val_f1:.4f}")

        # If this epoch is the best so far, save the model and its predictions
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch_preds = val_preds # <-- STORE THE PREDICTIONS
            torch.save(model.state_dict(), os.path.join(output_dir, f'best_model_fold_{fold_id}.pth'))
            print(f"    -> New best model saved with F1-Score: {best_val_f1:.4f}")

    # Return the predictions from the best epoch
    return best_epoch_preds

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on a given dataset and returns true labels and predictions.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

# ===================================================================
# 3. MAIN EXECUTION SCRIPT
# ===================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    parser = argparse.ArgumentParser(description="CNN Training Pipeline for Apnea Detection.")
    parser.add_argument('--data_dir', type=str, default='../data/bishkek_csr/03_train_ready')
    parser.add_argument('--base_output_dir', type=str, default='./cnn_lono_results', help='Base directory to save all experiment results.')
    parser.add_argument('--output_dir', type=str, default='./cnn_lono_results')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define features for the CNN (raw signals)

    param_grid = {
        'lr': [0.001, 0.01, 0.005],
        'epochs': [50, 75],
        'dropout_rate': [0.3, 0.5],
        'batch_size': [32, 64, 128]
    }
    hyperparameter_grid = list(ParameterGrid(param_grid))
    
    config = {
        'window_size': 375, 'step_size': 75,
        'feature_cols': ['breathingSignal', 'activityLevel', 'x', 'y', 'z'],
        'label_col': 'Label', 'session_id_col': 'SessionID', 'random_state': 42
    }
    data_paths = {
        'events': os.path.join(args.data_dir, 'event_exports'),
        'respeck': os.path.join(args.data_dir, 'respeck'),
        'nasal': os.path.join(args.data_dir, 'nasal_files')
    }

    print(f"--- Starting Hyperparameter Search ---")
    print(f"Total number of experiments to run: {len(hyperparameter_grid)}")
    

    df = load_and_prep_data(args.data_dir)
    X, y, groups = create_windows(df, config)

    all_fold_preds, all_fold_true = [], []
    logo = LeaveOneGroupOut()

    results_summary = []
        
    for i, params in enumerate(hyperparameter_grid):
        run_id = f"run_{i+1}_lr{params['lr']}_ep{params['epochs']}_dr{params['dropout_rate']}_bs{params['batch_size']}"
        run_output_dir = os.path.join(args.base_output_dir, run_id)
        os.makedirs(run_output_dir, exist_ok=True)
        
        print(f"\n\n{'='*70}\nSTARTING EXPERIMENT RUN {i+1}/{len(hyperparameter_grid)}\n{'='*70}")
        print(f"Parameters: {params}")
        
        with open(os.path.join(run_output_dir, 'params.json'), 'w') as f:
            json.dump(params, f, indent=4)

        # --- LONO Cross-Validation for this set of parameters ---
        all_fold_preds, all_fold_true = [], []
        fold_metrics = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        logo = LeaveOneGroupOut()
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            test_night = np.unique(groups[test_idx])[0]
            print(f"\n--- FOLD {fold + 1}/{logo.get_n_splits(groups=groups)} --- Testing on: {test_night} ---")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            n_samples, n_features, n_timesteps = X_train.shape
            X_train_flat = X_train.reshape(n_samples, -1)
            smote = SMOTE(random_state=config['random_state'])
            X_train_res, y_train_res = smote.fit_resample(X_train_flat, y_train)
            X_train_res_cnn = X_train_res.reshape(-1, n_features, n_timesteps)

            train_dataset = TensorDataset(torch.from_numpy(X_train_res_cnn), torch.from_numpy(y_train_res))
            val_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

            model = OSA_CNN(n_features=n_features, dropout_rate=params['dropout_rate']).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            
            train_l, val_l, val_f1 = train_and_evaluate_fold(
                model, train_loader, val_loader, criterion, optimizer, device, 
                params['epochs'], run_output_dir, fold + 1
            )
            fold_metrics['train_loss'].append(train_l)
            fold_metrics['val_loss'].append(val_l)
            fold_metrics['val_f1'].append(val_f1)

            best_model_for_fold = OSA_CNN(n_features=n_features, dropout_rate=params['dropout_rate']).to(device)
            best_model_for_fold.load_state_dict(torch.load(os.path.join(run_output_dir, f'best_model_fold_{fold+1}.pth')))
            
            _, fold_preds = evaluate_model(best_model_for_fold, val_loader, device)
            all_fold_preds.extend(fold_preds)
            all_fold_true.extend(y_test)

    # --- Save all artifacts for THIS experimental run ---
    # 1. Plot and save learning curves for each fold
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    for f in range(len(fold_metrics['train_loss'])):
        axes[0].plot(fold_metrics['train_loss'][f], label=f'Fold {f+1}')
        axes[1].plot(fold_metrics['val_loss'][f], label=f'Fold {f+1}')
        axes[2].plot(fold_metrics['val_f1'][f], label=f'Fold {f+1}')
    axes[0].set_title('Training Loss per Epoch')
    axes[1].set_title('Validation Loss per Epoch')
    axes[2].set_title('Validation F1-Score per Epoch')
    axes[2].set_xlabel('Epoch')
    for ax in axes: ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, 'learning_curves.png'))
    plt.close()

    # 2. Save aggregated report and confusion matrix
    report = classification_report(all_fold_true, all_fold_preds, target_names=['Normal (0)', 'Apnea (1)'], output_dict=True, zero_division=0)
    with open(os.path.join(run_output_dir, 'final_classification_report.txt'), 'w') as f:
        f.write(classification_report(all_fold_true, all_fold_preds, target_names=['Normal (0)', 'Apnea (1)'], zero_division=0))
    # Final Aggregated Evaluation and Saving
    print(f"\n\n{'='*70}\nLONO CROSS-VALIDATION COMPLETE\n{'='*70}")
    
    cm = confusion_matrix(all_fold_true, all_fold_preds, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Normal', 'Apnea'], yticklabels=['Normal', 'Apnea'])
    plt.title('CNN - Aggregated Normalized Confusion Matrix (LONO)')
    plt.savefig(os.path.join(args.output_dir, 'final_cnn_confusion_matrix.png'))
    plt.close()

    results_summary.append({
        'run_id': run_id,
        'params': str(params),
        'f1_score_apnea': report.get('Apnea (1)', {}).get('f1-score', 0.0),
        'recall_apnea': report.get('Apnea (1)', {}).get('recall', 0.0),
        'precision_apnea': report.get('Apnea (1)', {}).get('precision', 0.0),
        'accuracy': report.get('accuracy', 0.0)
    })

    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(os.path.join(args.base_output_dir, 'master_experiment_summary.csv'), index=False)
    print(f"\n\n{'='*70}\nALL EXPERIMENTS COMPLETE\n{'='*70}")
    print(summary_df)

if __name__ == '__main__':
    main()
