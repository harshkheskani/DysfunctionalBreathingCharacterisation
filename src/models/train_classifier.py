# train_classifier.py (Complete, Standalone Version)

import argparse
import logging
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import glob # Needed for load_process_data

# =======================================================================================
# --- BLOCK 1: ADDED - Generator Class Definition ---
# This must be the EXACT same architecture as the one saved in generator.pth
# =======================================================================================
class Generator(nn.Module):
    def __init__(self, latent_dim, seq_len, n_features):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (1, 256)),
        )
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.2)
        self.out = nn.Sequential(
            nn.Linear(256, n_features),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.model(z).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(x)
        return self.out(lstm_out)

# =======================================================================================
# --- BLOCK 2: ADDED - Data Loading and Processing Function ---
# This is a direct copy from the finalized train_gan.py script.
# =======================================================================================
def load_process_data(data_path):
    """
    Loads all session data, performs one-hot encoding, windows the data,
    and returns X, y, groups, and the final feature names.
    """
    logging.info("Loading and processing data...")

    all_sessions_df_list = []
    EVENTS_FOLDER = os.path.join(data_path, '03_train_ready/event_exports')
    RESPECK_FOLDER = os.path.join(data_path, '03_train_ready/respeck')
    NASAL_FOLDER = os.path.join(data_path, '03_train_ready/nasal_files')
    FEATURES_FOLDER = os.path.join(data_path, '03_train_ready/respeck_features')
    APNEA_EVENTS = ['Obstructive Apnea']

    event_files = glob.glob(os.path.join(EVENTS_FOLDER, '*_event_export.csv'))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {EVENTS_FOLDER}")
    
    for event_file_path in event_files:
        base_name = os.path.basename(event_file_path)
        session_id = base_name.split('_event_export.csv')[0]
        respeck_file_path = os.path.join(RESPECK_FOLDER, f'{session_id}_respeck.csv')
        nasal_file_path = os.path.join(NASAL_FOLDER, f'{session_id}_nasal.csv')
        feature_file_path = os.path.join(FEATURES_FOLDER, f'{session_id}_respeck_features.csv')
        
        if not all(os.path.exists(p) for p in [respeck_file_path, nasal_file_path, feature_file_path]):
            continue
        
        df_events = pd.read_csv(event_file_path, decimal=',')
        df_nasal = pd.read_csv(nasal_file_path)
        df_respeck = pd.read_csv(respeck_file_path)
        df_features = pd.read_csv(feature_file_path)

        df_events.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True)
        df_nasal.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True, errors='ignore')
        df_respeck.rename(columns={'alignedTimestamp': 'timestamp_unix'}, inplace=True)
        
        df_features['timestamp_unix'] = pd.to_datetime(df_features['startTimestamp'], format="mixed").astype('int64') // 10**6
        df_features['timestamp_unix_end'] = pd.to_datetime(df_features['endTimestamp'], format="mixed").astype('int64') // 10**6
        
        for df_ in [df_events, df_nasal, df_respeck]:
            df_['timestamp_unix'] = pd.to_numeric(df_['timestamp_unix'], errors='coerce')
            df_.dropna(subset=['timestamp_unix'], inplace=True)
            df_['timestamp_unix'] = df_['timestamp_unix'].astype('int64')

        start_time = max(df_nasal['timestamp_unix'].min(), df_respeck['timestamp_unix'].min())
        end_time = min(df_nasal['timestamp_unix'].max(), df_respeck['timestamp_unix'].max())
        
        df_respeck = df_respeck[(df_respeck['timestamp_unix'] >= start_time) & (df_respeck['timestamp_unix'] <= end_time)].copy()

        if df_respeck.empty:
            continue

        df_respeck = df_respeck.sort_values('timestamp_unix')
        df_features = df_features.sort_values('timestamp_unix')

        df_session_merged = pd.merge_asof(df_respeck, df_features, on='timestamp_unix', direction='backward')
        
        cols_to_drop = ['Unnamed: 0', 'startTimestamp', 'endTimestamp', 'timestamp_unix_end', 'type']
        df_session_merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        if df_session_merged.empty:
            continue
            
        df_session_merged['Label'] = 0
        df_events['Duration_ms'] = (df_events['Duration'] * 1000).astype('int64')
        df_events['end_time_unix'] = df_events['timestamp_unix'] + df_events['Duration_ms']
        df_apnea_events = df_events[df_events['Event'].isin(APNEA_EVENTS)].copy()

        for _, event in df_apnea_events.iterrows():
            start_event = event['timestamp_unix']
            end_event = event['end_time_unix']
            df_session_merged.loc[df_session_merged['timestamp_unix'].between(start_event, end_event), 'Label'] = 1

        df_session_merged['SessionID'] = session_id
        all_sessions_df_list.append(df_session_merged)

    df = pd.concat(all_sessions_df_list, ignore_index=True)
    
    NUMERICAL_FEATURES = [
        'breathingSignal', 'activityLevel', 'breathingRate', 'area', 'extremas', 
        'meanActivityLevel', 'peakRespiratoryFlow', 'duration', 'BR_md', 'BR_mean', 
        'BR_std', 'AL_md', 'AL_mean', 'AL_std', 'RRV', 'RRV3MA', 'breath_regularity'
    ]
    CATEGORICAL_FEATURES = ['activityType']
    
    if CATEGORICAL_FEATURES[0] in df.columns:
        df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix='activity')
        
    one_hot_cols = [col for col in df.columns if col.startswith('activity_')]
    FINAL_FEATURE_COLUMNS = NUMERICAL_FEATURES + one_hot_cols

    df[FINAL_FEATURE_COLUMNS] = df[FINAL_FEATURE_COLUMNS].ffill().bfill()
        
    SAMPLING_RATE_HZ = 12.5
    WINDOW_DURATION_SEC = 30
    WINDOW_SIZE = int(WINDOW_DURATION_SEC * SAMPLING_RATE_HZ)
    OVERLAP_PERCENTAGE = 0.80
    STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENTAGE))

    X, y, groups = [], [], []
    for session_id, session_df in df.groupby('SessionID'):
        for i in range(0, len(session_df) - WINDOW_SIZE, STEP_SIZE):
            window_df = session_df.iloc[i : i + WINDOW_SIZE]
            if not all(col in window_df.columns for col in FINAL_FEATURE_COLUMNS):
                continue
            features = window_df[FINAL_FEATURE_COLUMNS].values
            label = 1 if window_df['Label'].sum() > 0 else 0
            
            X.append(features)
            y.append(label)
            groups.append(session_id)
            
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    
    return X, y, groups, FINAL_FEATURE_COLUMNS

# =======================================================================================
# --- The rest of your script is already correct ---
# =======================================================================================

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_synthetic_data(generator, scaler, n_samples, latent_dim, device, seq_len, n_features):
    """
    Generate synthetic data using the GAN generator.
    """
    generator.eval()
    
    z = torch.randn(n_samples, latent_dim).to(device)
    with torch.no_grad():
        synthetic_data_scaled = generator(z).cpu().numpy()
        
    # Inverse transform to get data back to the original scale
    synthetic_data_unscaled = scaler.inverse_transform(
        synthetic_data_scaled.reshape(-1, n_features)
    ).reshape(n_samples, seq_len, n_features)
    
    # Labels are all 1 (apnea)
    y_synthetic = np.ones(n_samples)
    
    return synthetic_data_unscaled, y_synthetic

# train and eval
def main(args):
    # --- Setup ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Full Dataset ---
    X, y, groups, feature_names = load_process_data(args.data_path)
    seq_len, n_features = X.shape[1], X.shape[2]

    # --- Load Pre-trained GAN and Scaler ---
    logging.info("Loading pre-trained GAN generator and scaler...")
    
    # Load scaler
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Initialize Generator model and load its state
    generator = Generator(args.latent_dim, seq_len, n_features).to(device)
    generator.load_state_dict(torch.load(args.generator_path, map_location=device))
    logging.info("Generator and scaler loaded successfully.")

    # --- Setup Cross-Validation ---
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    logging.info(f"Starting Leave-One-Group-Out cross-validation with {n_folds} folds.")
    
    all_preds = []
    all_true = []

    # --- Cross-Validation Loop ---
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_session = np.unique(groups[test_idx])[0]
        logging.info(f"\n--- FOLD {fold + 1}/{n_folds} | Testing on Session: {test_session} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Data Augmentation with GAN ---
        logging.info("Performing data augmentation with GAN...")
        
        X_train_normal = X_train[y_train == 0]
        X_train_apnea = X_train[y_train == 1]
        
        n_normal = len(X_train_normal)
        n_apnea_original = len(X_train_apnea)
        n_to_generate = n_normal - n_apnea_original

        if n_to_generate > 0:
            logging.info(f"Original balance: {n_normal} normal vs. {n_apnea_original} apnea. Generating {n_to_generate} synthetic samples.")
            X_synthetic, y_synthetic = generate_synthetic_data(generator, scaler, n_to_generate, args.latent_dim, device, seq_len, n_features)
            
            X_train_apnea_aug = np.concatenate([X_train_apnea, X_synthetic])
            y_train_apnea_aug = np.concatenate([y_train[y_train == 1], y_synthetic])
        else:
            logging.info("Dataset is already balanced or has more apnea samples. Not generating synthetic data for this fold.")
            X_train_apnea_aug = X_train_apnea
            y_train_apnea_aug = y_train[y_train == 1]
        
        X_train_balanced = np.concatenate([X_train_normal, X_train_apnea_aug])
        y_train_balanced = np.concatenate([y_train[y_train == 0], y_train_apnea_aug])
        
        X_train_flat = X_train_balanced.reshape(X_train_balanced.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        X_train_final, y_train_final = shuffle(X_train_flat, y_train_balanced, random_state=42)
        
        logging.info(f"Final training set size for this fold: {len(y_train_final)}. Distribution: {Counter(y_train_final)}")

        # --- Train XGBoost Classifier ---
        logging.info("Training XGBoost model...")
        xgb_classifier = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=400,
            learning_rate=0.01,
            max_depth=5,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_classifier.fit(X_train_final, y_train_final)
        
        # --- Evaluate and Store Results ---
        logging.info("Evaluating on the test fold...")
        fold_preds = xgb_classifier.predict(X_test_flat)
        all_preds.extend(fold_preds)
        all_true.extend(y_test)

    # --- Final Aggregated Evaluation ---
    logging.info("\n====================================================")
    logging.info("Cross-Validation Complete. Aggregated Results:")
    logging.info("====================================================")
    
    class_names = ['Normal (0)', 'Apnea (1)']
    report = classification_report(all_true, all_preds, target_names=class_names)
    logging.info(f"\nClassification Report:\n{report}")
    
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    cm = confusion_matrix(all_true, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title('XGBoost - Aggregated Normalized Confusion Matrix (GAN Augmented)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()
    
    logging.info(f"Results and plots saved to '{args.output_dir}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Classifier with GAN-Augmented Data")
    
    parser.add_argument('--data_path', type=str, required=True, help='Base path to the data directory (e.g., ../../data/bishkek_csr/)')
    parser.add_argument('--generator_path', type=str, required=True, help='Path to the pre-trained generator.pth file')
    parser.add_argument('--scaler_path', type=str, required=True, help='Path to the fitted gan_scaler.pkl file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results and plots')
    
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension used for the GAN (must match the trained GAN)')
    
    args = parser.parse_args()
    main(args)