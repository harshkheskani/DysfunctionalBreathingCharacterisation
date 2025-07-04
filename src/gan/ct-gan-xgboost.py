# ==============================================================================
# SCRIPT: ct-gan_xgboost.py
# ==============================================================================
# This script performs the full end-to-end process:
# 1. Loads and preprocesses multi-session breath and event data.
# 2. Aggregates features into 30-second windows.
# 3. Trains a CTGAN model on the minority class (OSA events).
# 4. Generates synthetic OSA data to create a balanced training set.
# 5. Trains an XGBoost classifier on the balanced data.
# 6. Evaluates the classifier and saves all models and results to a unique folder.
# ==============================================================================

import pandas as pd
from scipy.stats import mode
import numpy as np
import glob
import os
import json
from datetime import datetime

# --- ML/DS Imports ---
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, average_precision_score
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import matplotlib.pyplot as plt

# --- Configuration ---
EVENTS_FOLDER = '../data/bishkek_csr/03_train_ready/event_exports'
FEATURES_FOLDER = '../data/bishkek_csr/03_train_ready/respeck_features'
APNEA_EVENT_LABELS = ['Obstructive Apnea']
LOCAL_TIMEZONE = 'Asia/Bishkek'
RESULTS_BASE_DIR = './results'
RANDOM_STATE = 42

# --- List of DataFrames to collect results ---
all_labeled_windows_list = []

# --- Get all session IDs from the feature files ---
all_feature_files = glob.glob(f'{FEATURES_FOLDER}/*_respeck_features.csv')
session_ids = [os.path.basename(f).split('_respeck_features.csv')[0] for f in all_feature_files]

print(f"Found {len(session_ids)} sessions. Processing one by one...")

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================

for session_id in session_ids:
    print(f"  - Processing session: {session_id}")
    
    event_file = os.path.join(EVENTS_FOLDER, f'{session_id}_event_export.csv')
    feature_file = os.path.join(FEATURES_FOLDER, f'{session_id}_respeck_features.csv')

    if not os.path.exists(event_file) or not os.path.exists(feature_file):
        print(f"    - WARNING: Missing a file for this session. Skipping.")
        continue

    features_df = pd.read_csv(feature_file)
    df_events = pd.read_csv(event_file, decimal=',')
    features_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    features_df['endTimestamp_dt'] = pd.to_datetime(features_df['endTimestamp'], format="mixed").dt.tz_convert(LOCAL_TIMEZONE)
    features_df.set_index('endTimestamp_dt', inplace=True)
    
    agg_dict = {
        'area': ['mean', 'std'], 'extremas': ['mean', 'std'], 'meanActivityLevel': ['mean', 'std'],
        'peakRespiratoryFlow': ['mean', 'std'], 'duration': ['mean', 'std'], 'BR_md': ['mean', 'std'],
        'BR_mean': ['mean', 'std'], 'BR_std': ['mean', 'std'], 'AL_md': ['mean', 'std'],
        'AL_mean': ['mean', 'std'], 'AL_std': ['mean', 'std'], 'RRV': ['mean', 'std'],
        'RRV3MA': ['mean', 'std'], 'breath_regularity': ['mean', 'std'],
        'type': ['count', lambda x: (x == 'Inhalation').sum()],
        'modeActivityType': [lambda x: mode(x, keepdims=True, nan_policy='omit')[0][0] if not x.empty else np.nan]
    }
    
    df_windows_session = features_df.groupby(pd.Grouper(freq='30S')).agg(agg_dict)
    df_windows_session.columns = ['_'.join(map(str, col)).strip('_') for col in df_windows_session.columns.values]
    df_windows_session.dropna(subset=['type_count'], inplace=True)
    df_windows_session.fillna(0, inplace=True)

    if df_windows_session.empty:
        continue

    df_windows_session['window_end_unix_ms'] = (df_windows_session.index.astype(np.int64) // 10**6)
    df_windows_session['window_start_unix_ms'] = df_windows_session['window_end_unix_ms'] - 30000

    df_events.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True)
    df_events.dropna(subset=['timestamp_unix', 'Duration'], inplace=True)
    dt_series_utc = pd.to_datetime(df_events['timestamp_unix'], unit='ms', utc=True)
    dt_series_local = dt_series_utc.dt.tz_convert(LOCAL_TIMEZONE)
    df_events['timestamp_unix_ms_local'] = (dt_series_local.astype(np.int64) // 10**6)
    df_events['end_time_unix_ms_local'] = df_events['timestamp_unix_ms_local'] + (df_events['Duration'] * 1000).astype('int64')

    df_apnea_events = df_events[df_events['Event'].isin(APNEA_EVENT_LABELS)]

    df_windows_session['Label'] = 0
    for _, event in df_apnea_events.iterrows():
        start_event_ms = event['timestamp_unix_ms_local']
        end_event_ms = event['end_time_unix_ms_local']
        overlap_mask = (df_windows_session['window_start_unix_ms'] < end_event_ms) & (df_windows_session['window_end_unix_ms'] > start_event_ms)
        df_windows_session.loc[overlap_mask, 'Label'] = 1
        
    all_labeled_windows_list.append(df_windows_session)

print("\n--- Combining all processed sessions ---")
final_windowed_df = pd.concat(all_labeled_windows_list)

y_windows = final_windowed_df['Label']
X_windows = final_windowed_df.drop(columns=['Label', 'window_start_unix_ms', 'window_end_unix_ms', 'type_count'])

print("\n--- Final Data Ready ---")
print(f"Final DataFrame shape: {X_windows.shape}")
print(f"Final class distribution:\n{y_windows.value_counts(normalize=True)}")

# ==============================================================================
# MODEL TRAINING AND EVALUATION
# ==============================================================================

# --- Create a unique directory for this run's results ---
run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
run_dir = os.path.join(RESULTS_BASE_DIR, f"run_{run_timestamp}")
os.makedirs(run_dir, exist_ok=True)
print(f"\nResults for this run will be saved in: {run_dir}")

# --- Step 1: Create a Stratified Train/Test Split ---
print("\n--- Step 1: Splitting data into training and testing sets ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_windows, y_windows, test_size=0.25, random_state=RANDOM_STATE, stratify=y_windows
)

# --- Step 2: Isolate the Minority Class for GAN Training ---
print("\n--- Step 2: Isolating minority class (OSA events) for GAN ---")
X_train_osa = X_train[y_train == 1]
print(f"Number of real OSA samples to train GAN on: {len(X_train_osa)}")

# --- Step 3: Train the CTGAN Synthesizer ---
print("\n--- Step 3: Training CTGAN on real OSA data ---")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=X_train_osa)
synthesizer = CTGANSynthesizer(metadata, epochs=500, verbose=True)
synthesizer.fit(X_train_osa)
synthesizer_path = os.path.join(run_dir, 'ctgan_synthesizer.pkl')
synthesizer.save(filepath=synthesizer_path)
print(f"CTGAN synthesizer saved to {synthesizer_path}")

# --- Step 4: Generate Synthetic Data ---
num_normal_in_train = (y_train == 0).sum()
num_osa_in_train = (y_train == 1).sum()
num_to_generate = num_normal_in_train - num_osa_in_train
print(f"\n--- Step 4: Generating {num_to_generate} synthetic OSA samples ---")
synthetic_osa_features = synthesizer.sample(num_rows=num_to_generate)

# --- Step 5: Create the New Balanced Training Dataset ---
print("\n--- Step 5: Combining real and synthetic data for a balanced training set ---")
y_synthetic = np.ones(num_to_generate)
X_train_balanced = pd.concat([X_train, synthetic_osa_features], ignore_index=True)
y_train_balanced = np.concatenate([y_train, y_synthetic])

# --- Step 6: Train the XGBoost Classifier ---
print("\n--- Step 6: Training XGBoost classifier on the balanced data ---")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=400,
    learning_rate=0.001,
    max_depth=5,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    # For GPU acceleration on the cluster (if XGBoost is GPU-enabled)
    tree_method='gpu_hist', 
    predictor='gpu_predictor'
)
model.fit(X_train_balanced, y_train_balanced)
classifier_path = os.path.join(run_dir, 'xgboost_classifier.json')
model.save_model(classifier_path)
print(f"XGBoost classifier saved to {classifier_path}")

# --- Step 7: Evaluate on the Original, Imbalanced Test Set ---
print("\n--- Step 7: Evaluating model on the original, imbalanced test set ---")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# --- Step 8: Save All Results ---
report_dict = classification_report(y_test, y_pred, target_names=['Normal (0)', 'OSA (1)'], output_dict=True)
report_dict['auprc'] = average_precision_score(y_test, y_pred_proba)
report_dict['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

metrics_path = os.path.join(run_dir, 'classification_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(report_dict, f, indent=4)
print(f"Classification report and metrics saved to {metrics_path}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'OSA (1)']))
print(f"Area Under PR Curve (AUPRC): {report_dict['auprc']:.4f}")
print(f"Area Under ROC Curve (AUC-ROC): {report_dict['roc_auc']:.4f}")

cm_path = os.path.join(run_dir, 'confusion_matrix.png')
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Normal', 'OSA'], cmap='Blues')
disp.figure_.suptitle("Confusion Matrix on Original Test Data (CTGAN + XGBoost)")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix plot saved to {cm_path}")

print("\n--- SCRIPT COMPLETE ---")