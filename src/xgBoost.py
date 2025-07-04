import pandas as pd
import numpy as np
import glob
import os
import argparse  # For command-line arguments
import xgboost as xgb
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
import time

def load_and_prep_data(data_paths):
    """
    Loads all session data, merges features, applies labels, and returns a single DataFrame.
    NOTE: For production, this logic should live in a separate `makeDataset.py` script.
    """
    print("--- Starting Data Loading and Preprocessing ---")

    EVENTS_FOLDER = data_paths['events']
    RESPECK_FOLDER = data_paths['respeck']
    FEATURES_FOLDER = data_paths['features']
    APNEA_EVENT_LABELS = ['Obstructive Apnea']

    all_sessions_df_list = []
    event_files = glob.glob(os.path.join(EVENTS_FOLDER, '*_event_export.csv'))
    
    if not event_files:
        raise FileNotFoundError(f"No event files found in '{EVENTS_FOLDER}'.")

    print(f"Found {len(event_files)} event files...")
    for event_file_path in event_files:
        base_name = os.path.basename(event_file_path)
        session_id = base_name.split('_event_export.csv')[0]
        respeck_file_path = os.path.join(RESPECK_FOLDER, f'{session_id}_respeck.csv')
        # We don't need the nasal file for this pipeline as overlap is implicitly handled by the feature merge
        feature_file_path = os.path.join(FEATURES_FOLDER, f'{session_id}_respeck_features.csv')
        
        if not all(os.path.exists(p) for p in [respeck_file_path, feature_file_path]):
            print(f"  - WARNING: Skipping session '{session_id}'. A corresponding file is missing.")
            continue
        
        # --- Load necessary data sources ---
        df_events = pd.read_csv(event_file_path, decimal=',')
        df_respeck = pd.read_csv(respeck_file_path)
        df_features = pd.read_csv(feature_file_path)

        # --- Standardize timestamp columns ---
        df_events.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True)
        df_respeck.rename(columns={'alignedTimestamp': 'timestamp_unix'}, inplace=True)
        df_features.rename(columns={'startTimestamp': 'timestamp_unix'}, inplace=True)

        # Convert to numeric milliseconds
        df_events['timestamp_unix'] = pd.to_numeric(df_events['timestamp_unix'], errors='coerce').astype('int64')
        df_respeck['timestamp_unix'] = pd.to_numeric(df_respeck['timestamp_unix'], errors='coerce').astype('int64')
        
        # --- THIS IS THE FIX FOR YOUR ERROR ---
        df_features['timestamp_unix'] = pd.to_datetime(df_features['timestamp_unix'], format='mixed').astype('int64') // 10**6
        # ------------------------------------

        # --- Trim data to the feature file's range ---
        # Instead of using a nasal file, we can just use the range of the features themselves
        start_time = df_features['timestamp_unix'].min()
        end_time = (pd.to_datetime(df_features['endTimestamp'], format='mixed').astype('int64') // 10**6).max()
        df_respeck = df_respeck[(df_respeck['timestamp_unix'] >= start_time) & (df_respeck['timestamp_unix'] <= end_time)].copy()

        df_respeck = df_respeck.sort_values('timestamp_unix')
        df_features = df_features.sort_values('timestamp_unix')

        df_session_merged = pd.merge_asof(df_respeck, df_features, on='timestamp_unix', direction='backward')

        cols_to_drop = ['Unnamed: 0', 'endTimestamp', 'type']
        df_session_merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        df_session_merged['Label'] = 0
        df_events['Duration_ms'] = (df_events['Duration'] * 1000).astype('int64')
        df_events['end_time_unix'] = df_events['timestamp_unix'] + df_events['Duration_ms']
        df_apnea_events = df_events[df_events['Event'].isin(APNEA_EVENT_LABELS)].copy()

        for _, event in df_apnea_events.iterrows():
            mask = df_session_merged['timestamp_unix'].between(event['timestamp_unix'], event['end_time_unix'])
            df_session_merged.loc[mask, 'Label'] = 1

        df_session_merged['SessionID'] = session_id
        all_sessions_df_list.append(df_session_merged)

    df = pd.concat(all_sessions_df_list, ignore_index=True)
    
    # Final NaN imputation
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].ffill().bfill()
            
    print(f"--- Data Preprocessing Complete. Final DataFrame shape: {df.shape} ---")
    return df

def create_windows(df, config):
    # This function also remains the same, but we will use the 2D aggregated feature version
    print("--- Starting 2D Aggregated Windowing Process ---")
    X_windowed_features, y, groups = [], [], []
    
    for session_id, session_df in df.groupby(config['session_id_col']):
        for i in range(0, len(session_df) - config['window_size'], config['step_size']):
            window_df = session_df.iloc[i : i + config['window_size']]
            
            features = {}
            for col in config['feature_cols']:
                features[f'{col}_mean'] = window_df[col].mean()
                features[f'{col}_std'] = window_df[col].std()
                features[f'{col}_max'] = window_df[col].max()
            
            label = 1 if window_df[config['label_col']].sum() > 0 else 0
            
            X_windowed_features.append(features)
            y.append(label)
            groups.append(session_id)

    X = pd.DataFrame(X_windowed_features).values
    y = np.asarray(y)
    groups = np.asarray(groups)

    # Generate feature names for the new 2D data
    feature_names = list(pd.DataFrame(X_windowed_features).columns)

    print(f"--- Windowing Complete. Shape of X: {X.shape} ---")
    return X, y, groups, feature_names

def main():
    parser = argparse.ArgumentParser(description="XGBoost Hyperparameter Tuning Pipeline.")
    parser.add_argument('--data_dir', type=str, default='../data/bishkek_csr/03_train_ready', help='Base directory for data folders.')
    parser.add_argument('--output_dir', type=str, default='./tuned_xgboost_results', help='Directory to save results and plots.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    # --- 1. Define Configs, Feature Set, and Hyperparameter Grid ---
    config = {
        'window_size': 375,
        'step_size': 75,
        'feature_cols': ['modeActivityType', 'BR_md', 'RRV3MA', 'BR_mean', 'activityLevel', 'BR_std', 'AL_md', 'breath_regularity'],
        'label_col': 'Label',
        'session_id_col': 'SessionID',
        'random_state': 42
    }
    
    param_grid = {
        'classifier__n_estimators': [200, 400, 600, 800],
        'classifier__max_depth': [3, 4, 5, 6, 7],
        'classifier__learning_rate': [0.01, 0.05, 0.001, 0.005],
        'classifier__gamma': [0, 0.1, 1],
        'classifier__subsample': [0.8, 1.0],
    }

    # --- 2. Load Data and Create Windows ONCE ---
    data_paths = {
        'events': os.path.join(args.data_dir, 'event_exports'),
        'respeck': os.path.join(args.data_dir, 'respeck'),
        'features': os.path.join(args.data_dir, 'respeck_features')
    }
    df = load_and_prep_data(data_paths)
    X, y, groups, feature_names = create_windows(df, config)

    # --- 3. Perform LONO with Nested GridSearchCV ---
    print(f"\n{'='*70}\nSTARTING NESTED CROSS-VALIDATION FOR HYPERPARAMETER TUNING\n{'='*70}")
    
    all_preds, all_true, all_importances, all_best_params = [], [], [], []
    
    logo = LeaveOneGroupOut()
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_night = np.unique(groups[test_idx])[0]
        print(f"\n--- LONO FOLD {fold + 1}/{logo.get_n_splits(groups=groups)} --- Testing on Night: {test_night} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=config['random_state'])),
            ('classifier', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', device="cuda", random_state=config['random_state']))
        ])
        
        # Inner CV for GridSearchCV (e.g., 3-fold stratified)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        print(f"  - Best Hyperparameters for this fold: {best_params}")
        all_best_params.append({'fold': fold + 1, 'test_night': test_night, **best_params})
        
        # Evaluate the best model from the search on the held-out test set
        best_model = grid_search.best_estimator_
        preds = best_model.predict(X_test)
        all_preds.extend(preds)
        all_true.extend(y_test)
        
        # Get feature importances from the trained classifier within the pipeline
        importances = best_model.named_steps['classifier'].feature_importances_
        all_importances.append(pd.DataFrame({'feature': feature_names, 'importance': importances}))

    # --- 4. Final Aggregated Evaluation and Saving ---
    print(f"\n\n{'='*70}\nNESTED CROSS-VALIDATION COMPLETE\n{'='*70}")

    # Save the best parameters found in each fold
    pd.DataFrame(all_best_params).to_csv(os.path.join(args.output_dir, 'best_params_per_fold.csv'), index=False)

    # Save Classification Report
    report = classification_report(all_true, all_preds, target_names=['Normal (0)', 'Apnea (1)'])
    print("\nAggregated Classification Report:\n", report)
    with open(os.path.join(args.output_dir, 'final_classification_report.txt'), 'w') as f:
        f.write(report)

    # Save Confusion Matrix
    cm = confusion_matrix(all_true, all_preds, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Normal', 'Apnea'], yticklabels=['Normal', 'Apnea'])
    plt.title('Final Aggregated Normalized Confusion Matrix (LONO with GridSearchCV)')
    plt.savefig(os.path.join(args.output_dir, 'final_confusion_matrix.png'))
    plt.close()

    # Save Feature Importance
    importance_df = pd.concat(all_importances).groupby('feature')['importance'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 10))
    sns.barplot(x=importance_df.head(20), y=importance_df.head(20).index)
    plt.title('Top 20 Feature Importances (Averaged Across LONO Folds)')
    plt.savefig(os.path.join(args.output_dir, 'final_feature_importances.png'))
    plt.close()

    end_time = time.time()
    print(f"\nTotal execution time: {((end_time - start_time) / 60):.2f} minutes.")
    print(f"All results saved to: {args.output_dir}")

# Add this block to the very end of your xgBoost.py file

if __name__ == '__main__':
    main()
