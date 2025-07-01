import pandas as pd
import glob
import os 
import yaml
import numpy as np
from collections import Counter

def process_and_label_data(config):
    EVENTS_FOLDER = config['data_paths']['events_folder']
    APNEA_EVENT_LABELS = config['labeling']['apnea_event_labels']
    RESPECK_FOLDER = config['data_paths']['respeck_folder']
    NASAL_FOLDER = config['data_paths']['nasal_folder']
    FEATURES_FOLDER = config['data_paths']['features_folder']
    FEATURE_COLUMNS = config['windowing']['feature_columns']


    APNEA_EVENT_LABELS = config['labeling']['apnea_event_labels']

    all_sessions_df_list = []
    event_files = glob.glob(os.path.join(EVENTS_FOLDER, '*_event_export.csv'))

    if not event_files:
        raise FileNotFoundError(f"No event files found in '{EVENTS_FOLDER}'.")

    print(f"Found {len(event_files)} event files. Processing each one...")

    for event_file_path in event_files:
        
        # --- 1. Setup paths and IDs ---
        base_name = os.path.basename(event_file_path)
        session_id = base_name.split('_event_export.csv')[0]
        respeck_file_path = os.path.join(RESPECK_FOLDER, f'{session_id}_respeck.csv')
        nasal_file_path = os.path.join(NASAL_FOLDER, f'{session_id}_nasal.csv')
        feature_file_path = os.path.join(FEATURES_FOLDER, f'{session_id}_respeck_features.csv')
        
        if not all(os.path.exists(p) for p in [respeck_file_path, nasal_file_path, feature_file_path]):
            print(f"  - WARNING: Skipping session '{session_id}'. A corresponding file is missing.")
            continue
        print(f"  - Processing session: {session_id}")
        
        # --- 2. Load all data sources ---
        df_events = pd.read_csv(event_file_path, decimal=',')
        df_nasal = pd.read_csv(nasal_file_path)
        df_respeck = pd.read_csv(respeck_file_path)
        df_features = pd.read_csv(feature_file_path)

        # --- 3. Standardize timestamp columns and types ---
        df_events.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True)
        df_nasal.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True, errors='ignore')
        df_respeck.rename(columns={'alignedTimestamp': 'timestamp_unix'}, inplace=True)
        
        df_features['timestamp_unix'] = pd.to_datetime(df_features['startTimestamp'], format="mixed")
        df_features['timestamp_unix'] = df_features['timestamp_unix'].astype('int64') // 10**6

        df_features['timestamp_unix_end'] = pd.to_datetime(df_features['endTimestamp'], format="mixed")
        df_features['timestamp_unix_end'] = df_features['timestamp_unix_end'].astype('int64') // 10**6
        
        for df_ in [df_events, df_nasal, df_respeck]:
            df_['timestamp_unix'] = pd.to_numeric(df_['timestamp_unix'], errors='coerce')
            df_.dropna(subset=['timestamp_unix'], inplace=True)
            df_['timestamp_unix'] = df_['timestamp_unix'].astype('int64')

        # --- 4. Calculate the true overlapping time range ---
        start_time = max(df_nasal['timestamp_unix'].min(), df_respeck['timestamp_unix'].min())
        end_time = min(df_nasal['timestamp_unix'].max(), df_respeck['timestamp_unix'].max())
        
        # --- 5. Trim Respeck data to the overlapping time range ---
        df_respeck = df_respeck[(df_respeck['timestamp_unix'] >= start_time) & (df_respeck['timestamp_unix'] <= end_time)].copy()

        if df_respeck.empty:
            print(f"  - WARNING: Skipping session '{session_id}'. No Respeck data in the overlapping range.")
            continue

        print("  - Preparing and merging engineered features using Unix time intervals...")
        df_respeck = df_respeck.sort_values('timestamp_unix')
        df_features = df_features.sort_values('timestamp_unix')

        # Use merge_asof to find the correct feature window for each respeck data point
        df_session_merged = pd.merge_asof(
            df_respeck,
            df_features,
            on='timestamp_unix',
            direction='backward' # Finds the last feature window that started <= the respeck timestamp
        )

        cols_to_drop = ['Unnamed: 0','startTimestamp', 'endTimestamp', 'timestamp_unix_end']
        df_session_merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        if df_session_merged.empty:
            print(f"  - WARNING: Skipping session '{session_id}'. No merge matches found.")
            continue
            
        # --- 6. **NEW: Precise Interval-Based Labeling using Duration** ---
        print(f"  - Applying precise interval-based labels...")
        
        # ** Step 6a: Initialize the label column in the respeck data with 0 (Normal)
        df_session_merged['Label'] = 0
        
        # ** Step 6b: Calculate event end times using the 'Duration' column
        # The 'Duration' column has commas, which we handled with `decimal=','` at load time.
        # Convert duration from seconds to milliseconds to match the Unix timestamps.
        df_events['Duration_ms'] = (df_events['Duration'] * 1000).astype('int64')
        df_events['end_time_unix'] = df_events['timestamp_unix'] + df_events['Duration_ms']
        
        # ** Step 6c: Filter for only the apnea/hypopnea events we want to label as '1'
        df_apnea_events = df_events[df_events['Event'].isin(APNEA_EVENT_LABELS)].copy()

        # ** Step 6d: Efficiently label the respeck data using event intervals
        # This is much faster than looping. It checks which respeck timestamps fall
        # within any of the [start, end] intervals of the apnea events.
        for index, event in df_apnea_events.iterrows():
            start_event = event['timestamp_unix']
            end_event = event['end_time_unix']
            # Set the 'Label' to 1 for all respeck rows within this event's time interval
            df_session_merged.loc[df_session_merged['timestamp_unix'].between(start_event, end_event), 'Label'] = 1

        # --- 7. Finalize session data ---
        df_session_merged['SessionID'] = session_id
        all_sessions_df_list.append(df_session_merged)

    # --- Combine all nights and perform final processing ---
    if not all_sessions_df_list:
        raise ValueError("Processing failed. No data was loaded.")

    df = pd.concat(all_sessions_df_list, ignore_index=True)

    print("\n----------------------------------------------------")
    print("Data loading with PRECISE interval labeling complete.")
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Final class distribution in raw data: \n{df['Label'].value_counts(normalize=True)}")

    print("\nChecking for and imputing missing values (NaNs)...")
    for col in df:
        if col in df.columns:
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                print(f"  - Found {nan_count} NaNs in '{col}'. Applying forward-fill and backward-fill.")
                
                # Step 1: Forward-fill handles all NaNs except leading ones.
                df[col].ffill(inplace=True) 
                
                # Step 2: Backward-fill handles any remaining NaNs at the beginning of the file.
                df[col].bfill(inplace=True) 

    # Add a final check to ensure everything is clean
    final_nan_count = df[FEATURE_COLUMNS].isnull().sum().sum()
    if final_nan_count > 0:
        print(f"\nWARNING: {final_nan_count} NaNs still remain in feature columns after imputation. Please investigate.")
    else:
        print("\nImputation complete. No NaNs remain in feature columns.")
    
        # output_path = config['data_paths']['processed_output_path']
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # df.to_csv(output_path, index=False)
    # print(f"Processed data saved to {output_path}")
    return df

def create_windows(df, config):
    X = []
    y = []
    groups = [] 

    print("Starting the windowing process...")

    SAMPLING_RATE_HZ = config['windowing']['sampling_rate_hz']
    FEATURE_COLUMNS = config['windowing']['feature_columns']
    LABEL_COLUMN = config['windowing']['label_column']
    STEP_SIZE = config['windowing']['step_size']
    WINDOW_DURATION_SEC = config['windowing']['window_duration_sec']
    WINDOW_SIZE = int(WINDOW_DURATION_SEC * SAMPLING_RATE_HZ)

    # --- 3. Loop through each session (night) to create windows ---
    # We group by SessionID to ensure windows do not cross over between nights.
    for session_id, session_df in df.groupby('SessionID'):
        for i in range(0, len(session_df) - WINDOW_SIZE, STEP_SIZE):
            
            window_df = session_df.iloc[i : i + WINDOW_SIZE]
            
            features = window_df[FEATURE_COLUMNS].values
            
            # --- CORRECTED LABELING LOGIC ---
            # The 'Label' column already contains 0s and 1s.
            # If the sum of labels in the window is > 0, it means there's at least one '1' (Apnea).
            if window_df[LABEL_COLUMN].sum() > 0:
                label = 1 # Apnea
            else:
                label = 0 # Normal
            # ------------------------------------
                
            X.append(features)
            y.append(label)
            groups.append(session_id)

    # --- 4. Convert the lists into efficient NumPy arrays ---
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    # --- 5. Print a summary of the results ---
    print("\nData windowing complete.")
    print("----------------------------------------------------")
    print(f"Shape of X (features): {X.shape} -> (Num_Windows, Window_Size, Num_Features)")
    print(f"Shape of y (labels):   {y.shape}")
    print(f"Shape of groups (IDs): {groups.shape}")
    print(f"Final class distribution across all windows: {Counter(y)} (0=Normal, 1=Apnea)")

    return X, y, groups


if __name__ == '__main__':
    with open('configs/binary_cnn.yaml', 'r') as f:
        config = yaml.safe_load(f)
    process_and_label_data(config)