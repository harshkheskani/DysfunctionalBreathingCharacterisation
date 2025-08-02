# FILE: train_from_features.py
# Run this on your local machine.

import pandas as pd
import numpy as np
import os
import glob
import xgboost as xgb
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import gc
from collections import Counter
# Optional: To use SMOTE in Option B
from imblearn.over_sampling import SMOTE

def train_and_evaluate_from_cache(feature_dir, label_to_name_map, use_smote=False):
    """
    Loads pre-computed features for each session, runs Leave-One-Group-Out CV,
    trains an XGBoost model, and evaluates the performance.

    Args:
        feature_dir (str): Path to the directory with the Parquet feature files.
        label_to_name_map (dict): Maps integer labels to human-readable names.
        use_smote (bool): If True, applies SMOTE to the training features.
    """
    print(f"\n🚀 STARTING: MODEL TRAINING FROM CACHE IN '{feature_dir}'")
    print(f"   Using SMOTE on features: {use_smote}")
    print("="*80)

    # --- 1. Load metadata and find feature files ---
    try:
        le = LabelEncoder()
        le.classes_ = np.load(os.path.join(feature_dir, 'label_encoder_classes.npy'), allow_pickle=True)
        N_OUTPUTS = len(le.classes_)
    except FileNotFoundError:
        print(f"Error: 'label_encoder_classes.npy' not found in '{feature_dir}'.")
        return

    # Get all feature files and extract session IDs
    feature_files = glob.glob(os.path.join(feature_dir, '*_features.parquet'))
    if not feature_files:
        print(f"Error: No '*_features.parquet' files found in '{feature_dir}'.")
        return
        
    # Extract session IDs from filenames
    session_ids = sorted([os.path.basename(f).replace('_features.parquet', '') for f in feature_files])
    print(f"Found {len(session_ids)} sessions to use for Leave-One-Patient-Out cross-validation.")

    # --- 2. Set up cross-validation ---
    logo = LeaveOneGroupOut()
    all_predictions, all_true_labels = [], []
    
    # logo.split(session_ids) gives indices. We use these to pick session ID strings.
    for fold, (train_indices, test_indices) in enumerate(logo.split(X=session_ids, groups=session_ids)):
        train_session_ids = [session_ids[i] for i in train_indices]
        # In LOGO, there's always exactly one test index per fold
        test_session_id = session_ids[test_indices[0]]
        
        print(f"\n--- FOLD {fold + 1}/{len(session_ids)} (Testing on: {test_session_id}) ---")

        # --- Load and concatenate training data ---
        print(f"  Loading {len(train_session_ids)} training sessions...")
        train_dfs = [pd.read_parquet(os.path.join(feature_dir, f'{sid}_features.parquet')) for sid in train_session_ids]
        df_train = pd.concat(train_dfs, ignore_index=True)
        
        y_train = df_train.pop('label')
        X_train = df_train.drop(columns=['session_id'])

        # --- Load test data ---
        print(f"  Loading test session: {test_session_id}")
        df_test = pd.read_parquet(os.path.join(feature_dir, f'{test_session_id}_features.parquet'))
        y_test = df_test.pop('label')
        X_test = df_test.drop(columns=['session_id'])

        # --- Scale Data ---
        print("  Scaling data...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # --- OPTIONAL: Handle imbalance with SMOTE on features ---
        X_train_to_fit, y_train_to_fit = X_train_scaled, y_train
        if use_smote:
            print(f"  Applying SMOTE. Original training distribution: {Counter(y_train)}")
            try:
                # k_neighbors must be less than the smallest class size
                min_class_size = min(Counter(y_train).values())
                smote = SMOTE(random_state=42, k_neighbors=max(1, min_class_size - 1))
                X_train_to_fit, y_train_to_fit = smote.fit_resample(X_train_scaled, y_train)
                print(f"  Resampled training distribution: {Counter(y_train_to_fit)}")
            except ValueError as e:
                print(f"  SMOTE failed: {e}. Using original data. This often happens if a class is too small.")

        # --- Train XGBoost Model ---
        print("  Training XGBoost model...")
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=N_OUTPUTS,
            
            # --- TUNED PARAMETERS ---
            n_estimators=500,        # More trees to learn more complex patterns
            max_depth=4,             # Shallower trees to prevent overfitting on synthetic data
            learning_rate=0.05,      # A smaller learning rate requires more estimators but is more robust
            subsample=0.8,           # Use 80% of data for training each tree (combats overfitting)
            colsample_bytree=0.8,    # Use 80% of features for training each tree (combats overfitting)
            gamma=0.1,               # A small gamma value for regularization

            # --- KEEP THESE ---
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        )
        model.fit(X_train_to_fit, y_train_to_fit)
        
        # --- Predict and Evaluate ---
        y_pred = model.predict(X_test_scaled)
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
        
        print(f"  Fold balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
        del df_train, df_test, X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, X_train_to_fit, y_train_to_fit, model
        gc.collect()

    # --- 3. Final Aggregated Results ---
    print("\n🏆 AGGREGATED RESULTS")
    print("="*80)
    print(f"Overall Balanced Accuracy: {balanced_accuracy_score(all_true_labels, all_predictions):.4f}")
    
    # Create class names from the label encoder and the mapping
    class_names = [label_to_name_map.get(c, f"Unknown Class {c}") for c in le.classes_]

    print("\n📋 CLASSIFICATION REPORT:")
    print(classification_report(all_true_labels, all_predictions, target_names=class_names, zero_division=0))

    print("\n📊 CONFUSION MATRIX:")
    cm = confusion_matrix(all_true_labels, all_predictions, labels=le.classes_)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)


# --- HOW TO RUN THIS SCRIPT ---
if __name__ == '__main__':
    # 1. SET THE PATH to the folder you downloaded from the cluster
    LOCAL_FEATURE_DIR = './features'
    
    # 2. DEFINE THE MAPPING from the integer labels to their names
    # This must match what you used on the cluster!
    LABEL_TO_EVENT_GROUP_NAME = {
        0: 'Normal',
        1: 'Obstructive Apnea',
        2: 'Hypopnea',
        3: 'Central/Mixed Apnea',
        4: 'Desaturation'
    }

    # 3. CHOOSE YOUR METHOD and run the script from your terminal:
    #    python train_from_features.py
    
    # # --- Option A: No SMOTE (Recommended first) ---
    # train_and_evaluate_from_cache(
    #     feature_dir='./features',
    #     label_to_name_map=LABEL_TO_EVENT_GROUP_NAME,
    #     use_smote=False
    # )
    
 #   --- Option B: Use SMOTE on features (Uncomment to try) ---
    print("\n\n" + "#"*80)
    print("### RUNNING AGAIN WITH SMOTE ON FEATURES ###")
    print("#"*80 + "\n")
    train_and_evaluate_from_cache(
        feature_dir=LOCAL_FEATURE_DIR,
        label_to_name_map=LABEL_TO_EVENT_GROUP_NAME,
        use_smote=True
    )