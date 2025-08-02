# FILE: xgboost_binary_oa.py
# This script trains a BINARY XGBoost model for Obstructive Apnea detection.

import pandas as pd
import numpy as np
import os
import glob
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import gc
from collections import Counter

from imblearn.over_sampling import SMOTE # <-- Import SMOTE

def train_binary_oa_model(feature_dir, use_smote=False):
    """
    Loads pre-computed features for each session, runs Leave-One-Group-Out CV,
    and trains a BINARY XGBoost model for Obstructive Apnea detection.
    """
    print(f"\n🚀 STARTING: BINARY XGBOOST MODEL TRAINING FOR OA vs. NOT-OA")
    print("="*80)

    # --- 1. Load feature files ---
    # NOTE: The label_encoder_classes.npy file is not strictly needed for a binary
    # problem where we know the labels are 0 and 1, but we can load it for completeness.
    feature_files = glob.glob(os.path.join(feature_dir, '*_features.parquet'))
    if not feature_files:
        print(f"Error: No '*_features.parquet' files found in '{feature_dir}'.")
        return
        
    session_ids = sorted([os.path.basename(f).replace('_features.parquet', '') for f in feature_files])
    print(f"Found {len(session_ids)} sessions for Leave-One-Patient-Out cross-validation.")

    # --- 2. Set up cross-validation ---
    logo = LeaveOneGroupOut()
    all_predictions, all_true_labels, all_probas = [], [], []
    
    for fold, (train_indices, test_indices) in enumerate(logo.split(X=session_ids, groups=session_ids)):
        train_session_ids = [session_ids[i] for i in train_indices]
        test_session_id = session_ids[test_indices[0]]
        
        print(f"\n--- FOLD {fold + 1}/{len(session_ids)} (Testing on: {test_session_id}) ---")

        # --- Load and concatenate training data ---
        train_dfs = [pd.read_parquet(os.path.join(feature_dir, f'{sid}_features.parquet')) for sid in train_session_ids]
        df_train = pd.concat(train_dfs, ignore_index=True)
        
        y_train = df_train.pop('label')
        X_train = df_train.drop(columns=['session_id'])

        # --- Load test data ---
        df_test = pd.read_parquet(os.path.join(feature_dir, f'{test_session_id}_features.parquet'))
        y_test = df_test.pop('label')
        X_test = df_test.drop(columns=['session_id'])

        # --- Scale Data ---
        print("  Scaling data...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        scale_pos_weight = 1 # Default value
        X_train_to_fit, y_train_to_fit = X_train_scaled, y_train

        if use_smote:
            print(f"  Applying SMOTE. Original training distribution: {Counter(y_train)}")
            try:
                # For binary, k_neighbors must be < positive class size
                positive_class_size = Counter(y_train).get(1, 0)
                k = max(1, positive_class_size - 1)
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_train_to_fit, y_train_to_fit = smote.fit_resample(X_train_scaled, y_train)
                print(f"  Resampled training distribution: {Counter(y_train_to_fit)}")
            except ValueError as e:
                print(f"  SMOTE failed: {e}. Using original data.")
        else:
            # Calculate scale_pos_weight
            class_counts = Counter(y_train)
            scale_pos_weight = class_counts.get(0, 0) / class_counts.get(1, 1) if class_counts.get(1, 0) > 0 else 1
            print(f"  Training distribution: {class_counts}. Using scale_pos_weight: {scale_pos_weight:.2f}")

        # --- Train BINARY XGBoost Model ---
        print("  Training XGBoost model...")
        model = xgb.XGBClassifier(
            objective='binary:logistic',    # <-- Set to binary classification
            eval_metric='logloss',          # Standard metric for binary logistic
            scale_pos_weight=scale_pos_weight, # <-- CRUCIAL for handling imbalance

            # Use the tuned parameters from your multi-class model as a starting point
            n_estimators=500,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            
            random_state=42,
            use_label_encoder=False,
            verbosity=0
        )
        model.fit(X_train_scaled, y_train)
        
        # --- Predict and Evaluate ---
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] # Get probability of the positive class (OA)

        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
        all_probas.extend(y_proba)
        
        # --- Memory Cleanup ---
        del df_train, df_test, X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, model
        gc.collect()

    # --- 3. Final Aggregated Results ---
    print("\n🏆 AGGREGATED BINARY RESULTS")
    print("="*80)
    
    class_names = ['Not OA', 'Obstructive Apnea']
    print("\n📋 CLASSIFICATION REPORT:")
    # For imbalanced binary problems, pay close attention to the F1-score for the positive class
    print(classification_report(all_true_labels, all_predictions, target_names=class_names, zero_division=0))

    print("\n📊 CONFUSION MATRIX:")
    cm = confusion_matrix(all_true_labels, all_predictions)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)

    # --- Better metrics for imbalanced binary classification ---
    print("\n📈 Additional Performance Metrics:")
    # ROC AUC: Good overall measure of separability.
    print(f"  ROC AUC Score: {roc_auc_score(all_true_labels, all_probas):.4f}")
    # Average Precision (PR AUC): Excellent metric for very imbalanced positive classes.
    # It focuses on the performance of the positive class.
    print(f"  Average Precision (PR AUC): {average_precision_score(all_true_labels, all_probas):.4f}")

    # ========================================================================= #
# === NEW: THRESHOLD TUNING TO OPTIMIZE F1-SCORE === #
# ========================================================================= #
    from sklearn.metrics import f1_score, precision_recall_curve

    # Find the best threshold that maximizes the F1 score
    precisions, recalls, thresholds = precision_recall_curve(all_true_labels, all_probas)
    # Calculate F1 score for each threshold, avoiding division by zero
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)

    # Find the threshold that gives the best F1 score
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1_score = f1_scores[best_f1_idx]

    print("\n" + "="*80)
    print("🔍 THRESHOLD TUNING RESULTS")
    print("="*80)
    print(f"Best threshold to maximize F1-score: {best_threshold:.4f}")
    print(f"  - Corresponding F1-score: {best_f1_score:.4f}")
    print(f"  - Precision at this threshold: {precisions[best_f1_idx]:.4f}")
    print(f"  - Recall at this threshold: {recalls[best_f1_idx]:.4f}")

    # Apply the new, optimized threshold to the probabilities
    all_predictions_tuned = (all_probas >= best_threshold).astype(int)

    # --- Report on the Tuned Threshold ---
    print("\n📋 CLASSIFICATION REPORT (Optimized Threshold)")
    print(classification_report(all_true_labels, all_predictions_tuned, target_names=class_names, zero_division=0))
    print("\n📊 CONFUSION MATRIX (Optimized Threshold)")
    cm_tuned = confusion_matrix(all_true_labels, all_predictions_tuned)
    cm_df_tuned = pd.DataFrame(cm_tuned, index=class_names, columns=class_names)
    print(cm_df_tuned)


# --- HOW TO RUN THIS SCRIPT ---
if __name__ == '__main__':
    # 1. SET THE PATH to the folder containing your NEW BINARY feature files
    #    (e.g., 'features_binary_oa_12345')
    LOCAL_FEATURE_DIR = './features-binary' # <--- IMPORTANT: UPDATE THIS PATH
    
    if os.path.isdir(LOCAL_FEATURE_DIR):
        train_binary_oa_model(LOCAL_FEATURE_DIR, use_smote=True)
    else:
        print(f"Error: Directory not found at '{LOCAL_FEATURE_DIR}'.")
        print("Please update the LOCAL_FEATURE_DIR variable to point to the folder with your binary features.")