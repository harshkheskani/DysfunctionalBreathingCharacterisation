import pandas as pd
import numpy as np
import yaml
import argparse
import wandb
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Assume data_loader.py exists from our previous refactoring
from makeDataset import process_and_label_data, create_windows

def train_xgboost(config):
    """
    Main function to run the XGBoost training and cross-validation.
    """
    # 1. Initialize Weights & Biases
    wandb.init(
        project=config['wandb']['project_name'],
        name=config['wandb']['run_name'],
        config=config
    )

    # 2. Load and Prepare Data
    # The data loading and windowing can be reused from your previous script.
    raw_df = process_and_label_data(config)
    X, y, groups = create_windows(raw_df, config)
    # Using dummy data for demonstration
    print("Loading and creating windows...")

    
    # 3. Setup Cross-Validation
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    print(f"Starting Leave-One-Night-Out cross-validation with {n_folds} folds...")

    all_preds = []
    all_true_labels = []
    
    # 4. Cross-Validation Loop
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_night = np.unique(groups[test_idx])[0]
        print(f"\n--- FOLD {fold + 1}/{n_folds} ---")
        print(f"Testing on Night: {test_night}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Reshape data for XGBoost (from 3D window to 2D features)
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

        # Initialize and train the model
        model = xgb.XGBClassifier(**config['xgboost_params'])
        model.fit(X_train_reshaped, y_train, verbose=False)

        # Evaluate and store predictions
        preds = model.predict(X_test_reshaped)
        all_preds.extend(preds)
        all_true_labels.extend(y_test)
        
        fold_f1 = f1_score(y_test, preds)
        wandb.log({f"fold_{fold+1}_f1_score": fold_f1, "fold": fold + 1})
        print(f"Fold {fold + 1} F1-Score: {fold_f1:.4f}")

    # 5. Final Aggregated Evaluation
    print("\n--- Aggregated Results Across All Folds ---")
    report = classification_report(all_true_labels, all_preds, output_dict=True)
    print(classification_report(all_true_labels, all_preds))
    
    # Log aggregated metrics to wandb
    wandb.log({
        "final_accuracy": report['accuracy'],
        "final_f1_macro": report['macro avg']['f1-score'],
        "final_f1_weighted": report['weighted avg']['f1-score']
    })

    # Create and log confusion matrix
    cm = confusion_matrix(all_true_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Apnea'], yticklabels=['Normal', 'Apnea'])
    plt.title('Aggregated Confusion Matrix (LONO)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close() # Close the plot to prevent it from displaying in the console

    # 6. Save the final model (trained on all data) and log as artifact
    print("\nTraining final model on all data...")
    X_reshaped = X.reshape(X.shape[0], -1)
    final_model = xgb.XGBClassifier(**config['xgboost_params'])
    final_model.fit(X_reshaped, y)
    
    model_filename = "final_xgboost_model.joblib"
    joblib.dump(final_model, model_filename)
    
    artifact = wandb.Artifact('xgboost-model', type='model')
    artifact.add_file(model_filename)
    wandb.log_artifact(artifact)
    print(f"Final model saved as {model_filename} and logged to W&B.")

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the XGBoost config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_xgboost(config)