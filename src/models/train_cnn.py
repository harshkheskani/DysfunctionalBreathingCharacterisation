import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import yaml
import argparse
import wandb
import numpy as np
from imblearn.over_sampling import SMOTE

# Import your modules
from models.cnn_model import OSA_CNN
from makeDataset import process_and_label_data, create_windows

def train(config):
    # Initialize Weights & Biases
    wandb.init(
        project=config['wandb']['project_name'],
        name=config['wandb']['run_name'],
        config=config
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and process data
    raw_df = process_and_label_data(config)
    X, y, groups = create_windows(raw_df, config)

    # Split data based on session IDs
    unique_session_ids = np.unique(groups)
    train_ids, test_ids = train_test_split(
        unique_session_ids, 
        test_size=config['training']['test_nights'] / len(unique_session_ids), 
        random_state=config['training']['random_state']
    )
    
    train_mask = np.isin(groups, train_ids)
    test_mask = np.isin(groups, test_ids)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Balance training data with SMOTE
    nsamples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape((nsamples, n_timesteps * n_features))
    smote = SMOTE(random_state=config['training']['random_state'])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
    X_train_resampled = X_train_resampled.reshape(-1, n_timesteps, n_features)

    # Create DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train_resampled).float(), torch.from_numpy(y_train_resampled).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Initialize model, loss, and optimizer
    model = OSA_CNN(n_features=n_features, n_outputs=config['model']['n_outputs']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # --- Training Loop ---
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = 100 * correct / total
        
        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        })
        
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Save the model artifact to W&B
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('model.pth')
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)