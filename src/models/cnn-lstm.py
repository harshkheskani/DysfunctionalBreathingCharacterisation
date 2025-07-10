import torch
import torch.nn as nn

# --- Example CNN-LSTM Model Definition (add to train_classifier.py) ---
class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_features, num_classes=2):
        super(CNNLSTMClassifier, self).__init__()
        # CNN layers to extract features from timesteps
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layer to process the sequence of extracted features
        # The input size to the LSTM will be 128 (from the last CNN)
        self.lstm = nn.LSTM(input_size=128, hidden_size=100, num_layers=1, batch_first=True)
        
        # Final classifier
        self.fc = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x has shape: (batch_size, sequence_length, num_features)
        # Conv1d expects (batch_size, num_features, sequence_length), so we permute
        x = x.permute(0, 2, 1)
        
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Permute back for the LSTM: (batch_size, new_sequence_length, cnn_features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        _, (h_n, _) = self.lstm(x) # We only need the final hidden state
        
        # Get the hidden state of the last time step
        x = h_n.squeeze(0)
        
        # Classifier
        x = self.fc(x)
        return x