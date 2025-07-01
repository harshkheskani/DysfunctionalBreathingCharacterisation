import torch
import torch.nn as nn


class OSA_CNN(nn.Module):
    def __init__(self, n_features, n_outputs):
        super(OSA_CNN, self).__init__()
        
        # NOTE: PyTorch's Conv1d expects input shape (batch, features, timesteps)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        )
        
        # The output size of the conv blocks needs to be calculated to define the linear layer
        # For an input of 375, after one MaxPool1d(2), the size becomes floor(375/2) = 187
        flattened_size = 64 * 187 # (out_channels * sequence_length_after_pooling)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 100),
            nn.ReLU(),
            nn.Linear(100, n_outputs) # Output raw logits for CrossEntropyLoss
        )

    def forward(self, x):
        # --- CRITICAL STEP: Reshape input for PyTorch Conv1d ---
        # Input x has shape (batch, timesteps, features)
        # We permute it to (batch, features, timesteps)
        x = x.permute(0, 2, 1)
        # ----------------------------------------------------
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Now pass the features to the classifier
        output = self.classifier(x)
        return output