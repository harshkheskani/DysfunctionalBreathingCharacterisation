

import pandas as pd
import numpy as np
import glob
import os
from scipy import stats
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
import math
import argparse
from typing import Dict, List, Tuple, Optional

# ==============================================================================
# Enhanced Model Components with Visualization Support
# ==============================================================================

class VisualizableChannelAttention(nn.Module):
    """Channel attention mechanism with visualization support"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x):
        b, c, _ = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pooling  
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = avg_out + max_out
        attention_weights = self.sigmoid(out)
        
        # Store for visualization
        self.attention_weights = attention_weights.detach().cpu()
        
        return attention_weights.view(b, c, 1) * x

class VisualizableSpatialAttention(nn.Module):
    """Spatial attention with visualization support"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Store attention maps for visualization
        self.attention_map = None
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(x_cat))
        
        # Store for visualization
        self.attention_map = attention_map.detach().cpu()
        
        return attention_map * x

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class MultiScaleConv(nn.Module):
    """Multi-scale convolution with feature map storage for visualization"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Different kernel sizes to capture different temporal patterns
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )
        
        # 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )
        
        # Combine features
        self.combine = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # Store branch outputs for visualization
        self.branch_outputs = {}
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Store for visualization
        self.branch_outputs = {
            'kernel_3': branch1.detach().cpu(),
            'kernel_7': branch2.detach().cpu(),
            'kernel_15': branch3.detach().cpu(),
            'kernel_1': branch4.detach().cpu()
        }
        
        # Concatenate all branches
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.combine(out)
        
        return out

class EnhancedVisualizableCNN(nn.Module):
    """
    Enhanced CNN with comprehensive visualization capabilities
    """
    def __init__(self, n_features, n_outputs, n_timesteps):
        super().__init__()
        
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_timesteps = n_timesteps
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-scale feature extraction
        self.multiscale1 = MultiScaleConv(64, 128)
        self.pool1 = nn.MaxPool1d(2)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(128, 128, kernel_size=5, dropout=0.15)
        self.res_block2 = ResidualBlock(128, 256, kernel_size=5, stride=2, dropout=0.15)
        
        # Second multi-scale layer
        self.multiscale2 = MultiScaleConv(256, 256)
        self.pool2 = nn.MaxPool1d(2)
        
        # More residual blocks
        self.res_block3 = ResidualBlock(256, 256, kernel_size=3, dropout=0.2)
        self.res_block4 = ResidualBlock(256, 512, kernel_size=3, stride=2, dropout=0.2)
        
        # Attention mechanisms
        self.channel_attention = VisualizableChannelAttention(512)
        self.spatial_attention = VisualizableSpatialAttention()
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature size
        self.feature_size = 512
        
        # Classifier with proper regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_outputs)
        )
        
        # Storage for intermediate features and gradients
        self.intermediate_features = {}
        self.gradients = {}
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, store_features=False):
        # Input shape: (batch, timesteps, features)
        # Convert to: (batch, features, timesteps) for Conv1d
        x = x.permute(0, 2, 1)
        
        if store_features:
            self.intermediate_features['input'] = x.detach().cpu()
        
        # Initial feature extraction
        x = self.initial_conv(x)
        if store_features:
            self.intermediate_features['initial_conv'] = x.detach().cpu()
        
        # Multi-scale feature extraction
        x = self.multiscale1(x)
        if store_features:
            self.intermediate_features['multiscale1'] = x.detach().cpu()
        x = self.pool1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        if store_features:
            self.intermediate_features['res_block1'] = x.detach().cpu()
        x = self.res_block2(x)
        if store_features:
            self.intermediate_features['res_block2'] = x.detach().cpu()
        
        # Second multi-scale layer
        x = self.multiscale2(x)
        if store_features:
            self.intermediate_features['multiscale2'] = x.detach().cpu()
        x = self.pool2(x)
        
        # More residual blocks
        x = self.res_block3(x)
        if store_features:
            self.intermediate_features['res_block3'] = x.detach().cpu()
        x = self.res_block4(x)
        if store_features:
            self.intermediate_features['res_block4'] = x.detach().cpu()
        
        # Apply attention mechanisms
        x = self.channel_attention(x)
        if store_features:
            self.intermediate_features['after_channel_attention'] = x.detach().cpu()
        
        x = self.spatial_attention(x)
        if store_features:
            self.intermediate_features['after_spatial_attention'] = x.detach().cpu()
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        if store_features:
            self.intermediate_features['global_pool'] = x.detach().cpu()
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_attention_weights(self):
        """Extract attention weights for visualization"""
        return {
            'channel_attention': self.channel_attention.attention_weights,
            'spatial_attention': self.spatial_attention.attention_map
        }
    
    def get_multiscale_features(self):
        """Extract multi-scale feature maps"""
        return {
            'multiscale1': self.multiscale1.branch_outputs,
            'multiscale2': self.multiscale2.branch_outputs
        }

# ==============================================================================
# Visualization Functions
# ==============================================================================

class AttentionVisualizer:
    """Class for visualizing attention mechanisms and model interpretability"""
    
    def __init__(self, model, feature_names, class_names, device='cpu'):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.device = device
    
    def visualize_attention_weights(self, sample_data, true_label, predicted_label, 
                                  save_path=None, show_plot=True):
        """Visualize channel and spatial attention weights"""
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass with feature storage
            sample_tensor = torch.from_numpy(sample_data).float().unsqueeze(0).to(self.device)
            _ = self.model(sample_tensor, store_features=True)
            
            # Get attention weights
            attention_weights = self.model.get_attention_weights()
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Attention Analysis\nTrue: {self.class_names[true_label]}, '
                        f'Predicted: {self.class_names[predicted_label]}', fontsize=14)
            
            # Channel attention visualization
            if attention_weights['channel_attention'] is not None:
                channel_weights = attention_weights['channel_attention'].squeeze().numpy()
                
                # Top contributing channels
                top_channels = np.argsort(channel_weights)[-10:][::-1]
                axes[0, 0].bar(range(len(top_channels)), channel_weights[top_channels])
                axes[0, 0].set_title('Top 10 Channel Attention Weights')
                axes[0, 0].set_xlabel('Channel Index')
                axes[0, 0].set_ylabel('Attention Weight')
                axes[0, 0].set_xticks(range(len(top_channels)))
                axes[0, 0].set_xticklabels(top_channels, rotation=45)
                
                # All channel weights heatmap
                im1 = axes[0, 1].imshow(channel_weights.reshape(1, -1), aspect='auto', cmap='viridis')
                axes[0, 1].set_title('All Channel Attention Weights')
                axes[0, 1].set_xlabel('Channel')
                axes[0, 1].set_ylabel('Attention')
                plt.colorbar(im1, ax=axes[0, 1])
            
            # Spatial attention visualization
            if attention_weights['spatial_attention'] is not None:
                spatial_weights = attention_weights['spatial_attention'].squeeze().numpy()
                
                # Time series plot
                time_points = np.arange(len(spatial_weights))
                axes[1, 0].plot(time_points, spatial_weights)
                axes[1, 0].set_title('Spatial Attention Over Time')
                axes[1, 0].set_xlabel('Time Steps')
                axes[1, 0].set_ylabel('Attention Weight')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Heatmap
                im2 = axes[1, 1].imshow(spatial_weights.reshape(1, -1), aspect='auto', cmap='viridis')
                axes[1, 1].set_title('Spatial Attention Heatmap')
                axes[1, 1].set_xlabel('Time Steps')
                axes[1, 1].set_ylabel('Attention')
                plt.colorbar(im2, ax=axes[1, 1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if show_plot:
                plt.show()
            else:
                plt.close()
    
    def visualize_multiscale_features(self, sample_data, save_path=None, show_plot=True):
        """Visualize multi-scale feature maps"""
        self.model.eval()
        
        with torch.no_grad():
            sample_tensor = torch.from_numpy(sample_data).float().unsqueeze(0).to(self.device)
            _ = self.model(sample_tensor, store_features=True)
            
            multiscale_features = self.model.get_multiscale_features()
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('Multi-Scale Feature Maps', fontsize=16)
            
            for layer_idx, (layer_name, features) in enumerate(multiscale_features.items()):
                for branch_idx, (branch_name, feature_map) in enumerate(features.items()):
                    if feature_map is not None:
                        # Average across channels for visualization
                        avg_feature = feature_map.squeeze().mean(dim=0).numpy()
                        
                        axes[layer_idx, branch_idx].plot(avg_feature)
                        axes[layer_idx, branch_idx].set_title(f'{layer_name} - {branch_name}')
                        axes[layer_idx, branch_idx].set_xlabel('Time Steps')
                        axes[layer_idx, branch_idx].set_ylabel('Feature Value')
                        axes[layer_idx, branch_idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if show_plot:
                plt.show()
            else:
                plt.close()
        

    def visualize_feature_importance_by_class(self, X_test, y_test, n_samples_per_class=5, 
                                                save_path=None, show_plot=True):
        """
        Analyzes which internal model channels are most important for each class by
        visualizing the average channel attention weights.
        """
        self.model.eval()
        
        class_importance = {}
        num_channels = 0
        
        for class_idx in range(len(self.class_names)):
            class_mask = y_test == class_idx
            if not np.any(class_mask):
                continue
                
            class_samples = X_test[class_mask][:n_samples_per_class]
            channel_attentions = []
            
            for sample in class_samples:
                with torch.no_grad():
                    sample_tensor = torch.from_numpy(sample).float().unsqueeze(0).to(self.device)
                    _ = self.model(sample_tensor, store_features=True)
                    
                    attention_weights = self.model.get_attention_weights()
                    if attention_weights['channel_attention'] is not None:
                        weights = attention_weights['channel_attention'].squeeze().numpy()
                        channel_attentions.append(weights)
                        if num_channels == 0:
                            num_channels = len(weights) # Get the number of channels (e.g., 512)
            
            if channel_attentions:
                class_importance[class_idx] = np.mean(channel_attentions, axis=0)
        
        # --- PLOTTING LOGIC CORRECTED ---
        # We now use a line plot to show the attention distribution across internal channels.
        fig, ax = plt.subplots(figsize=(15, 8))
        
        if not class_importance:
            print("Warning: No class importance data to plot.")
            plt.close()
            return {}

        for class_idx, importance_vector in class_importance.items():
            # Ensure the importance_vector has the expected length
            if len(importance_vector) != num_channels:
                print(f"Warning: Skipping plot for class {self.class_names[class_idx]} due to mismatched shape.")
                continue
            ax.plot(np.arange(num_channels), importance_vector, 
                    label=self.class_names[class_idx], alpha=0.8)
        
        ax.set_xlabel(f'Channel Index in Final Convolutional Layer (Total {num_channels} channels)')
        ax.set_ylabel('Average Channel Attention Weight')
        ax.set_title('Class-wise Average Channel Attention Distribution')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(0, num_channels - 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return class_importance
    
    def generate_comprehensive_analysis(self, X_test, y_test, y_pred, output_dir='attention_analysis'):
        """Generate comprehensive attention analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating comprehensive attention analysis...")
        
        # 1. Feature importance by class
        print("1. Analyzing feature importance by class...")
        class_importance = self.visualize_feature_importance_by_class(
            X_test, y_test, 
            save_path=os.path.join(output_dir, 'feature_importance_by_class.png'),
            show_plot=False
        )
        
        # 2. Sample analysis for each class
        print("2. Analyzing attention patterns for sample predictions...")
        for class_idx in range(len(self.class_names)):
            # Find correctly classified samples
            correct_mask = (y_test == class_idx) & (y_pred == class_idx)
            incorrect_mask = (y_test == class_idx) & (y_pred != class_idx)
            
            if np.any(correct_mask):
                sample_idx = np.where(correct_mask)[0][0]
                self.visualize_attention_weights(
                    X_test[sample_idx], y_test[sample_idx], y_pred[sample_idx],
                    save_path=os.path.join(output_dir, f'attention_correct_{self.class_names[class_idx].lower().replace(" ", "_")}.png'),
                    show_plot=False
                )
                
                self.visualize_multiscale_features(
                    X_test[sample_idx],
                    save_path=os.path.join(output_dir, f'multiscale_correct_{self.class_names[class_idx].lower().replace(" ", "_")}.png'),
                    show_plot=False
                )
            
            if np.any(incorrect_mask):
                sample_idx = np.where(incorrect_mask)[0][0]
                self.visualize_attention_weights(
                    X_test[sample_idx], y_test[sample_idx], y_pred[sample_idx],
                    save_path=os.path.join(output_dir, f'attention_incorrect_{self.class_names[class_idx].lower().replace(" ", "_")}.png'),
                    show_plot=False
                )
        
        # 3. Generate summary report
        self._generate_analysis_report(class_importance, output_dir)
        
        print(f"Analysis complete! Results saved in '{output_dir}' directory.")
    
    def _generate_analysis_report(self, class_importance, output_dir):
        """Generate a text report of the analysis"""
        report_path = os.path.join(output_dir, 'attention_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("ATTENTION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("FEATURE IMPORTANCE BY CLASS (Top 5 features)\n")
            f.write("-" * 45 + "\n")
            
            for class_idx, importance in class_importance.items():
                f.write(f"\n{self.class_names[class_idx]}:\n")
                top_features = np.argsort(importance)[-5:][::-1]
                for i, feat_idx in enumerate(top_features):
                    f.write(f"  {i+1}. {self.feature_names[feat_idx]}: {importance[feat_idx]:.4f}\n")
            
            f.write(f"\nTotal features analyzed: {len(self.feature_names)}\n")
            f.write(f"Total classes: {len(self.class_names)}\n")

# ==============================================================================
# Integration with existing code
# ==============================================================================


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Your existing feature engineering functions remain the same...
def add_breathing_fft_features(df, sampling_rate=12.5):
    """
    Adds FFT features focused exclusively on breathing signal for sleep apnea detection.
    """
    print("Engineering breathing signal FFT features...")
    
    # FFT window parameters - optimized for sleep apnea detection
    FFT_WINDOW_SIZE = int(60 * sampling_rate)  # 60 seconds for better frequency resolution
    MIN_WINDOW_SIZE = int(20 * sampling_rate)  # Minimum 20 seconds
    
    def compute_breathing_fft_features(signal_series):
        """
        Compute comprehensive FFT features for breathing signal.
        Focuses on respiratory patterns and apnea detection.
        """
        n_samples = len(signal_series)
        
        # Initialize output arrays
        power_very_low = np.zeros(n_samples)      # 0.008-0.04 Hz (HRV, very slow patterns)
        power_low = np.zeros(n_samples)           # 0.04-0.1 Hz (slow respiratory patterns)
        power_normal_resp = np.zeros(n_samples)   # 0.1-0.3 Hz (normal breathing 6-18 BPM)
        power_fast_resp = np.zeros(n_samples)     # 0.3-0.6 Hz (fast breathing 18-36 BPM)
        power_high = np.zeros(n_samples)          # 0.6-2.0 Hz (artifacts, noise)
        
        spectral_centroid = np.zeros(n_samples)   # Center frequency of breathing
        spectral_spread = np.zeros(n_samples)     # Frequency distribution width
        dominant_frequency = np.zeros(n_samples)  # Most prominent frequency
        respiratory_regularity = np.zeros(n_samples)  # How regular is breathing
        apnea_indicator = np.zeros(n_samples)     # Low frequency power ratio
        
        signal_values = signal_series.values
        
        for i in range(n_samples):
            # Define window around current sample
            start_idx = max(0, i - FFT_WINDOW_SIZE // 2)
            end_idx = min(n_samples, i + FFT_WINDOW_SIZE // 2)
            
            # Ensure minimum window size
            if end_idx - start_idx < MIN_WINDOW_SIZE:
                if i < n_samples // 2:
                    end_idx = min(n_samples, start_idx + MIN_WINDOW_SIZE)
                else:
                    start_idx = max(0, end_idx - MIN_WINDOW_SIZE)
            
            window_signal = signal_values[start_idx:end_idx]
            
            if len(window_signal) >= MIN_WINDOW_SIZE:
                # Preprocess signal
                window_signal = window_signal - np.mean(window_signal)
                
                # Apply Hanning window
                if len(window_signal) > 1:
                    hann_window = np.hanning(len(window_signal))
                    windowed_signal = window_signal * hann_window
                else:
                    windowed_signal = window_signal
                
                # Compute FFT
                fft_vals = np.fft.fft(windowed_signal)
                fft_magnitude = np.abs(fft_vals[:len(fft_vals)//2])
                freqs = np.fft.fftfreq(len(windowed_signal), 1/sampling_rate)[:len(fft_vals)//2]
                
                if len(freqs) == 0:
                    continue
                
                # Define sleep-specific frequency bands
                very_low_mask = (freqs >= 0.008) & (freqs < 0.04)   # HRV, very slow patterns
                low_mask = (freqs >= 0.04) & (freqs < 0.1)          # Slow respiratory
                normal_resp_mask = (freqs >= 0.1) & (freqs < 0.3)   # Normal breathing (6-18 BPM)


def add_breathing_fft_features(df, sampling_rate=12.5):
    """
    Adds FFT features focused exclusively on breathing signal for sleep apnea detection.
    """
    print("Engineering breathing signal FFT features...")
    
    # FFT window parameters - optimized for sleep apnea detection
    FFT_WINDOW_SIZE = int(60 * sampling_rate)  # 60 seconds for better frequency resolution
    MIN_WINDOW_SIZE = int(20 * sampling_rate)  # Minimum 20 seconds
    
    def compute_breathing_fft_features(signal_series):
        """
        Compute comprehensive FFT features for breathing signal.
        Focuses on respiratory patterns and apnea detection.
        """
        n_samples = len(signal_series)
        
        # Initialize output arrays
        power_very_low = np.zeros(n_samples)      # 0.008-0.04 Hz (HRV, very slow patterns)
        power_low = np.zeros(n_samples)           # 0.04-0.1 Hz (slow respiratory patterns)
        power_normal_resp = np.zeros(n_samples)   # 0.1-0.3 Hz (normal breathing 6-18 BPM)
        power_fast_resp = np.zeros(n_samples)     # 0.3-0.6 Hz (fast breathing 18-36 BPM)
        power_high = np.zeros(n_samples)          # 0.6-2.0 Hz (artifacts, noise)
        
        spectral_centroid = np.zeros(n_samples)   # Center frequency of breathing
        spectral_spread = np.zeros(n_samples)     # Frequency distribution width
        dominant_frequency = np.zeros(n_samples)  # Most prominent frequency
        respiratory_regularity = np.zeros(n_samples)  # How regular is breathing
        apnea_indicator = np.zeros(n_samples)     # Low frequency power ratio
        
        signal_values = signal_series.values
        
        for i in range(n_samples):
            # Define window around current sample
            start_idx = max(0, i - FFT_WINDOW_SIZE // 2)
            end_idx = min(n_samples, i + FFT_WINDOW_SIZE // 2)
            
            # Ensure minimum window size
            if end_idx - start_idx < MIN_WINDOW_SIZE:
                if i < n_samples // 2:
                    end_idx = min(n_samples, start_idx + MIN_WINDOW_SIZE)
                else:
                    start_idx = max(0, end_idx - MIN_WINDOW_SIZE)
            
            window_signal = signal_values[start_idx:end_idx]
            
            if len(window_signal) >= MIN_WINDOW_SIZE:
                # Preprocess signal
                window_signal = window_signal - np.mean(window_signal)
                
                # Apply Hanning window
                if len(window_signal) > 1:
                    hann_window = np.hanning(len(window_signal))
                    windowed_signal = window_signal * hann_window
                else:
                    windowed_signal = window_signal
                
                # Compute FFT
                fft_vals = np.fft.fft(windowed_signal)
                fft_magnitude = np.abs(fft_vals[:len(fft_vals)//2])
                freqs = np.fft.fftfreq(len(windowed_signal), 1/sampling_rate)[:len(fft_vals)//2]
                
                if len(freqs) == 0:
                    continue
                
                # Define sleep-specific frequency bands
                very_low_mask = (freqs >= 0.008) & (freqs < 0.04)   # HRV, very slow patterns
                low_mask = (freqs >= 0.04) & (freqs < 0.1)          # Slow respiratory
                normal_resp_mask = (freqs >= 0.1) & (freqs < 0.3)   # Normal breathing (6-18 BPM)
                fast_resp_mask = (freqs >= 0.3) & (freqs < 0.6)     # Fast breathing (18-36 BPM)
                high_mask = (freqs >= 0.6) & (freqs < 2.0)          # Artifacts
                
                # Respiratory frequency range (combine normal + fast)
                resp_mask = (freqs >= 0.1) & (freqs < 0.6)
                
                # Compute power in each band
                power_very_low[i] = np.sum(fft_magnitude[very_low_mask]**2) if np.any(very_low_mask) else 0
                power_low[i] = np.sum(fft_magnitude[low_mask]**2) if np.any(low_mask) else 0
                power_normal_resp[i] = np.sum(fft_magnitude[normal_resp_mask]**2) if np.any(normal_resp_mask) else 0
                power_fast_resp[i] = np.sum(fft_magnitude[fast_resp_mask]**2) if np.any(fast_resp_mask) else 0
                power_high[i] = np.sum(fft_magnitude[high_mask]**2) if np.any(high_mask) else 0
                
                # Total power in respiratory range
                resp_power = np.sum(fft_magnitude[resp_mask]**2) if np.any(resp_mask) else 0
                total_power = np.sum(fft_magnitude**2)
                
                if total_power > 1e-10:
                    # Spectral centroid (weighted by respiratory frequencies)
                    if resp_power > 0:
                        resp_freqs = freqs[resp_mask]
                        resp_magnitude = fft_magnitude[resp_mask]
                        if len(resp_freqs) > 0 and np.sum(resp_magnitude**2) > 0:
                            spectral_centroid[i] = np.sum(resp_freqs * resp_magnitude**2) / np.sum(resp_magnitude**2)
                    
                    # Spectral spread in respiratory range
                    if resp_power > 0 and spectral_centroid[i] > 0:
                        resp_freqs = freqs[resp_mask]
                        resp_magnitude = fft_magnitude[resp_mask]
                        if len(resp_freqs) > 0:
                            spectral_spread[i] = np.sqrt(np.sum((resp_freqs - spectral_centroid[i])**2 * resp_magnitude**2) / np.sum(resp_magnitude**2))
                    
                    # Respiratory regularity (inverse of spectral spread - more regular = higher value)
                    respiratory_regularity[i] = 1.0 / (1.0 + spectral_spread[i]) if spectral_spread[i] > 0 else 1.0
                    
                    # Apnea indicator (ratio of low frequency to respiratory frequency power)
                    if resp_power > 0:
                        apnea_indicator[i] = (power_very_low[i] + power_low[i]) / resp_power
                    else:
                        apnea_indicator[i] = 10.0  # High value indicates potential apnea
                
                # Dominant frequency in respiratory range
                if np.any(resp_mask):
                    resp_fft = fft_magnitude[resp_mask]
                    if len(resp_fft) > 0:
                        dominant_idx = np.argmax(resp_fft)
                        resp_freqs = freqs[resp_mask]
                        dominant_frequency[i] = resp_freqs[dominant_idx] if dominant_idx < len(resp_freqs) else 0
        
        return pd.DataFrame({
            'power_very_low': power_very_low,
            'power_low': power_low,
            'power_normal_resp': power_normal_resp,
            'power_fast_resp': power_fast_resp,
            'power_high': power_high,
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'dominant_frequency': dominant_frequency,
            'respiratory_regularity': respiratory_regularity,
            'apnea_indicator': apnea_indicator
        }, index=signal_series.index)
    
    # Apply comprehensive FFT features to breathing signal only
    print("  - Computing comprehensive FFT features for breathing signal...")
    breathing_fft_features = df.groupby('SessionID')['breathingSignal'].apply(
        lambda x: compute_breathing_fft_features(x)
    ).reset_index(level=0, drop=True)
    
    # Add breathing FFT features
    for feature_name in breathing_fft_features.columns:
        df[f'breathing_fft_{feature_name}'] = breathing_fft_features[feature_name]
    
    # Log added features
    breathing_fft_names = [f'breathing_fft_{name}' for name in breathing_fft_features.columns]
    
    print(f"Breathing FFT features added: {breathing_fft_names}")
    print(f"Total new FFT features: {len(breathing_fft_names)}\n")
    
    return df

def enhanced_add_signal_features_breathing_focused(df):
    """
    Enhanced version focused on breathing signal features only.
    """
    print("Engineering breathing-focused signal features...")
    
    # Your existing features
    ROLLING_WINDOW_SIZE = 25
    df['breathing_signal_rolling_mean'] = df.groupby('SessionID')['breathingSignal'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
    )
    df['breathing_signal_rolling_std'] = df.groupby('SessionID')['breathingSignal'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).std()
    )
    df['accel_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    
    print(f"Time-domain features added: {['breathing_signal_rolling_mean', 'breathing_signal_rolling_std', 'accel_magnitude']}")
    
    # Add breathing-focused FFT features
    df = add_breathing_fft_features(df, sampling_rate=12.5)
    
    return df

def get_corrected_feature_columns():
    """
    Returns the correct feature columns that match what the functions actually create.
    """
    # Base features from your original data
    base_features = ['breathingSignal', 'activityLevel', 'breathingRate', 'x', 'y', 'z']
    
    # Time-domain features from enhanced_add_signal_features_breathing_focused
    time_domain_features = ['breathing_signal_rolling_mean', 'breathing_signal_rolling_std', 'accel_magnitude']
    
    # FFT features from add_breathing_fft_features (these are the ACTUAL feature names created)
    breathing_fft_features = [
        'breathing_fft_power_very_low',
        'breathing_fft_power_low', 
        'breathing_fft_power_normal_resp',
        'breathing_fft_power_fast_resp',
        'breathing_fft_power_high',
        'breathing_fft_spectral_centroid',
        'breathing_fft_spectral_spread',
        'breathing_fft_dominant_frequency',
        'breathing_fft_respiratory_regularity',
        'breathing_fft_apnea_indicator'
    ]
    
    all_features = base_features + time_domain_features + breathing_fft_features
    return all_features


# ==============================================================================
# 3. Main Execution Block
# ==============================================================================
def main():
    """Main function to run the data processing, training, and evaluation pipeline."""

    parser = argparse.ArgumentParser(description="Train an attention-based CNN for sleep apnea detection.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing event_exports, respeck, and nasal_files folders.')
    parser.add_argument('--base_output_dir', type=str, required=True,
                        help='Base directory to save results (e.g., plots, checkpoints).')
    parser.add_argument('--visualize', action='store_true',
                    help='Generate attention visualizations')
    args = parser.parse_args()
    # --- Configuration & Constants ---
    print("--- 1. Setting up configuration and constants ---")
    EVENTS_FOLDER = os.path.join(args.data_dir, 'event_exports')
    RESPECK_FOLDER = os.path.join(args.data_dir, 'respeck')
    NASAL_FOLDER = os.path.join(args.data_dir, 'nasal_files')

    EVENT_GROUP_TO_LABEL = {
        1: ['Obstructive Apnea'],
        2: ['Hypopnea', 'Central Hypopnea', 'Obstructive Hypopnea'],
        3: ['Central Apnea', 'Mixed Apnea'],
        4: ['Desaturation']
    }
    LABEL_TO_EVENT_GROUP_NAME = {
        0: 'Normal',
        1: 'Obstructive Apnea',
        2: 'Hypopnea Events',
        3: 'Central/Mixed Apnea',
        4: 'Desaturation'
    }
    N_OUTPUTS = len(EVENT_GROUP_TO_LABEL) + 1
    CLASS_NAMES = [LABEL_TO_EVENT_GROUP_NAME[i] for i in range(N_OUTPUTS)]

    SAMPLING_RATE_HZ = 12.5
    WINDOW_DURATION_SEC = 30
    WINDOW_SIZE = int(WINDOW_DURATION_SEC * SAMPLING_RATE_HZ)
    OVERLAP_PERCENTAGE = 0.80
    STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENTAGE))
    
    RANDOM_STATE = 42
    EPOCHS = 100
    BATCH_SIZE = 64
    
    FEATURE_COLUMNS = get_corrected_feature_columns()

    # --- Data Loading ---
    print("\n--- 2. Loading and preprocessing data ---")
    all_sessions_df_list = []
    event_files = glob.glob(os.path.join(EVENTS_FOLDER, '*_event_export.csv'))

    if not event_files:
        raise FileNotFoundError(f"No event files found in '{EVENTS_FOLDER}'.")

    print(f"Found {len(event_files)} event files. Processing each one...")

    for event_file_path in event_files:
        base_name = os.path.basename(event_file_path)
        session_id = base_name.split('_event_export.csv')[0]
        respeck_file_path = os.path.join(RESPECK_FOLDER, f'{session_id}_respeck.csv')
        nasal_file_path = os.path.join(NASAL_FOLDER, f'{session_id}_nasal.csv')

        if not all(os.path.exists(p) for p in [respeck_file_path, nasal_file_path]):
            print(f"  - WARNING: Skipping session '{session_id}'. A corresponding file is missing.")
            continue
        print(f"  - Processing session: {session_id}")
        
        df_events = pd.read_csv(event_file_path, decimal=',')
        df_nasal = pd.read_csv(nasal_file_path)
        df_respeck = pd.read_csv(respeck_file_path)

        df_events.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True)
        df_nasal.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True, errors='ignore')
        df_respeck.rename(columns={'alignedTimestamp': 'timestamp_unix'}, inplace=True)

        for df_ in [df_events, df_nasal, df_respeck]:
            df_['timestamp_unix'] = pd.to_numeric(df_['timestamp_unix'], errors='coerce')
            df_.dropna(subset=['timestamp_unix'], inplace=True)
            df_['timestamp_unix'] = df_['timestamp_unix'].astype('int64')

        start_time = max(df_nasal['timestamp_unix'].min(), df_respeck['timestamp_unix'].min())
        end_time = min(df_nasal['timestamp_unix'].max(), df_respeck['timestamp_unix'].max())
        
        df_respeck = df_respeck[(df_respeck['timestamp_unix'] >= start_time) & (df_respeck['timestamp_unix'] <= end_time)].copy()

        if df_respeck.empty:
            print(f"  - WARNING: Skipping session '{session_id}'. No Respeck data in the overlapping range.")
            continue

        print(f"  - Applying precise interval-based labels...")
        df_respeck['Label'] = 0
        df_events['Duration_ms'] = (df_events['Duration'] * 1000).astype('int64')
        df_events['end_time_unix'] = df_events['timestamp_unix'] + df_events['Duration_ms']
        
        for label_id, event_names_in_group in EVENT_GROUP_TO_LABEL.items():
            df_filtered_events = df_events[df_events['Event'].isin(event_names_in_group)]
            for _, event in df_filtered_events.iterrows():
                start_event = event['timestamp_unix']
                end_event = event['end_time_unix']
                df_respeck.loc[df_respeck['timestamp_unix'].between(start_event, end_event), 'Label'] = label_id

        df_respeck['SessionID'] = session_id
        all_sessions_df_list.append(df_respeck)

    if not all_sessions_df_list:
        raise ValueError("Processing failed. No data was loaded.")

    df = pd.concat(all_sessions_df_list, ignore_index=True)

    print("\n----------------------------------------------------")
    print("Data loading with PURE signals complete.")
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Final class distribution in raw data: \n{df['Label'].value_counts(normalize=True)}")

    # --- Feature Engineering & Imputation ---
    print("\n--- 3. Engineering features, imputing missing values, and normalizing ---")
    df = enhanced_add_signal_features_breathing_focused(df)
    
    # Fill std NaNs that can occur at the start of a group
    df['breathing_signal_rolling_std'].bfill(inplace=True)

    print("Checking for and imputing missing values (NaNs)...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"  - Found {df[col].isnull().sum()} NaNs in '{col}'. Applying forward-fill and backward-fill.")
            df[col].ffill(inplace=True)
            df[col].bfill(inplace=True)
    
    final_nan_count = df[FEATURE_COLUMNS].isnull().sum().sum()
    if final_nan_count > 0:
        print(f"\nWARNING: {final_nan_count} NaNs still remain in feature columns after imputation. Please investigate.")
    else:
        print("\nImputation complete. No NaNs remain in feature columns.")

    # --- Per-Session Normalization ---
    print("\nApplying per-session (per-subject) normalization...")
    df_normalized = df.copy()
    for session_id in df['SessionID'].unique():
        session_mask = df['SessionID'] == session_id
        session_features = df.loc[session_mask, FEATURE_COLUMNS]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(session_features)
        df_normalized.loc[session_mask, FEATURE_COLUMNS] = scaled_features
    print("Normalization complete.")

    # --- Windowing Data ---
    print("\n--- 4. Creating time-series windows ---")
    print(f"Number of classes: {N_OUTPUTS}")
    print(f"Class names: {CLASS_NAMES}")

    X, y, groups = [], [], []
    print("\nStarting the windowing process on normalized data...")
    for session_id, session_df in df_normalized.groupby('SessionID'):
        for i in range(0, len(session_df) - WINDOW_SIZE, STEP_SIZE):
            window_df = session_df.iloc[i : i + WINDOW_SIZE]
            features = window_df[FEATURE_COLUMNS].values
            label = stats.mode(window_df['Label'])[0]
            X.append(features)
            y.append(label)
            groups.append(session_id)

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    print("\nData windowing complete.")
    print("----------------------------------------------------")
    print(f"Shape of X (features): {X.shape}")
    print(f"Shape of y (labels):   {y.shape}")
    print(f"Shape of groups (IDs): {groups.shape}")
    print(f"Final class distribution across all windows: {Counter(y)}")
    
    # --- Device Setup ---
    print("\n--- 5. Setting up PyTorch device ---")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # --- Initial Train-Test Split & SMOTE ---
    print("\n--- 6. Performing initial train-test split for preliminary check ---")
    unique_session_ids = np.unique(groups)
    train_ids, test_ids = train_test_split(
        unique_session_ids, 
        test_size=2,
        random_state=RANDOM_STATE
    )
    train_mask = np.isin(groups, train_ids)
    test_mask = np.isin(groups, test_ids)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print("Train-test split complete.")
    print(f"Training set class distribution: {Counter(y_train)}")
    print(f"Testing set class distribution:  {Counter(y_test)}")

    nsamples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape((nsamples, n_timesteps * n_features))

    print("\nBalancing the training data using SMOTE...")
    print(f"  - Original training distribution: {Counter(y_train)}")
    
    class_counts = Counter(y_train)
    min_class_count = min(count for label, count in class_counts.items() if label != 0)
    k = min_class_count - 1

    if k < 1:
        print(f"Warning: Smallest minority class has {min_class_count} samples. Falling back to RandomOverSampler.")
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    else:
        print(f"Smallest minority class has {min_class_count} samples. Setting k_neighbors for SMOTE to {k}.")
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)

    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_reshaped, y_train)
    print(f"  - Resampled training distribution: {Counter(y_train_resampled)}")
    
    X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], n_timesteps, n_features))
    X_train_tensor = torch.from_numpy(X_train_resampled).float()
    y_train_tensor = torch.from_numpy(y_train_resampled).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("\nPyTorch DataLoaders created successfully.")

    # --- Initial Model Training & Evaluation ---
    print("\n--- 7. Training and Evaluating on the initial split ---")
    model = EnhancedVisualizableCNN(n_features=n_features, n_outputs=N_OUTPUTS, n_timesteps=n_timesteps).to(device)
    print("\nPyTorch Improved CNN model created and moved to device.")
    
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=20, verbose=False, path='checkpoint.pt')

    print("\nStarting PyTorch model training with Early Stopping and LR Scheduler...")
    for epoch in range(EPOCHS):
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

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
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
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    print("\nModel training complete. Loading best model weights...")
    model.load_state_dict(torch.load('checkpoint.pt'))
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print('\nClassification Report (Initial Split)')
    print('---------------------------------------')
    print(classification_report(all_labels, all_preds, labels=range(N_OUTPUTS), target_names=CLASS_NAMES, zero_division=0))
    
    print('\nConfusion Matrix (Initial Split)')
    print('--------------------------------')
    cm = confusion_matrix(all_labels, all_preds, labels=range(N_OUTPUTS))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix (Initial Split)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha="right")
    plt.savefig("confusion_matrix_initial_split.png", bbox_inches='tight')
    # plt.show()

    # --- Leave-One-Night-Out Cross-Validation ---
    print("\n--- 8. Starting Leave-One-Night-Out Cross-Validation ---")
    all_fold_predictions, all_fold_true_labels = [], []
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    max_grad_norm = 1.0

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_night = np.unique(groups[test_idx])[0]
        print(f"--- FOLD {fold + 1}/{n_folds} (Testing on Night: {test_night}) ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"  - Original training distribution: {Counter(y_train)}")
        nsamples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape((nsamples, n_timesteps * n_features))
        
        class_counts = Counter(y_train)
        minority_classes = {label: count for label, count in class_counts.items() if label != 0}
        
        if not minority_classes:
            print("  - No minority classes in this fold's training data. Skipping resampling.")
            X_train_resampled, y_train_resampled = X_train_reshaped, y_train
        else:
            min_class_count = min(minority_classes.values())
            k = min_class_count - 1
            if k < 1:
                print(f"  - Warning: Smallest minority class has {min_class_count} samples. Using RandomOverSampler.")
                sampler = RandomOverSampler(random_state=RANDOM_STATE)
            else:
                print(f"  - Smallest minority class has {min_class_count} samples. Setting k_neighbors for SMOTE to {k}.")
                sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_reshaped, y_train)

        print(f"  - Resampled training distribution: {Counter(y_train_resampled)}")
        X_train_resampled = X_train_resampled.reshape(-1, n_timesteps, n_features)

        X_train_tensor = torch.from_numpy(X_train_resampled).float()
        y_train_tensor = torch.from_numpy(y_train_resampled).long()
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).long()
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        model = EnhancedVisualizableCNN(n_features=n_features, n_outputs=N_OUTPUTS, n_timesteps=n_timesteps).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=20, verbose=False, path=f'lono_checkpoint_fold_attn_viz_{fold}.pt')
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                running_loss += loss.item()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"  - Early stopping triggered at epoch {epoch + 1}.")
                break

        print(f"  - Training complete for fold {fold + 1}.")
        
        model.load_state_dict(torch.load(f'lono_checkpoint_fold_attn_viz_{fold}.pt'))
        model.eval()
        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                fold_preds.extend(predicted.cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())
        
        all_fold_predictions.extend(fold_preds)
        all_fold_true_labels.extend(fold_labels)
        print(f"  - Evaluation complete for fold {fold + 1}.\n")
        if args.visualize:  # Only visualize first 3 folds to avoid too many files
            print(f"  - Generating attention visualizations for fold {fold + 1}...")
            fold_visualizer = AttentionVisualizer(model, FEATURE_COLUMNS, CLASS_NAMES, device)
            fold_output_dir = os.path.join(args.base_output_dir, f'attention_analysis_fold_{fold+1}')
            
            fold_visualizer.generate_comprehensive_analysis(
                X_test, y_test, np.array(fold_preds), 
                output_dir=fold_output_dir
            )

    # --- FINAL AGGREGATED LONO EVALUATION ---
    print("\n====================================================")
    print("Leave-One-Night-Out Cross-Validation Complete.")
    print("Aggregated Results Across All Folds:")
    print("====================================================")
    
    print(classification_report(all_fold_true_labels, all_fold_predictions, labels=range(N_OUTPUTS), target_names=CLASS_NAMES, zero_division=0))
    
    cm = confusion_matrix(all_fold_true_labels, all_fold_predictions, labels=range(N_OUTPUTS))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Aggregated Normalized Confusion Matrix (LONO)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha="right")
    plt.savefig("confusion_matrix_lono_aggregated_viz.png", bbox_inches='tight')
    # plt.show()


# ==============================================================================
# 4. Script Entry Point
# ==============================================================================
if __name__ == "__main__":
    main()