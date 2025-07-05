# train_gan.py (WGAN-GP Version)

import argparse
import logging
import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =======================================================================================
# 1. DATA LOADING & PREPROCESSING FUNCTION (No changes needed here)
# =======================================================================================
def load_process_data(data_path):
    # This function from your script remains the same.
    # It correctly loads, preprocesses, and windows the data.
    # ... (Your existing load_process_data function) ...
    logging.info("Loading and processing data...")
    all_sessions_df_list = []
    EVENTS_FOLDER = os.path.join(data_path, '03_train_ready/event_exports') 
    RESPECK_FOLDER = os.path.join(data_path, '03_train_ready/respeck')
    NASAL_FOLDER = os.path.join(data_path, '03_train_ready/nasal_files')
    FEATURES_FOLDER = os.path.join(data_path, '03_train_ready/respeck_features')
    APNEA_EVENTS = ['Obstructive Apnea']
    event_files = glob.glob(os.path.join(EVENTS_FOLDER, '*_event_export.csv'))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {EVENTS_FOLDER}")
    logging.info(f"Found {len(event_files)} event files.")
    for event_file_path in event_files:
        base_name = os.path.basename(event_file_path)
        session_id = base_name.split('_event_export.csv')[0]
        respeck_file_path = os.path.join(RESPECK_FOLDER, f'{session_id}_respeck.csv')
        nasal_file_path = os.path.join(NASAL_FOLDER, f'{session_id}_nasal.csv')
        feature_file_path = os.path.join(FEATURES_FOLDER, f'{session_id}_respeck_features.csv')
        if not all(os.path.exists(p) for p in [respeck_file_path, nasal_file_path, feature_file_path]):
            logging.warning(f"Skipping session '{session_id}'. A corresponding file is missing.")
            continue
        logging.info(f"Processing session: {session_id}")
        df_events = pd.read_csv(event_file_path, decimal=',')
        df_nasal = pd.read_csv(nasal_file_path)
        df_respeck = pd.read_csv(respeck_file_path)
        df_features = pd.read_csv(feature_file_path)
        df_events.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True)
        df_nasal.rename(columns={'UnixTimestamp': 'timestamp_unix'}, inplace=True, errors='ignore')
        df_respeck.rename(columns={'alignedTimestamp': 'timestamp_unix'}, inplace=True)
        df_features['timestamp_unix'] = pd.to_datetime(df_features['startTimestamp'], format="mixed").astype('int64') // 10**6
        df_features['timestamp_unix_end'] = pd.to_datetime(df_features['endTimestamp'], format="mixed").astype('int64') // 10**6
        for df_ in [df_events, df_nasal, df_respeck]:
            df_['timestamp_unix'] = pd.to_numeric(df_['timestamp_unix'], errors='coerce')
            df_.dropna(subset=['timestamp_unix'], inplace=True)
            df_['timestamp_unix'] = df_['timestamp_unix'].astype('int64')
        start_time = max(df_nasal['timestamp_unix'].min(), df_respeck['timestamp_unix'].min())
        end_time = min(df_nasal['timestamp_unix'].max(), df_respeck['timestamp_unix'].max())
        df_respeck = df_respeck[(df_respeck['timestamp_unix'] >= start_time) & (df_respeck['timestamp_unix'] <= end_time)].copy()
        if df_respeck.empty:
            logging.warning(f"Skipping session '{session_id}' due to no overlapping Respeck data.")
            continue
        df_respeck = df_respeck.sort_values('timestamp_unix')
        df_features = df_features.sort_values('timestamp_unix')
        df_session_merged = pd.merge_asof(df_respeck, df_features, on='timestamp_unix', direction='backward')
        cols_to_drop = ['Unnamed: 0', 'startTimestamp', 'endTimestamp', 'timestamp_unix_end', 'type']
        df_session_merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        if df_session_merged.empty:
            logging.warning(f"Skipping session '{session_id}' due to no merge matches.")
            continue
        df_session_merged['Label'] = 0
        df_events['Duration_ms'] = (df_events['Duration'] * 1000).astype('int64')
        df_events['end_time_unix'] = df_events['timestamp_unix'] + df_events['Duration_ms']
        df_apnea_events = df_events[df_events['Event'].isin(APNEA_EVENTS)].copy()
        for _, event in df_apnea_events.iterrows():
            start_event = event['timestamp_unix']
            end_event = event['end_time_unix']
            df_session_merged.loc[df_session_merged['timestamp_unix'].between(start_event, end_event), 'Label'] = 1
        df_session_merged['SessionID'] = session_id
        all_sessions_df_list.append(df_session_merged)
    if not all_sessions_df_list:
        raise ValueError("Processing failed. No data was loaded.")
    df = pd.concat(all_sessions_df_list, ignore_index=True)
    logging.info(f"Data loaded. Raw DataFrame shape: {df.shape}")
    NUMERICAL_FEATURES = [
        'breathingSignal', 'activityLevel', 'breathingRate', 'area', 'extremas', 
        'meanActivityLevel', 'peakRespiratoryFlow', 'duration', 'BR_md', 'BR_mean', 
        'BR_std', 'AL_md', 'AL_mean', 'AL_std', 'RRV', 'RRV3MA', 'breath_regularity'
    ]
    CATEGORICAL_FEATURES = ['activityType']
    if CATEGORICAL_FEATURES[0] in df.columns:
        logging.info(f"One-hot encoding '{CATEGORICAL_FEATURES[0]}'...")
        df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix='activity')
    one_hot_cols = [col for col in df.columns if col.startswith('activity_')]
    FINAL_FEATURE_COLUMNS = NUMERICAL_FEATURES + one_hot_cols
    logging.info(f"Total number of features after encoding: {len(FINAL_FEATURE_COLUMNS)}")
    LABEL_COLUMN = 'Label'
    SESSION_ID_COLUMN = 'SessionID'
    logging.info("Imputing missing values in feature columns...")
    df[FINAL_FEATURE_COLUMNS] = df[FINAL_FEATURE_COLUMNS].ffill().bfill()
    if df[FINAL_FEATURE_COLUMNS].isnull().sum().sum() > 0:
        raise ValueError("NaNs remain in the feature data after imputation.")
    logging.info("Starting windowing process...")
    SAMPLING_RATE_HZ = 12.5
    WINDOW_DURATION_SEC = 30
    WINDOW_SIZE = int(WINDOW_DURATION_SEC * SAMPLING_RATE_HZ)
    OVERLAP_PERCENTAGE = 0.80
    STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENTAGE))
    X, y, groups = [], [], []
    for session_id, session_df in df.groupby(SESSION_ID_COLUMN):
        for i in range(0, len(session_df) - WINDOW_SIZE, STEP_SIZE):
            window_df = session_df.iloc[i : i + WINDOW_SIZE]
            if not all(col in window_df.columns for col in FINAL_FEATURE_COLUMNS):
                logging.warning(f"Skipping a window in session {session_id} due to missing feature columns.")
                continue
            features = window_df[FINAL_FEATURE_COLUMNS].values
            label = 1 if window_df[LABEL_COLUMN].sum() > 0 else 0
            X.append(features)
            y.append(label)
            groups.append(session_id)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    logging.info(f"Windowing complete. X shape: {X.shape}, y shape: {y.shape}")
    logging.info(f"Class distribution in windows: {Counter(y)}")
    return X, y, groups, FINAL_FEATURE_COLUMNS


# =======================================================================================
# 2. WGAN-GP MODEL DEFINITIONS
# =======================================================================================

class Generator(nn.Module):
    def __init__(self, latent_dim, seq_len, n_features):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), # --- WGAN-GP CHANGE: Added BatchNorm for stability ---
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (1, 256)),
        )
        # --- WGAN-GP CHANGE: Deeper Generator with two LSTM layers and Dropout ---
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.2)
        self.out = nn.Sequential(
            nn.Linear(256, n_features),
            nn.Tanh() # Still use Tanh, requires MinMaxScaler on data
        )

    def forward(self, z):
        x = self.model(z).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(x)
        return self.out(lstm_out)

# --- WGAN-GP CHANGE: The Discriminator is now called a Critic ---
class Critic(nn.Module):
    def __init__(self, seq_len, n_features):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(n_features, 128, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            # --- WGAN-GP CHANGE: No final activation function ---
            # It outputs a raw score (the "criticism"), not a probability.
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.model(lstm_out[:, -1, :])

# --- WGAN-GP CHANGE: Gradient Penalty Calculation ---
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.randn(real_samples.size(0), 1, 1, device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    critic_interpolates = critic(interpolates)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# =======================================================================================
# 3. GAN EVALUATION FUNCTION (No changes needed here)
# =======================================================================================
def evaluate_and_plot(generator, scaler, X_real_apnea, latent_dim, device, output_dir, feature_names):
    # This function from your script remains the same.
    # ... (Your existing evaluate_and_plot function) ...
    logging.info("Starting GAN evaluation...")
    generator.eval()
    N_ANALYSIS = min(1000, len(X_real_apnea))
    if N_ANALYSIS < 5:
        logging.warning(f"Not enough real apnea samples ({len(X_real_apnea)}) for a full evaluation. Skipping plots.")
        return
    z = torch.randn(N_ANALYSIS, latent_dim).to(device)
    with torch.no_grad():
        synthetic_data_scaled = generator(z).cpu().numpy()
    synthetic_data_unscaled = scaler.inverse_transform(
        synthetic_data_scaled.reshape(-1, X_real_apnea.shape[2])
    ).reshape(N_ANALYSIS, X_real_apnea.shape[1], X_real_apnea.shape[2])
    real_samples_unscaled = X_real_apnea[np.random.choice(len(X_real_apnea), N_ANALYSIS, replace=False)]
    logging.info("Creating qualitative comparison plot...")
    features_to_plot_map = {'breathingSignal': None, 'breathingRate': None, 'breath_regularity': None}
    for i, name in enumerate(feature_names):
        if name in features_to_plot_map:
            features_to_plot_map[name] = i
    logging.info(f"Attempting to plot features: {features_to_plot_map}")
    if any(v is None for v in features_to_plot_map.values()):
        logging.warning("Could not find all specified features for plotting. Defaulting to first 3 features.")
        indices_to_plot = [0, 1, 2]
        names_for_plot = feature_names[:3]
    else:
        indices_to_plot = list(features_to_plot_map.values())
        names_for_plot = list(features_to_plot_map.keys())
    fig, axes = plt.subplots(len(indices_to_plot), 2, figsize=(15, 10), sharex=True)
    if len(indices_to_plot) == 1:
        axes = np.array([axes])
    fig.suptitle('GAN Evaluation: Real vs. Synthetic Samples', fontsize=16)
    for i, (idx, name) in enumerate(zip(indices_to_plot, names_for_plot)):
        for k in range(min(5, N_ANALYSIS)):
            axes[i, 0].plot(real_samples_unscaled[k, :, idx], alpha=0.7)
            axes[i, 1].plot(synthetic_data_unscaled[k, :, idx], alpha=0.7)
        axes[i, 0].set_title(f'Real Samples - {name}')
        axes[i, 1].set_title(f'Synthetic Samples - {name}')
        axes[i, 0].grid(True, linestyle='--', alpha=0.5)
        axes[i, 1].grid(True, linestyle='--', alpha=0.5)
    plot_path = os.path.join(output_dir, 'real_vs_synthetic_plot.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Qualitative plot saved to {plot_path}")
    if N_ANALYSIS <= 40:
        logging.warning(f"t-SNE perplexity ({40}) is too high for the number of samples ({N_ANALYSIS}). Skipping t-SNE plot.")
        logging.info("Evaluation complete.")
        return
    logging.info("Running PCA and t-SNE for quantitative evaluation...")
    real_flat = real_samples_unscaled.reshape(N_ANALYSIS, -1)
    synth_flat = synthetic_data_unscaled.reshape(N_ANALYSIS, -1)
    pca = PCA(n_components=50)
    pca.fit(real_flat)
    pca_real = pca.transform(real_flat)
    pca_synth = pca.transform(synth_flat)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_data = tsne.fit_transform(np.concatenate((pca_real, pca_synth), axis=0))
    plot_df = pd.DataFrame({'t-SNE-1': tsne_data[:, 0], 't-SNE-2': tsne_data[:, 1], 'label': ['Real'] * N_ANALYSIS + ['Synthetic'] * N_ANALYSIS})
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='t-SNE-1', y='t-SNE-2', hue='label', data=plot_df, alpha=0.5)
    plt.title('t-SNE Visualization of Real vs. Synthetic Data')
    tsne_path = os.path.join(output_dir, 'tsne_visualization.png')
    plt.savefig(tsne_path)
    plt.close()
    logging.info(f"t-SNE plot saved to {tsne_path}")
    logging.info("Evaluation complete.")

# =======================================================================================
# 4. MAIN TRAINING SCRIPT (WGAN-GP Version)
# =======================================================================================

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    X, y, groups, feature_names = load_process_data(args.data_path)
    
    unique_groups = np.unique(groups)
    train_groups, _ = train_test_split(unique_groups, test_size=0.2, random_state=42)
    X_train, y_train = X[np.isin(groups, train_groups)], y[np.isin(groups, train_groups)]
    X_train_apnea = X_train[y_train == 1]
    
    if len(X_train_apnea) < 5:
        logging.error("Not enough apnea samples to train the GAN. Halting.")
        return

    scaler = MinMaxScaler(feature_range=(-1, 1))
    n_samples, n_timesteps, n_features = X_train_apnea.shape
    X_train_apnea_scaled = scaler.fit_transform(X_train_apnea.reshape(-1, n_features)).reshape(n_samples, n_timesteps, n_features)
    
    scaler_path = os.path.join(args.output_dir, 'gan_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    train_loader_gan = DataLoader(TensorDataset(torch.from_numpy(X_train_apnea_scaled).float()), batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    generator = Generator(args.latent_dim, seq_len, n_features).to(device)
    critic = Critic(seq_len, n_features).to(device) # --- WGAN-GP CHANGE: Critic instead of Discriminator
    
    # --- WGAN-GP CHANGE: Different optimizer parameters ---
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.9)) # Critic's optimizer

    logging.info("Starting WGAN-GP training...")
    interval_start_time = time.time()
    
    # --- WGAN-GP CHANGE: New Training Loop ---
    for epoch in range(args.epochs):
        for i, (real_seqs,) in enumerate(train_loader_gan):
            real_seqs = real_seqs.to(device)
            
            # ---------------------
            #  Train Critic
            # ---------------------
            optimizer_C.zero_grad()
            
            # Generate a batch of fake sequences
            z = torch.randn(real_seqs.size(0), args.latent_dim).to(device)
            fake_seqs = generator(z)
            
            # Get critic scores
            real_validity = critic(real_seqs)
            fake_validity = critic(fake_seqs)
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_seqs.data, fake_seqs.data, device)
            
            # Critic loss
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + args.lambda_gp * gradient_penalty
            
            c_loss.backward()
            optimizer_C.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            # Train the generator only every `n_critic` iterations
            if i % args.n_critic == 0:
                optimizer_G.zero_grad()
                
                # Generate a batch of fake sequences
                fake_seqs_for_g = generator(z)
                
                # Loss measures how well it fools the critic
                g_loss = -torch.mean(critic(fake_seqs_for_g))
                
                g_loss.backward()
                optimizer_G.step()

        # --- Logging Block ---
        if (epoch + 1) % args.log_interval == 0:
            interval_end_time = time.time()
            elapsed_time = interval_end_time - interval_start_time
            logging.info(
                f"[Epoch {epoch+1:>5}/{args.epochs}] | "
                f"Critic loss: {c_loss.item():.4f} | "
                f"Generator loss: {g_loss.item():.4f} | "
                f"Time/interval: {elapsed_time:.2f}s"
            )
            interval_start_time = time.time()
            
    logging.info("WGAN-GP training complete.")
    
    generator_path = os.path.join(args.output_dir, 'generator.pth')
    torch.save(generator.state_dict(), generator_path)
    logging.info(f"Generator model saved to {generator_path}")
    
    evaluate_and_plot(generator, scaler, X_train_apnea, args.latent_dim, device, args.output_dir, feature_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="WGAN-GP for Apnea Data Generation")
    
    # --- Paths ---
    parser.add_argument('--data_path', type=str, default='../../data/bishkek_csr/', help='Base path to the data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save models and plots')
    
    # --- GAN Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs') # WGANs may need more epochs
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for Adam optimizer') # Often a slightly lower LR is better
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of the latent space (noise vector)')
    parser.add_argument('--log_interval', type=int, default=200, help='Log training progress every N epochs')
    
    # --- WGAN-GP Specific Hyperparameters ---
    parser.add_argument('--n_critic', type=int, default=5, help='Number of critic training iterations per generator training iteration')
    parser.add_argument('--lambda_gp', type=int, default=10, help='Lambda coefficient for the gradient penalty')

    args = parser.parse_args()
    main(args)