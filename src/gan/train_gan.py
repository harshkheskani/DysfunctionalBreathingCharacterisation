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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_process_data():
    """
    Loads all session data, performs windowing and returns X, y , groups
    """

    logging.info("Loading and processing data...")

    all_sessions_df_list = []

    EVENTS_FOLDER = '../../data/bishkek_csr/03_train_ready/event_exports' 
    RESPECK_FOLDER = '../../data/bishkek_csr/03_train_ready/respeck'
    NASAL_FOLDER = '../../data/bishkek_csr/03_train_ready/nasal_files'
    FEATURES_FOLDER = '../../data/bishkek_csr/03_train_ready/respeck_features'

    APNEA_EVENTS = ['Obstructive Apnea']


    event_files = glob.glob(os.path.join(EVENTS_FOLDER, '*_event_export.csv'))

    if not event_files:
        raise FileNotFoundError(f"No event files found in {EVENTS_FOLDER}")
    
    logging.info(f"Found {len(event_files)} event files.")

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
        df_apnea_events = df_events[df_events['Event'].isin(APNEA_EVENTS)].copy()

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
    
    if 'modeActivityType' in df.columns:
        logging.info("One-hot encoding 'modeActivityType'...")
        df = pd.get_dummies(df, columns=['modeActivityType'], prefix='activity')

    FEATURE_COLUMNS = [
        'breathingSignal', 'activityLevel', 'breathingRate',
        'area', 'extremas', 'meanActivityLevel','modeActivityType','peakRespiratoryFlow',
        'duration','BR_md','BR_mean','BR_std','AL_md','AL_mean','AL_std','RRV','RRV3MA',
        'breath_regularity'
    ]
    
    one_hot_cols = [col for col in df.columns if col.startswith('activity_')]
    FINAL_FEATURE_COLUMNS = FEATURE_COLUMNS + one_hot_cols
    logging.info(f"Total number of features after encoding: {len(FINAL_FEATURE_COLUMNS)}")

    LABEL_COLUMN = 'Label'
    SESSION_ID_COLUMN = 'SessionID'

    for col in FEATURE_COLUMNS:
        if df[col].isnull().sum() > 0:
            # Reassign the filled series back to the column
            df[col] = df[col].ffill()
            df[col] = df[col].bfill()
        
    # --- Windowing Logic ---
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
            features = window_df[FEATURE_COLUMNS].values
            label = 1 if window_df[LABEL_COLUMN].sum() > 0 else 0
            
            X.append(features)
            y.append(label)
            groups.append(session_id)
            
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    
    logging.info(f"Windowing complete. X shape: {X.shape}, y shape: {y.shape}")
    logging.info(f"Class distribution in windows: {Counter(y)}")
    
    return X, y, groups, FEATURE_COLUMNS


# =======================================================================================
# 2. GAN MODEL DEFINITIONS
# =======================================================================================

class Generator(nn.Module):
    def __init__(self, latent_dim, seq_len, n_features):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Unflatten(1, (1, 256)),
        )
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.out = nn.Sequential(nn.Linear(128, n_features), nn.Tanh())

    def forward(self, z):
        x = self.model(z).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(x)
        return self.out(lstm_out)

class Discriminator(nn.Module):
    def __init__(self, seq_len, n_features):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(n_features, 128, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.model(lstm_out[:, -1, :])
    
# =======================================================================================
# =======================================================================================
# 3. GAN EVALUATION FUNCTIONS
# =======================================================================================

def evaluate_and_plot(generator, scaler, X_real_apnea, latent_dim, device, output_dir):
    """Generates plots for qualitative and quantitative evaluation."""
    logging.info("Starting GAN evaluation...")
    generator.eval()
    
    # --- Generate Synthetic Data ---
    N_ANALYSIS = min(1000, len(X_real_apnea)) # Use a reasonable number for analysis
    z = torch.randn(N_ANALYSIS, latent_dim).to(device)
    with torch.no_grad():
        synthetic_data_scaled = generator(z).cpu().numpy()
    
    synthetic_data_unscaled = scaler.inverse_transform(
        synthetic_data_scaled.reshape(-1, X_real_apnea.shape[2])
    ).reshape(N_ANALYSIS, X_real_apnea.shape[1], X_real_apnea.shape[2])
    
    real_samples_unscaled = X_real_apnea[np.random.choice(len(X_real_apnea), N_ANALYSIS, replace=False)]

    # --- 1. Qualitative Plot ---
    logging.info("Creating qualitative comparison plot...")
    feature_indices = [0, 2, 17] # breathingSignal, breathingRate, breath_regularity
    fig, axes = plt.subplots(len(feature_indices), 2, figsize=(15, 10), sharex=True)
    for i, idx in enumerate(feature_indices):
        for k in range(5): # Plot 5 samples
            axes[i, 0].plot(real_samples_unscaled[k, :, idx], alpha=0.7)
            axes[i, 1].plot(synthetic_data_unscaled[k, :, idx], alpha=0.7)
        axes[i, 0].set_title(f'Real Samples (Feature {idx})')
        axes[i, 1].set_title(f'Synthetic Samples (Feature {idx})')
    plt.savefig(os.path.join(output_dir, 'real_vs_synthetic_plot.png'))
    plt.close()

    # --- 2. Quantitative t-SNE Plot ---
    logging.info("Running PCA and t-SNE for quantitative evaluation...")
    real_flat = real_samples_unscaled.reshape(N_ANALYSIS, -1)
    synth_flat = synthetic_data_unscaled.reshape(N_ANALYSIS, -1)
    
    pca = PCA(n_components=50)
    pca_data = pca.fit_transform(np.concatenate((real_flat, synth_flat), axis=0))
    
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_data = tsne.fit_transform(pca_data)

    plot_df = pd.DataFrame({
        't-SNE-1': tsne_data[:, 0], 't-SNE-2': tsne_data[:, 1],
        'label': ['Real'] * N_ANALYSIS + ['Synthetic'] * N_ANALYSIS
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='t-SNE-1', y='t-SNE-2', hue='label', data=plot_df, alpha=0.5)
    plt.title('t-SNE Visualization of Real vs. Synthetic Data')
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    plt.close()
    logging.info(f"Evaluation plots saved to '{output_dir}'.")


# =======================================================================================
# 4. MAIN TRAINING SCRIPT
# =======================================================================================

def main(args):
    # --- Setup ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # --- Load Data ---
    X, y, groups, feature_names = load_process_data(args.data_path)
    
    # --- Split Data ---
    # We only need the training set to train the GAN
    unique_groups = np.unique(groups)
    train_groups, _ = train_test_split(unique_groups, test_size=0.2, random_state=42)
    train_mask = np.isin(groups, train_groups)
    
    X_train, y_train = X[train_mask], y[train_mask]
    
    # Isolate apnea windows for GAN training
    X_train_apnea = X_train[y_train == 1]
    logging.info(f"Isolated {len(X_train_apnea)} apnea windows from the training set.")
    
    # --- Scale Data ---
    scaler = MinMaxScaler(feature_range=(-1, 1))
    n_samples, n_timesteps, n_features = X_train_apnea.shape
    X_train_apnea_scaled = scaler.fit_transform(X_train_apnea.reshape(-1, n_features)).reshape(n_samples, n_timesteps, n_features)
    
    # Save the scaler
    scaler_path = os.path.join(args.output_dir, 'gan_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to {scaler_path}")
    
    # --- DataLoader ---
    train_data_gan = TensorDataset(torch.from_numpy(X_train_apnea_scaled).float())
    train_loader_gan = DataLoader(train_data_gan, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # --- Initialize Models, Optimizers, Loss ---
    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    generator = Generator(args.latent_dim, seq_len, n_features).to(device)
    discriminator = Discriminator(seq_len, n_features).to(device)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    
    # --- Training Loop ---
    logging.info("Starting GAN training...")
    for epoch in range(args.epochs):
        for i, (real_seqs,) in enumerate(train_loader_gan):
            real_seqs = real_seqs.to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_pred = discriminator(real_seqs)
            d_real_loss = adversarial_loss(real_pred, torch.ones_like(real_pred))
            
            z = torch.randn(args.batch_size, args.latent_dim).to(device)
            fake_seqs = generator(z)
            fake_pred = discriminator(fake_seqs.detach())
            d_fake_loss = adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_pred = discriminator(fake_seqs)
            g_loss = adversarial_loss(g_pred, torch.ones_like(g_pred))
            g_loss.backward()
            optimizer_G.step()
        
        if (epoch + 1) % args.log_interval == 0:
            logging.info(f"[Epoch {epoch+1}/{args.epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    logging.info("GAN training complete.")
    
    # --- Save Generator Model ---
    generator_path = os.path.join(args.output_dir, 'generator.pth')
    torch.save(generator.state_dict(), generator_path)
    logging.info(f"Generator model saved to {generator_path}")
    
    # --- Final Evaluation ---
    evaluate_and_plot(generator, scaler, X_train_apnea, args.latent_dim, device, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time-Series GAN for Apnea Data Generation")
    
    # --- Paths ---
    parser.add_argument('--data_path', type=str, default='../../data/bishkek_csr/', help='Base path to the data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save models and plots')
    
    # --- GAN Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for Adam optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of the latent space (noise vector)')
    parser.add_argument('--log_interval', type=int, default=200, help='Log training progress every N epochs')

    args = parser.parse_args()
    main(args)

