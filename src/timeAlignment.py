import pandas as pd
import numpy as np
import pytz
import plotly.graph_objects as go
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.signal import correlate, butter, filtfilt
from scipy.stats import zscore

def loadDataCSV(PSG, RESPECK):
    respeck_df = pd.read_csv(RESPECK)
    respeck_df['timestamp'] = pd.to_datetime(respeck_df['alignedTimestamp'], unit='ms')
    tz = pytz.timezone('Asia/Bishkek')
    respeck_df['timestamp'] = respeck_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tz)

    psg_df = pd.read_csv(PSG)
    psg_df['timestamp'] = pd.to_datetime(psg_df['UnixTimestamp'], unit='ms')
    tz = pytz.timezone('Asia/Bishkek')
    psg_df['timestamp'] = psg_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tz)

    return respeck_df, psg_df

def plotSignals(psg_df, respeck_df):
    overlap_start = respeck_df['timestamp'].min()
    overlap_end = min(respeck_df['timestamp'].max(), psg_df['timestamp'].max())

    ten_min_delta = overlap_start + timedelta(minutes=10)

    respeck_overlap = respeck_df[
        (respeck_df['timestamp'] <= overlap_end) 
    ]

    psg_overlap = psg_df[
        (psg_df['timestamp'] <= overlap_end) 
    ]

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=psg_overlap['timestamp'],
        y=psg_overlap['Resp chest'],
        mode='lines',
        name='Resp Nasal - PSG',
        line=dict(color='blue')
    ))

    fig1.update_layout(
        title='PSG Resp Chest Signal',
        xaxis_title='Timestamp',
        yaxis_title='Resp Nasal Amplitude',
        hovermode='x unified',
        height=400
    )

    # === SECOND PLOT: Respeck Signals ===
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=respeck_overlap['timestamp'],
        y=respeck_overlap['breathingSignal'],
        mode='lines',
        name='Breathing Signal - Respeck',
        line=dict(color='red')
    ))
    fig1.show()
    fig2.show()

def cleanDataCrossCorrelation(psg_df, respeck_df):
    # We only need the signal column with its timestamp index.
    psg_signal = psg_df[['Resp chest']].copy()
    psg_signal['timestamp'] = pd.to_datetime(psg_df['timestamp'])
    psg_signal.set_index('timestamp', inplace=True)
    psg_signal.sort_index(inplace=True)

    # --- Respeck ---
    # We only need the signal column with its timestamp index.
    resp_signal = respeck_df[['breathingSignal']].copy()
    resp_signal['timestamp'] = pd.to_datetime(respeck_df['timestamp'])
    resp_signal.set_index('timestamp', inplace=True)
    resp_signal.sort_index(inplace=True)


    print("--- Handling Duplicate Timestamps ---")
    if psg_signal.index.has_duplicates:
        print(f"Found {psg_signal.index.duplicated().sum()} duplicates in PSG. Aggregating by mean.")
        psg_signal = psg_signal.groupby(level=0).mean()

    if resp_signal.index.has_duplicates:
        print(f"Found {resp_signal.index.duplicated().sum()} duplicates in Respeck. Aggregating by mean.")
        resp_signal = resp_signal.groupby(level=0).mean()

    # --- Step 2: Unify, Resample, and Clean on a SINGLE Grid ---

    # Combine the two signals into one DataFrame. `axis=1` is key.
    # The index will be the union of all timestamps.
    # For any given row, one column will have data, the other will be NaN.
    print("--- Combining Signals ---")
    combined_signals = pd.concat([psg_signal, resp_signal], axis=1)
    print(f"Shape of combined signals: {combined_signals.shape}")
    fs=100
    # Define the uniform sampling period
    sampling_period = f'{1000/fs}ms'

    # Resample the entire combined DataFrame. This creates ONE common time grid.
    # Both columns are now resampled onto these exact same timestamps.
    resampled_combined = combined_signals.resample(sampling_period).mean()
    print(f"Shape after resampling: {resampled_combined.shape}")

    # Interpolate to fill the gaps created by resampling
    interpolated_combined = resampled_combined.interpolate(method='linear')

    # Drop any rows where AT LEAST ONE signal is still NaN.
    # This removes the parts at the beginning and end where the signals don't overlap.
    # This effectively performs the 'inner join'.
    final_aligned_df = interpolated_combined.dropna()

    print(f"\nShape AFTER final cleaning: {final_aligned_df.shape}")
    assert not final_aligned_df.empty, "Alignment failed! Check original data overlap."


    # --- Step 3: Extract the Final Aligned Signals ---

    # Both columns now share the exact same index and are perfectly aligned.
    psg_clean_signal = final_aligned_df['Resp chest'].values
    resp_clean_signal = final_aligned_df['breathingSignal'].values

    print(f"\n\nSuccess! Created perfectly aligned signal arrays with {len(psg_clean_signal)} common data points.")

def crossCorrelated(psg_clean_signal, resp_clean_signal):
    
    print("--- Starting Pre-processing for Cross-Correlation ---")
    fs = 100
    # --- Step 1: Design a Band-Pass Filter ---
    # We want to keep frequencies between 0.1 Hz and 1.0 Hz (a safe range for breathing)
    LOWCUT = 0.1
    HIGHCUT = 1.0
    FILTER_ORDER = 2

    # Get the filter coefficients
    nyquist = 0.5 * fs
    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist
    b, a = butter(FILTER_ORDER, [low, high], btype='band')

    # --- Step 2: Apply the Filter to Both Signals ---
    # Use filtfilt for a zero-phase filter (doesn't add its own time lag)
    psg_filtered = filtfilt(b, a, psg_clean_signal)
    resp_filtered = filtfilt(b, a, resp_clean_signal)

    print("Signals have been band-pass filtered.")

    # --- Step 3: Normalize the Filtered Signals (Z-score) ---
    # This puts both signals on the same scale, making the correlation more robust.
    psg_processed = zscore(psg_filtered)
    resp_processed = zscore(resp_filtered)

    print("Filtered signals have been normalized.")


    # --- Step 5: Perform Cross-Correlation on the CLEAN signals ---
    print("\n--- Performing Cross-Correlation on Filtered Signals ---")
    correlation = correlate(psg_processed, resp_processed, mode='full', method='fft')
    n_samples = len(psg_processed)
    lags = np.arange(-n_samples + 1, n_samples)
    lag_in_samples = lags[np.argmax(correlation)]
    time_shift_seconds = lag_in_samples / fs

    print(f"\nOptimal lag found at: {lag_in_samples} samples")
    print(f"This corresponds to a time shift of: {time_shift_seconds:.3f} seconds.")

def alignRespeckTime(respeck_file, offset_seconds):
    offset_ms = offset_seconds * 1000
    respeck_df = pd.read_csv(respeck_file)

    respeck_df['alignedTimestamp'] = respeck_df['interpolatedPhoneTimestamp'] + offset_ms

    respeck_df['oldTimestamp'] = pd.to_datetime(respeck_df['interpolatedPhoneTimestamp'], unit='ms')
    tz = pytz.timezone('Asia/Bishkek')
    respeck_df['oldTimestamp'] = respeck_df['oldTimestamp'].dt.tz_localize('UTC').dt.tz_convert(tz)
    respeck_df.set_index('oldTimestamp', inplace=True, drop=False)

    respeck_df['timestamp'] = pd.to_datetime(respeck_df['alignedTimestamp'], unit='ms')
    tz = pytz.timezone('Asia/Bishkek')
    respeck_df['timestamp'] = respeck_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tz)

    print(f"Applying the offset from your analysis: {offset_seconds} seconds ({offset_ms} ms)\n")
    print("Original vs. Aligned Timestamps (in milliseconds and human-readable format):")
    print(respeck_df.columns)
    print(respeck_df[[
        'interpolatedPhoneTimestamp',
        'oldTimestamp',
        'alignedTimestamp',
        'timestamp',
    ]].head())

    # now save it

    new_order = ['alignedTimestamp', 'timestamp']

    for col in respeck_df.columns:
        if col not in new_order and col != 'oldTimestamp':
            new_order += [col]

    respeck_df_aligned = respeck_df[new_order]

    respeck_df_aligned.to_csv(respeck_file, index=False)


# alignRespeckTime('../../data/bishkek_csr/03_train_ready/respeck/04-04-2025_respeck.csv', 44.475)