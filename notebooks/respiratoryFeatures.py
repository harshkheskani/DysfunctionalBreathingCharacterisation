import numpy as np
import neurokit2 as nk # type: ignore
import pandas as pd
from scipy.integrate import simpson


def cal_timeseries_instantaneous_rr(signal):
    """
    Calculate the instantaneous respiratory rate (breaths per minute) from a given respiratory signal.

    Parameters:
    - signal (array-like): The respiratory signal data.

    Returns:
    - rsp_rate (array-like): The computed respiratory rate over time.
    """
    rsp_rate = nk.rsp_rate(signal, troughs=None, sampling_rate=12, window=10,
                           hop_size=1, method='trough', peak_method='khodadad2018',
                           interpolation_method='monotone_cubic')
    return rsp_rate


def calc_respiratory_vol_time(signal):
    """
    Calculate the Respiratory Volume per Time (RVT), which quantifies respiration depth and rate.

    Parameters:
    - signal (array-like): The respiratory signal data.

    Returns:
    - rvt (array-like): The computed respiratory volume time series.
    """
    rvt = nk.rsp_rvt(signal, sampling_rate=12, method='harrison2021',
                     boundaries=[1.5, 0.1], iterations=10,
                     show=False, silent=False)
    return rvt


def calc_peak_trough(signal):
    """
    Detect peaks and troughs in a respiratory signal and calculate symmetry metrics.

    Parameters:
    - signal (array-like): The respiratory signal data.

    Returns:
    - peak_signal (array-like): The detected peaks in the signal.
    - peak_trough_symmetry (numpy array): Symmetry measure between peak and trough amplitudes.
    - rise_decay_symmetry (numpy array): Symmetry measure between rise and decay phases.
    - info (dict): Additional information from peak detection.
    """
    # Detect peaks in the respiratory signal
    peak_signal, info = nk.rsp_peaks(signal)
    
    # Compute symmetry metrics for peak-trough and rise-decay phases
    peak_trough = nk.rsp_symmetry(signal, peak_signal, troughs=None,
                                  interpolation_method='monotone_cubic', show=False)
    
    # Extract symmetry values
    peak_trough_symmetry = peak_trough["RSP_Symmetry_PeakTrough"].to_numpy()
    rise_decay_symmetry = peak_trough["RSP_Symmetry_RiseDecay"].to_numpy()

    return peak_signal, peak_trough_symmetry, rise_decay_symmetry, info


def auc_per_breath_simpson(signal):
    """
    Compute the Area Under the Curve (AUC) per breath using Simpson’s rule.

    This method integrates the absolute signal between consecutive troughs
    to estimate breath-by-breath respiratory volume.

    Parameters:
    - signal (array-like): The respiratory signal data.

    Returns:
    - auc_values (numpy array): The computed AUC values for each breath.
    """
    # Extract peak and trough information
    _, _, _, info = calc_peak_trough(signal)
    troughs = info["RSP_Troughs"]
    
    # Ensure troughs are sorted
    troughs = sorted(troughs)

    auc_values = []

    # Iterate through each pair of consecutive troughs to compute AUC
    for i in range(len(troughs) - 1):
        start, end = troughs[i], troughs[i + 1]
        
        # Define x and y values for integration
        x_values = np.arange(start, end + 1)
        y_values = signal[start:end + 1]

        # Compute AUC using Simpson's rule
        auc = simpson(np.abs(y_values), x=x_values)
        auc_values.append(auc)

    return np.array(auc_values)


def calculate_breath_duration(signal):
    """
    Calculate breath durations based on the indices of troughs and peaks.

    Parameters:
    - signal: The respiratory signal array

    Returns:
    - breath_durations_seconds: Array of breath durations in seconds
    - inhalation_seconds: Array of inhalation durations in seconds
    - exhalation_seconds: Array of exhalation durations in seconds
    """
    _, _, _, info = calc_peak_trough(signal)
    troughs = sorted(info["RSP_Troughs"])  # Inhalation onsets
    peaks = sorted(info["RSP_Peaks"])  # Exhalation onsets

    sampling_rate = 12.5  # Hz

    # Compute breath durations (Trough-to-Trough)
    breath_durations = np.array([
        (troughs[i+1] - troughs[i]) / sampling_rate for i in range(len(troughs) - 1)
    ])

    # Compute inhalation durations in seconds (Trough-to-Peak)
    inhalation_durations = []
    for trough in troughs:
        peak_candidates = [p for p in peaks if p > trough]
        if peak_candidates:
            inhalation_durations.append((peak_candidates[0] - trough) / sampling_rate)

    # Compute exhalation durations in seconds (Peak-to-Trough)
    exhalation_durations = []
    for peak in peaks:
        trough_candidates = [t for t in troughs if t > peak]
        if trough_candidates:
            exhalation_durations.append((trough_candidates[0] - peak) / sampling_rate)

    return np.array(breath_durations), np.array(inhalation_durations), np.array(exhalation_durations)