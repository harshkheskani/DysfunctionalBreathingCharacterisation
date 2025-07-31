from respiratoryFeatures import *


def calculate_TS_breathFeatures(timestamps, signal):
    """
    Compute various time-series respiratory features from a given respiratory signal.

    This function extracts multiple breathing-related features, including respiratory rate,
    volume, amplitude, symmetry measures, breath durations, and area under the curve (AUC)
    per breath.

    Parameters:
    - timestamps (array-like): The corresponding timestamps of the respiratory signal.
    - signal (array-like): The respiratory signal data.

    Returns:
    - window_dict (dict): A dictionary containing computed respiratory features:
        - "timestamp": Original timestamps
        - "breathingSignal": Original respiratory signal
        - "rr": Instantaneous respiratory rate
        - "rvt": Respiratory volume over time
        - "amplitude": Respiratory signal amplitude
        - "peak_trough_symmetry": Symmetry measure between peak and trough amplitudes
        - "rise_decay_symmetry": Symmetry measure between rise and decay phases
        - "auc_values": Area Under the Curve (AUC) per breath
        - "peaks_binary": Binary representation of detected peaks
        - "troughs_binary": Binary representation of detected troughs
        - "breath_durations": Duration of each breath
        - "inhalation_durations": Duration of inhalation phases
        - "exhalation_durations": Duration of exhalation phases
    
    Raises:
    - Exception: Catches and prints any errors encountered during computation.
    """
    try:
        # Calculate instantaneous respiratory rate
        rr = cal_timeseries_instantaneous_rr(signal)

        # Compute respiratory volume over time
        rvt = calc_respiratory_vol_time(signal)

        # Compute peak-trough symmetry and rise-decay symmetry
        peak_signal, peak_trough_symmetry, rise_decay_symmetry, info = calc_peak_trough(signal)

        # Calculate respiratory amplitude
        amplitude = nk.rsp_amplitude(signal, peak_signal)

        # Extract binary peak and trough signals
        peaks_binary = peak_signal['RSP_Peaks']
        troughs_binary = peak_signal['RSP_Troughs']

        # Compute AUC for each breath using Simpson’s rule
        auc_values = auc_per_breath_simpson(signal)

        # Calculate breath durations, inhalation durations, and exhalation durations
        breath_durations, inhalation_durations, exhalation_durations = calculate_breath_duration(signal)

        # Store all computed features in a dictionary
        window_dict = {
            "timestamp": timestamps,
            "breathingSignal": signal,
            "rr": rr,
            "rvt": rvt,
            "amplitude": amplitude,
            "peak_trough_symmetry": peak_trough_symmetry,
            "rise_decay_symmetry": rise_decay_symmetry,
            "auc_values": auc_values,
            "peaks_binary": peaks_binary,
            "troughs_binary": troughs_binary,
            "breath_durations": breath_durations,
            "inhalation_durations": inhalation_durations,
            "exhalation_durations": exhalation_durations
        }
        print(window_dict)

        return window_dict

    except Exception as e:
        # Print any encountered errors
        print(f"Error in calculate_TS_breathFeatures: {e}")
