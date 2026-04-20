"""
features.py — Shared constants and feature extraction for the DEAP EEG pipeline.

This is the single source of truth for all constants. Import FS, BANDS,
ELECTRODE_REGIONS, and FAA_PAIRS from here everywhere. Never hardcode them
elsewhere.

Feature vector structure (131 features per trial):
    [0:128]   Band power: 32 electrodes × 4 bands, order ch0_theta, ch0_alpha,
              ch0_beta, ch0_gamma, ch1_theta, ..., ch31_gamma
    [128:131] FAA: log(right_alpha) - log(left_alpha) for 3 frontal pairs
              (Fp1-Fp2, AF3-AF4, F3-F4)

FAA features are a log-difference transformation of a subset of the alpha band
features already present in [0:128]. This is intentional: the explicit ratio
encodes hemispheric asymmetry directly rather than requiring the model to infer
it from separate left and right alpha features. Document this in the paper's
feature extraction section.
"""

import numpy as np
from scipy.signal import welch


FS = 128

BANDS = {
    'theta': (4,  8),
    'alpha': (8,  13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

# DEAP 10-20 channel order — verified against DEAP README
# Indices 0–31 correspond to columns 0–31 of the preprocessed .dat EEG array
CHANNEL_NAMES = [
    'Fp1', 'AF3', 'F3',  'F7',  'FC5', 'FC1', 'C3',  'T7',   # 0–7
    'CP5', 'CP1', 'P3',  'P7',  'PO3', 'O1',  'Oz',  'Pz',   # 8–15
    'Fp2', 'AF4', 'Fz',  'F4',  'F8',  'FC6', 'FC2', 'Cz',   # 16–23
    'C4',  'T8',  'CP6', 'CP2', 'P4',  'P8',  'PO4', 'O2',   # 24–31
]

ELECTRODE_REGIONS = {
    'frontal_left':    [0, 1, 2, 3, 4, 5],      # Fp1, AF3, F3, F7, FC5, FC1
    'frontal_right':   [16, 17, 19, 20, 21, 22], # Fp2, AF4, F4, F8, FC6, FC2
    'central':         [6, 18, 23, 24],          # C3, Fz, Cz, C4
    'parietal_left':   [8, 9, 10, 11, 15],       # CP5, CP1, P3, P7, Pz
    'parietal_right':  [26, 27, 28, 29],         # CP6, CP2, P4, P8
    'occipital_left':  [12, 13, 14],             # PO3, O1, Oz
    'occipital_right': [30, 31],                 # PO4, O2
    'temporal_left':   [7],                      # T7
    'temporal_right':  [25],                     # T8
}

# Frontal electrode pairs for FAA = log(right_alpha) - log(left_alpha)
# Each tuple is (left_channel_idx, right_channel_idx)
FAA_PAIRS = [
    (0,  16),   # Fp1 — Fp2
    (1,  17),   # AF3 — AF4
    (2,  19),   # F3  — F4
]

# Welch PSD parameters — 2-second window at 128 Hz, 50% overlap
# Standard EEG analysis convention; document in paper methods section
NPERSEG  = 256
NOVERLAP = 128

# Derived constants (used in models.py)
N_BANDS     = len(BANDS)      # 4
N_ELECTRODES = 32
N_BP_FEATURES = N_ELECTRODES * N_BANDS  # 128
N_FAA_FEATURES = len(FAA_PAIRS)          # 3
N_FEATURES  = N_BP_FEATURES + N_FAA_FEATURES  # 131



def _band_power(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    """
    Integrate PSD within [low, high) Hz using the trapezoidal rule.

    Args:
        psd:   power spectral density values
        freqs: corresponding frequency bins
        low:   lower bound (Hz, inclusive)
        high:  upper bound (Hz, exclusive)

    Returns:
        scalar band power
    """
    mask = (freqs >= low) & (freqs < high)
    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_trial(signal: np.ndarray) -> np.ndarray:
    """
    Extract 131 features from a single trial.

    Args:
        signal: (32, 7680) — 32 EEG channels, 7680 baseline-corrected samples

    Returns:
        features: (131,) — 128 band-power features + 3 FAA features
    """
    band_list  = list(BANDS.keys())
    band_powers = np.zeros((N_ELECTRODES, N_BANDS))  # (32, 4)

    for ch in range(N_ELECTRODES):
        freqs, psd = welch(signal[ch], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        for b, (low, high) in enumerate(BANDS.values()):
            band_powers[ch, b] = _band_power(psd, freqs, low, high)

    # 128 band-power features: flattened (32, 4) row-major
    # → ch0_theta, ch0_alpha, ch0_beta, ch0_gamma, ch1_theta, ...
    bp_features = band_powers.flatten()  # (128,)

    # 3 FAA features: log(right_alpha) - log(left_alpha)
    # Small epsilon avoids log(0); values are PSD power so always >= 0
    alpha_idx = band_list.index('alpha')
    eps = 1e-10
    faa = np.array([
        np.log(band_powers[r, alpha_idx] + eps) - np.log(band_powers[l, alpha_idx] + eps)
        for l, r in FAA_PAIRS
    ])  # (3,)

    return np.concatenate([bp_features, faa])  # (131,)


def extract_all(X_eeg: np.ndarray) -> np.ndarray:
    """
    Extract features for all trials.

    Args:
        X_eeg: (N, 32, 7680)

    Returns:
        features: (N, 131)
    """
    return np.array([extract_trial(X_eeg[i]) for i in range(len(X_eeg))])


def extract_peripheral(X_peripheral: np.ndarray) -> np.ndarray:
    """
    Extract simple statistical features from peripheral channels.
    Uses per-channel mean across time — captures baseline autonomic level per trial.
    Document in paper that this is a simple summary statistic; richer peripheral
    features (HRV, SCR peaks) are left as future work.

    Args:
        X_peripheral: (N, 8, 7680)

    Returns:
        features: (N, 8)
    """
    return X_peripheral.mean(axis=2)  # (N, 8)


def feature_names() -> list[str]:
    """
    Return 131 human-readable feature names matching extract_trial output order.
    Used for interpretability labeling and figure axis annotation.

    Returns:
        list of 131 strings, e.g. ['Fp1_theta', 'Fp1_alpha', ..., 'FAA_F3_F4']
    """
    bp_names = [
        f"{CHANNEL_NAMES[ch]}_{band}"
        for ch in range(N_ELECTRODES)
        for band in BANDS
    ]
    faa_names = [
        f"FAA_{CHANNEL_NAMES[l]}_{CHANNEL_NAMES[r]}"
        for l, r in FAA_PAIRS
    ]
    return bp_names + faa_names  # len == 131
