"""
loader.py — DEAP dataset loading and baseline correction.
Validates every step: shapes, NaNs, Infs.
Write this first. Nothing downstream runs until this is confirmed working.
"""

import pickle
from pathlib import Path
import numpy as np

FS = 128
BASELINE_SAMPLES = 3 * FS
N_SUBJECTS = 32
N_TRIALS = 40
N_EEG = 32
N_PERIPHERAL = 8
N_SIGNAL_SAMPLES = 7680


def load_subject(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one DEAP .dat file, apply baseline correction, split EEG and peripheral.

    Baseline correction: subtract per-trial per-channel mean of the first 3 seconds
    (samples 0:384), then discard the baseline window. Signal of interest is the
    remaining 60 seconds (samples 384:8064 → 7680 samples).

    Args:
        path: path to subject .dat file (e.g. data/s01.dat)

    Returns:
        eeg: (40, 32, 7680)  — 40 trials, 32 EEG channels, 7680 samples
        peripheral: (40, 8,  7680)  — 40 trials, 8 peripheral channels, 7680 samples
        labels: (40, 4) — [valence, arousal, dominance, liking] per trial
    """
    with open(path, 'rb') as f:
        subject = pickle.load(f, encoding='latin1')

    data = subject['data']    # (40, 40, 8064)
    labels = subject['labels']  # (40, 4)

    baseline = data[:, :, :BASELINE_SAMPLES].mean(axis=2, keepdims=True)  # (40, 40, 1)
    data = data[:, :, BASELINE_SAMPLES:] - baseline # (40, 40, 7680)

    eeg = data[:, :N_EEG, :]   # (40, 32, 7680)
    peripheral = data[:, N_EEG:, :]   # (40, 8,  7680)

    return eeg, peripheral, labels

def load_all(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all 32 DEAP subjects in sorted order and concatenate into unified arrays.

    Runs shape assertions and NaN/Inf checks on every subject before concatenation.
    Prints a summary on success.

    Args:
        data_dir: directory containing s01.dat through s32.dat

    Returns:
        X_eeg:       (1280, 32, 7680)  — all subjects, EEG channels
        X_peripheral:(1280, 8,  7680)  — all subjects, peripheral channels
        y:           (1280, 4)         — all subjects, labels
        subject_ids: (1280,)           — subject index (1–32) per trial
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob('s*.dat'))

    assert len(files) == N_SUBJECTS, (
        f"Expected {N_SUBJECTS} subject files, found {len(files)} in {data_dir}"
    )

    eegs, peripherals, labels_all, ids = [], [], [], []

    for i, path in enumerate(files, start=1):
        eeg, peripheral, labels = load_subject(path)

        assert eeg.shape == (N_TRIALS, N_EEG,        N_SIGNAL_SAMPLES), \
            f"{path.name}: unexpected EEG shape {eeg.shape}"
        assert peripheral.shape == (N_TRIALS, N_PERIPHERAL, N_SIGNAL_SAMPLES), \
            f"{path.name}: unexpected peripheral shape {peripheral.shape}"
        assert labels.shape     == (N_TRIALS, 4), \
            f"{path.name}: unexpected labels shape {labels.shape}"

        assert not np.isnan(eeg).any(), f"{path.name}: NaN in EEG"
        assert not np.isinf(eeg).any(), f"{path.name}: Inf in EEG"
        assert not np.isnan(peripheral).any(), f"{path.name}: NaN in peripheral"
        assert not np.isinf(peripheral).any(), f"{path.name}: Inf in peripheral"

        eegs.append(eeg)
        peripherals.append(peripheral)
        labels_all.append(labels)
        ids.append(np.full(N_TRIALS, i, dtype=np.int32))

    X_eeg = np.concatenate(eegs) # (1280, 32, 7680)
    X_peripheral = np.concatenate(peripherals)  # (1280, 8,  7680)
    y = np.concatenate(labels_all)   # (1280, 4)
    subject_ids  = np.concatenate(ids) # (1280,)

    _print_summary(X_eeg, y, len(files))

    return X_eeg, X_peripheral, y, subject_ids

def _print_summary(X_eeg: np.ndarray, y: np.ndarray, n_subjects: int) -> None:
    dim_names = ['valence', 'arousal', 'dominance', 'liking']
    print(f"Loaded {n_subjects} subjects | {X_eeg.shape[0]} trials | "
          f"EEG shape: {X_eeg.shape}")
    for i, name in enumerate(dim_names):
        col = y[:, i]
        print(f"  {name:>10}: min={col.min():.1f}  max={col.max():.1f}  "
              f"mean={col.mean():.2f}  median={np.median(col):.1f}")
