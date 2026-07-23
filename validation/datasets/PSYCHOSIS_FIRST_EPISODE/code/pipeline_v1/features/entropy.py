"""Group D: entropy and algorithmic complexity (44) - spec section 7.4.

All four measures: per-epoch on robustly centered/scaled data, then
channel-then-region median (spec section 8's general aggregation rule).
Performance mitigation (confirmed with user): sample entropy specifically -
the dominant cost, ~1 call per channel per epoch - is capped to a
deterministic fixed-seed random sample of at most 40 epochs when more are
available. The other three measures use every retained epoch, since they are
cheap enough not to need it.
"""

import antropy as ant
import mne
import numpy as np

from pipeline_v1 import regions

PERM_ENTROPY_ORDER = 5
SAMPLE_ENTROPY_ORDER = 2
MAX_EPOCHS_FOR_SAMPLE_ENTROPY = 40
RANDOM_SEED = 97


def _robust_zscore(epoch: np.ndarray) -> np.ndarray:
    median = np.median(epoch)
    mad = np.median(np.abs(epoch - median)) * 1.4826
    return (epoch - median) / (mad + 1e-12)


def _select_epoch_subset(n_epochs: int, cap: int) -> np.ndarray:
    if n_epochs <= cap:
        return np.arange(n_epochs)
    rng = np.random.default_rng(RANDOM_SEED)
    return np.sort(rng.choice(n_epochs, size=cap, replace=False))


def _spectral_entropy_1_45hz(epoch: np.ndarray, sfreq: float) -> float:
    """Normalized Shannon entropy of the 1-45 Hz Welch PSD bins (not antropy's
    unscoped full-spectrum version, which cannot be restricted to a band)."""
    from scipy.signal import welch

    freqs, psd = welch(epoch, fs=sfreq, nperseg=min(len(epoch), 256))
    mask = (freqs >= 1.0) & (freqs <= 45.0)
    psd_band = psd[mask]
    psd_band = psd_band[psd_band > 0]
    if psd_band.size < 2:
        return np.nan
    p = psd_band / psd_band.sum()
    shannon = -np.sum(p * np.log2(p))
    return float(shannon / np.log2(psd_band.size))


def entropy_features(epochs: mne.Epochs, dataset: str) -> dict:
    data = epochs.get_data(picks="eeg") * 1e6  # (n_epochs, n_channels, n_times), volts -> uV
    ch_names = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True)]
    sfreq = epochs.info["sfreq"]
    n_epochs = data.shape[0]

    sample_ent_idx = _select_epoch_subset(n_epochs, MAX_EPOCHS_FOR_SAMPLE_ENTROPY)

    per_channel = {
        "sample_entropy": {},
        "permutation_entropy_normalized": {},
        "spectral_entropy_normalized": {},
        "lempel_ziv_complexity_normalized": {},
    }

    for ci, ch in enumerate(ch_names):
        sampen_vals, perm_vals, specent_vals, lzc_vals = [], [], [], []
        for ei in range(n_epochs):
            epoch = data[ei, ci, :]
            z = _robust_zscore(epoch)

            if ei in sample_ent_idx:
                sampen_vals.append(ant.sample_entropy(z, order=SAMPLE_ENTROPY_ORDER))
            perm_vals.append(ant.perm_entropy(z, order=PERM_ENTROPY_ORDER, delay=1, normalize=True))
            specent_vals.append(_spectral_entropy_1_45hz(epoch, sfreq))
            binary = (z >= np.median(z)).astype(int)
            lzc_vals.append(ant.lziv_complexity(binary.tolist(), normalize=True))

        per_channel["sample_entropy"][ch] = float(np.nanmedian(sampen_vals))
        per_channel["permutation_entropy_normalized"][ch] = float(np.nanmedian(perm_vals))
        per_channel["spectral_entropy_normalized"][ch] = float(np.nanmedian(specent_vals))
        per_channel["lempel_ziv_complexity_normalized"][ch] = float(np.nanmedian(lzc_vals))

    features = {}
    for measure, values in per_channel.items():
        for scope in regions.SCOPES:
            features[f"D_entropy_complexity__{measure}__{scope}"] = regions.median_region_value(
                values, dataset, scope
            )
    return features
