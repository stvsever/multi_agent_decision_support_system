"""Reusable EEG preprocessing and band/ROI power extraction for ds003944 + ds003947."""

import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# 6 macro-regions x hemisphere. Midline (z-suffix) channels are shared by both
# hemispheres of their macro-region. M2 (right mastoid, utility channel) is
# intentionally excluded from every ROI.
ROIS = {
    "Prefrontal-L": ["Fp1", "AF7", "AF3", "Fpz"],
    "Prefrontal-R": ["Fp2", "AF4", "AF6", "Fpz"],
    "Frontal-L": ["F7", "F5", "F3", "F1", "Fz"],
    "Frontal-R": ["F8", "F6", "F4", "F2", "Fz"],
    "Central-L": ["FT7", "FC5", "FC1", "T7", "C5", "C3", "C1", "Cz"],
    "Central-R": ["FT8", "FC6", "FC2", "T8", "C6", "C4", "C2", "Cz"],
    "Temporal-L": ["FT9", "T9", "TP9", "TP7"],
    "Temporal-R": ["FT10", "T10", "TP10", "TP8"],
    "Parietal-L": ["CP3", "CP1", "P7", "P5", "P3", "P1", "Pz"],
    "Parietal-R": ["CP4", "CP2", "P8", "P6", "P4", "P2", "Pz"],
    "Occipital-L": ["PO7", "PO3", "O1", "Oz", "Iz"],
    "Occipital-R": ["PO8", "PO4", "O2", "Oz", "Iz"],
}

# channels.tsv casing doesn't always match MNE's standard_1005 montage naming
# (e.g. ds003944 has "FP1"/"FPz"/"FP2", ds003947 has "OZ" - montage wants
# "Fp1"/"Fpz"/"Fp2"/"Oz"). Resolve case-insensitively against the montage
# itself rather than hardcoding every exception.
_MONTAGE_NAMES_BY_LOWER = {
    name.lower(): name for name in mne.channels.make_standard_montage("standard_1005").ch_names
}


def _canonical_channel_name(name: str) -> str:
    return _MONTAGE_NAMES_BY_LOWER.get(name.lower(), name)


def read_raw_brainvision_no_markers(vhdr_path: Path, **kwargs) -> mne.io.Raw:
    """Load a BrainVision recording that has no .vmrk marker file.

    Both ds003944 and ds003947 were exported from EEGLAB without event
    markers (pure resting-state, nothing to mark), so their .vhdr files
    have no "MarkerFile=" line under [Common Infos]. mne.io.read_raw_brainvision
    requires that line to be present and pointing at a real file, so we
    patch a temporary copy of the header to reference an empty placeholder
    marker file. The original .vhdr/.eeg files are never modified; the
    binary EEG data is referenced by absolute path, not copied.
    """
    vhdr_path = Path(vhdr_path).resolve()
    header_text = vhdr_path.read_text(encoding="utf-8")
    if "MarkerFile=" in header_text:
        # This file already declares a marker file - load it normally.
        return mne.io.read_raw_brainvision(vhdr_path, **kwargs)

    data_file = vhdr_path.parent / header_text.split("DataFile=")[1].splitlines()[0].strip()

    tmp_dir = Path(tempfile.gettempdir()) / "brainvision_no_markers"
    tmp_dir.mkdir(exist_ok=True)

    stub_marker = tmp_dir / "empty.vmrk"
    stub_marker.touch(exist_ok=True)

    patched_text = header_text.replace(
        "[Common Infos]",
        f"[Common Infos]\nMarkerFile={stub_marker}",
        1,
    ).replace(
        f"DataFile={data_file.name}",
        f"DataFile={data_file}",
        1,
    )

    patched_vhdr = tmp_dir / vhdr_path.name
    patched_vhdr.write_text(patched_text)

    return mne.io.read_raw_brainvision(patched_vhdr, **kwargs)


def _rename_channels_from_tsv(raw: mne.io.Raw, channels_tsv_path: Path) -> None:
    """Replace the anonymized EEG001..EEG064 labels with the real 10-10 names.

    The .vhdr channel labels are anonymized; the true channel names live in
    the BIDS *_channels.tsv sidecar, in the same order as the recording.
    """
    channels_tsv = pd.read_csv(channels_tsv_path, sep="\t")
    real_names = [_canonical_channel_name(name) for name in channels_tsv["name"]]
    if len(real_names) != len(raw.ch_names):
        raise ValueError(
            f"channels.tsv has {len(real_names)} rows but raw has {len(raw.ch_names)} channels"
        )
    raw.rename_channels(dict(zip(raw.ch_names, real_names)))

    ch_types = dict(zip(channels_tsv["name"].map(_canonical_channel_name), channels_tsv["type"]))
    type_map = {"EEG": "eeg", "EOG": "eog", "ECG": "ecg", "MISC": "misc"}
    raw.set_channel_types({ch: type_map[ch_types[ch]] for ch in raw.ch_names if ch_types[ch] in type_map})


def load_and_preprocess(
    vhdr_path: Path,
    channels_tsv_path: Path,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    epoch_duration: float = 4.0,
    amplitude_reject_uv: float = 150.0,
) -> tuple[mne.Epochs, dict]:
    """Load one recording and return cleaned, epoched data plus QC info."""
    raw = read_raw_brainvision_no_markers(vhdr_path, preload=True, verbose=False)
    _rename_channels_from_tsv(raw, channels_tsv_path)

    if "Misc" in raw.ch_names:
        raw.drop_channels(["Misc"])

    raw.set_montage("standard_1005")
    raw.set_eeg_reference("average", projection=False, ch_type="eeg", verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=["eeg", "eog", "ecg"], verbose=False)

    # Fixed component count (not a variance fraction like 0.99): keeps ICA
    # consistent across all subjects and avoids failures on recordings where
    # one component happens to dominate the explained variance.
    ica = mne.preprocessing.ICA(n_components=20, method="fastica", random_state=97, max_iter="auto")
    ica.fit(raw, verbose=False)

    exclude = set()
    if "VEOG" in raw.ch_names:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name="VEOG", verbose=False)
        exclude.update(eog_indices)
    if "ECG" in raw.ch_names:
        ecg_indices, _ = ica.find_bads_ecg(raw, ch_name="ECG", verbose=False)
        exclude.update(ecg_indices)
    ica.exclude = sorted(exclude)

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    epochs_template = mne.make_fixed_length_epochs(raw_clean, duration=epoch_duration, preload=True, verbose=False)
    n_epochs_total = len(epochs_template)

    # A handful of recordings are noisy enough that the standard 150 uV
    # threshold rejects every single epoch. Rather than losing the subject
    # entirely, progressively loosen the threshold until at least one epoch
    # survives, and report which threshold was actually used.
    for reject_uv in (amplitude_reject_uv, 250.0, 400.0):
        epochs = epochs_template.copy()
        epochs.drop_bad(reject=dict(eeg=reject_uv * 1e-6), verbose=False)
        if len(epochs) > 0:
            break
    n_epochs_kept = len(epochs)

    qc = {
        "n_epochs_total": n_epochs_total,
        "n_epochs_kept": n_epochs_kept,
        "reject_uv_used": reject_uv,
        "pct_epochs_kept": 100.0 * n_epochs_kept / n_epochs_total if n_epochs_total else 0.0,
        "n_ica_excluded": len(ica.exclude),
    }
    return epochs, qc


def extract_band_roi_features(epochs: mne.Epochs) -> dict:
    """Compute absolute + relative band power per ROI from cleaned epochs."""
    spectrum = epochs.compute_psd(method="welch", fmin=1, fmax=45, picks="eeg", verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs)
    mean_psd = psd.mean(axis=0)  # (n_channels, n_freqs)
    ch_names = spectrum.ch_names

    band_power = {}  # band -> array over channels
    for band, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        band_power[band] = np.trapezoid(mean_psd[:, mask], freqs[mask], axis=1)

    total_power = sum(band_power.values())  # (n_channels,) sum across the 5 contiguous bands

    features = {}
    for roi, roi_channels in ROIS.items():
        idx = [ch_names.index(ch) for ch in roi_channels if ch in ch_names]
        for band in BANDS:
            abs_power = float(band_power[band][idx].mean())
            rel_power = float((band_power[band][idx] / total_power[idx]).mean())
            features[f"{roi}_{band}_abs"] = abs_power
            features[f"{roi}_{band}_rel"] = rel_power

    return features
