"""Stage 4: crop/trim, three filtered branches, and harmonized resampling to 250 Hz."""

import mne

CROP_DURATION_S = 300.0
TRIM_EDGE_S = 3.0
TARGET_SFREQ = 250.0

BRANCH_PASSBANDS = {
    "main": (0.5, 45.0),
    "ica_fit": (1.0, 45.0),
    "microstate": (2.0, 20.0),
}


def crop_and_trim(raw: mne.io.Raw) -> mne.io.Raw:
    """Crop to the first 300 s (if available), then trim 3 s off each edge -> ~294 s."""
    raw = raw.copy()
    duration = raw.times[-1]
    crop_to = min(CROP_DURATION_S, duration)
    raw.crop(tmin=0.0, tmax=crop_to)
    trimmed_duration = raw.times[-1]
    if trimmed_duration > 2 * TRIM_EDGE_S:
        raw.crop(tmin=TRIM_EDGE_S, tmax=trimmed_duration - TRIM_EDGE_S)
    return raw


def make_branch(raw_cropped: mne.io.Raw, branch: str) -> mne.io.Raw:
    """Filter (at native rate, for correct antialiasing) then resample to 250 Hz."""
    l_freq, h_freq = BRANCH_PASSBANDS[branch]
    picks = ["eeg", "eog", "ecg"]
    out = raw_cropped.copy()
    out.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, method="fir", phase="zero", verbose=False)
    if out.info["sfreq"] > TARGET_SFREQ:
        out.resample(TARGET_SFREQ, verbose=False)
    return out


def build_branches(raw: mne.io.Raw) -> dict[str, mne.io.Raw]:
    """Return the three filtered/resampled branches used downstream."""
    cropped = crop_and_trim(raw)
    return {branch: make_branch(cropped, branch) for branch in BRANCH_PASSBANDS}
