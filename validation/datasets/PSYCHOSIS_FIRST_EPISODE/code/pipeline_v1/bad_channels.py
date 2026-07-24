"""Stage 5: automated bad-channel detection (pyprep-based)."""

import mne
from pyprep.find_noisy_channels import NoisyChannels

RANDOM_STATE = 97


def detect_bad_channels(raw_main_branch: mne.io.Raw) -> dict:
    """Run pyprep's NoisyChannels on the EEG picks of the main (0.5-45 Hz) branch.

    Covers: bad-by-deviation (extreme amplitude/variance), bad-by-correlation and
    bad-by-ransac (poor predictability from neighbors), bad-by-hf-noise (excessive
    high-frequency/line contamination), bad-by-flat (near-zero variance), and
    bad-by-dropout. Bridging is checked separately if this pyprep version exposes
    it; the pipeline does not fail if it does not.
    """
    eeg_raw = raw_main_branch.copy().pick(picks="eeg")
    noisy = NoisyChannels(eeg_raw, random_state=RANDOM_STATE)
    noisy.find_all_bads(channel_wise=False)

    reasons = {
        "bad_by_deviation": list(noisy.bad_by_deviation),
        "bad_by_correlation": list(noisy.bad_by_correlation),
        "bad_by_hf_noise": list(noisy.bad_by_hf_noise),
        "bad_by_flat": list(noisy.bad_by_flat),
        "bad_by_ransac": list(getattr(noisy, "bad_by_ransac", [])),
        "bad_by_dropout": list(getattr(noisy, "bad_by_dropout", [])),
    }

    bridged = []
    find_bridged = getattr(noisy, "find_bad_by_bridge", None)
    if callable(find_bridged):
        try:
            find_bridged()
            bridged = list(getattr(noisy, "bad_by_bridge", []))
            reasons["bad_by_bridge"] = bridged
        except Exception:
            pass  # bridging check is best-effort; absence does not fail the pipeline

    all_bad = sorted({str(ch) for ch in noisy.get_bads()} | {str(ch) for ch in bridged})
    return {"bad_channels": all_bad, "reasons": reasons}
