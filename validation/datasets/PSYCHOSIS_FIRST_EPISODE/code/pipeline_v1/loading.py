"""Stage 2: safe BrainVision loading, positional channel reconciliation, unit audit."""

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eeg_pipeline import read_raw_brainvision_no_markers  # noqa: E402

from pipeline_v1.io_bids import SubjectRecord  # noqa: E402

CHANNEL_TYPE_MAP = {"EEG": "eeg", "EOG": "eog", "ECG": "ecg", "MISC": "misc"}


class UnitAuditError(Exception):
    pass


def _canonical_casing_map(raw_names: list[str], eeg_type_names: set[str]) -> dict[str, str]:
    """Case-insensitively resolve EEG channel names to standard_1005's canonical casing.

    ds003944 needs FP1/FPz/FP2 -> Fp1/Fpz/Fp2; ds003947 needs OZ -> Oz. Rather than
    hardcode per-dataset casing fixes, resolve generically against the montage so
    this keeps working if either dataset's channel-name casing changes upstream.
    """
    montage = mne.channels.make_standard_montage("standard_1005")
    lower_to_canonical = {name.lower(): name for name in montage.ch_names}
    fixes = {}
    for name in raw_names:
        if name not in eeg_type_names:
            continue
        canonical = lower_to_canonical.get(name.lower())
        if canonical is None:
            raise UnitAuditError(f"Channel {name!r} not found in standard_1005 montage (even case-insensitively)")
        if canonical != name:
            fixes[name] = canonical
    return fixes


def load_and_reconcile(record: SubjectRecord) -> tuple[mne.io.Raw, pd.DataFrame]:
    """Load raw BrainVision data and rename EEG001..EEG064 to real 10-10 labels by row position."""
    raw = read_raw_brainvision_no_markers(record.vhdr_path, preload=True, verbose=False)
    channels = pd.read_csv(record.channels_tsv_path, sep="\t", dtype=str)

    if len(raw.ch_names) != len(channels):
        raise UnitAuditError(
            f"Header has {len(raw.ch_names)} channels but channels.tsv has {len(channels)} - "
            "cannot reconcile positionally"
        )

    bids_names = list(channels["name"])
    if len(set(bids_names)) != len(bids_names):
        raise UnitAuditError(f"Duplicate channel label in {record.channels_tsv_path}")

    # Positional (row-order) reconciliation - never alphabetical, per spec 3.1(1).
    raw.rename_channels(dict(zip(raw.ch_names, bids_names, strict=True)))

    eeg_type_names = {name for name, t in zip(bids_names, channels["type"]) if t.upper() == "EEG"}
    casing_fixes = _canonical_casing_map(raw.ch_names, eeg_type_names)
    if casing_fixes:
        raw.rename_channels(casing_fixes)

    type_map = {}
    for name, bids_type in zip(bids_names, channels["type"]):
        canonical_name = casing_fixes.get(name, name)
        try:
            type_map[canonical_name] = CHANNEL_TYPE_MAP[bids_type.upper()]
        except KeyError as exc:
            raise UnitAuditError(f"Unsupported channel type {bids_type!r} for {name}") from exc
    raw.set_channel_types(type_map, verbose=False)
    raw.set_montage("standard_1005", on_missing="warn", verbose=False)

    return raw, channels


def audit_and_fix_units(raw: mne.io.Raw) -> dict:
    """Verify EEG data is plausibly in volts (MNE's internal unit); rescale if not.

    channels.tsv declares microvolts, but BrainVision channel-resolution fields
    are sparse, so MNE's applied scaling isn't guaranteed correct. Check robust
    amplitude statistics on the actual loaded data before trusting it.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=eeg_picks)  # volts, per MNE convention
    median_abs_uv = np.median(np.abs(data)) * 1e6
    p99_abs_uv = np.percentile(np.abs(data), 99) * 1e6

    # Plausible scalp EEG: median absolute amplitude on the order of single-digit
    # to low tens of microvolts; a 1e6 unit error would put this at ~1e-5 or ~1e7.
    if median_abs_uv < 1e-2:
        raw.apply_function(lambda x: x * 1e6, picks=eeg_picks, channel_wise=False)
        applied_correction = "multiplied_by_1e6"
    elif median_abs_uv > 1e4:
        raw.apply_function(lambda x: x * 1e-6, picks=eeg_picks, channel_wise=False)
        applied_correction = "multiplied_by_1e-6"
    else:
        applied_correction = "none"

    data_after = raw.get_data(picks=eeg_picks)
    return {
        "median_abs_amplitude_uv_before": float(median_abs_uv),
        "p99_abs_amplitude_uv_before": float(p99_abs_uv),
        "median_abs_amplitude_uv_after": float(np.median(np.abs(data_after)) * 1e6),
        "unit_correction_applied": applied_correction,
    }
