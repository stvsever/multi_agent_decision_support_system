"""Dataset discovery and direct BrainVision loading for the FEP resting-EEG data.

These EEGLAB-exported BrainVision files store only generic labels (EEG001 ...
EEG064) in the .vhdr, while the true 10-10 names and channel types live in the
row-aligned ``*_channels.tsv`` sidecar. MNE's BrainVision reader would keep the
generic labels, so we read the VECTORIZED float32 binary directly and attach the
sidecar names positionally. This is the only load path that yields correctly
named electrodes for this dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from .config import DATASET_URLS, RAW_ROOT
from .montage import NON_CORTICAL, normalize_name

CHANNEL_TYPE_MAP = {"EEG": "eeg", "EOG": "eog", "ECG": "ecg", "MISC": "misc",
                    "EMG": "emg", "MEG": "misc"}


def discover_records() -> list[dict[str, Any]]:
    """List every resting recording across both datasets in a stable order."""
    records: list[dict[str, Any]] = []
    for dataset_id in sorted(DATASET_URLS):
        dataset_root = RAW_ROOT / dataset_id
        participants = pd.read_csv(dataset_root / "participants.tsv", sep="\t", dtype=str)
        meta = participants.set_index("participant_id").to_dict("index")
        for vhdr in sorted(dataset_root.glob("sub-*/eeg/*_task-*_eeg.vhdr")):
            participant_id = vhdr.parts[-3]
            stem = vhdr.name.removesuffix("_eeg.vhdr")
            records.append(
                {
                    "dataset_id": dataset_id,
                    "participant_id": participant_id,
                    "recording_id": f"{dataset_id}_{participant_id}",
                    "vhdr": vhdr,
                    "eeg": vhdr.with_name(f"{stem}_eeg.eeg"),
                    "channels_tsv": vhdr.with_name(f"{stem}_channels.tsv"),
                    "eeg_json": vhdr.with_name(f"{stem}_eeg.json"),
                    "source_type": meta.get(participant_id, {}).get("type", "n/a"),
                }
            )
    return records


def _parse_vhdr(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith((";", "[")):
            continue
        if "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def load_raw(record: dict[str, Any]) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """Load one recording as microvolt-scaled MNE Raw with true electrode names.

    Returns the Raw (in volts, MNE convention) and the parsed eeg.json sidecar.
    """
    channels = pd.read_csv(record["channels_tsv"], sep="\t", dtype=str)
    sidecar = json.loads(Path(record["eeg_json"]).read_text(encoding="utf-8"))
    header = _parse_vhdr(Path(record["vhdr"]))

    n_channels = int(header["NumberOfChannels"])
    if n_channels != len(channels):
        raise RuntimeError("BrainVision header and channels.tsv disagree on channel count")
    if header.get("DataOrientation", "").upper() != "VECTORIZED":
        raise RuntimeError("Only the audited VECTORIZED orientation is supported")
    if header.get("BinaryFormat", "").upper() != "IEEE_FLOAT_32":
        raise RuntimeError("Only the audited IEEE_FLOAT_32 format is supported")
    if Path(header["DataFile"]).name != Path(record["eeg"]).name:
        raise RuntimeError("Header DataFile does not match the discovered binary")

    names = [normalize_name(name) for name in channels["name"]]
    if len(set(names)) != len(names):
        raise RuntimeError("Electrode names are not unique after normalization")
    types = []
    for bids_type in channels["type"]:
        key = str(bids_type).upper()
        if key not in CHANNEL_TYPE_MAP:
            raise RuntimeError(f"Unsupported channel type: {bids_type}")
        types.append(CHANNEL_TYPE_MAP[key])

    sfreq = float(sidecar["SamplingFrequency"])
    if not np.isclose(1e6 / float(header["SamplingInterval"]), sfreq, rtol=1e-6):
        raise RuntimeError("Header and sidecar sampling frequencies disagree")

    eeg_path = Path(record["eeg"])
    bytes_per_frame = n_channels * np.dtype("<f4").itemsize
    if eeg_path.stat().st_size % bytes_per_frame:
        raise RuntimeError("EEG binary size is not a whole number of channel frames")
    n_samples = eeg_path.stat().st_size // bytes_per_frame
    if not np.isclose(n_samples / sfreq, float(sidecar["RecordingDuration"]),
                      atol=1.0 / sfreq):
        raise RuntimeError("Binary sample count and sidecar duration disagree")

    data_uv = np.memmap(eeg_path, dtype="<f4", mode="r",
                        shape=(n_channels, n_samples), order="C")
    data_v = np.asarray(data_uv, dtype=np.float64) * 1e-6
    info = mne.create_info(ch_names=names, sfreq=sfreq, ch_types=types, verbose="ERROR")
    raw = mne.io.RawArray(data_v, info, verbose="ERROR")
    raw.set_montage("standard_1005", on_missing="warn", verbose="ERROR")
    # Anything that is not a scalp cortical electrode is marked so it can never
    # enter a cortical feature or the average reference.
    raw.info["description"] = record["recording_id"]
    return raw, sidecar


def cortical_eeg_names(raw: mne.io.BaseRaw) -> list[str]:
    """EEG channels that are scalp cortical electrodes (mastoids and refs dropped)."""
    return [
        name
        for name in raw.ch_names
        if raw.get_channel_types(picks=[name])[0] == "eeg" and name not in NON_CORTICAL
    ]
