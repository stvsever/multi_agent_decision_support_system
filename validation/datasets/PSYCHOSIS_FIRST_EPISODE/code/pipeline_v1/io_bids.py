"""Stage 0/1: BIDS discovery, structural validation, and provenance hashing."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

AUDITED_SAMPLING_RATES_HZ = (1000.0, 3000.00030000003)
SAMPLING_RATE_RTOL = 1e-6
EXPECTED_CHANNEL_COUNT = 64
EXPECTED_TYPE_COUNTS = {"EEGChannelCount": 61, "EOGChannelCount": 1, "ECGChannelCount": 1, "MiscChannelCount": 1}


@dataclass
class SubjectRecord:
    dataset: str
    participant_id: str
    vhdr_path: Path
    eeg_data_path: Path
    channels_tsv_path: Path
    eeg_json_path: Path


class BidsValidationError(Exception):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_subject(dataset: str, dataset_path: Path, participant_id: str) -> SubjectRecord:
    """Find and structurally validate one subject's recording. Case-aware task-Rest/task-rest."""
    eeg_dir = dataset_path / participant_id / "eeg"
    vhdr_matches = sorted(eeg_dir.glob("*_task-*_eeg.vhdr"))
    if len(vhdr_matches) != 1:
        raise BidsValidationError(
            f"{dataset}/{participant_id}: expected exactly one BrainVision header, found {len(vhdr_matches)}"
        )
    vhdr_path = vhdr_matches[0]
    base = vhdr_path.name.removesuffix("_eeg.vhdr")
    channels_tsv_path = vhdr_path.with_name(f"{base}_channels.tsv")
    eeg_json_path = vhdr_path.with_name(f"{base}_eeg.json")

    if not channels_tsv_path.is_file():
        raise BidsValidationError(f"{dataset}/{participant_id}: missing {channels_tsv_path.name}")
    if not eeg_json_path.is_file():
        raise BidsValidationError(f"{dataset}/{participant_id}: missing {eeg_json_path.name}")

    header_text = vhdr_path.read_text(encoding="utf-8")
    data_filename = None
    for line in header_text.splitlines():
        if line.startswith("DataFile="):
            data_filename = line.split("DataFile=", 1)[1].strip()
            break
    if data_filename is None:
        raise BidsValidationError(f"{dataset}/{participant_id}: no DataFile= line in {vhdr_path.name}")
    eeg_data_path = vhdr_path.parent / data_filename
    if not eeg_data_path.is_file():
        raise BidsValidationError(f"{dataset}/{participant_id}: referenced binary {eeg_data_path.name} missing")

    return SubjectRecord(dataset, participant_id, vhdr_path, eeg_data_path, channels_tsv_path, eeg_json_path)


def validate_subject_record(record: SubjectRecord) -> dict:
    """Validate channel counts, sampling rate, and type counts. Returns parsed sidecar dict."""
    channels = pd.read_csv(record.channels_tsv_path, sep="\t", dtype=str)
    if len(channels) != EXPECTED_CHANNEL_COUNT:
        raise BidsValidationError(
            f"{record.dataset}/{record.participant_id}: channels.tsv has {len(channels)} rows, "
            f"expected {EXPECTED_CHANNEL_COUNT}"
        )

    n_channels_in_header = None
    for line in record.vhdr_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("NumberOfChannels="):
            n_channels_in_header = int(line.split("=", 1)[1].strip())
            break
    if n_channels_in_header != EXPECTED_CHANNEL_COUNT:
        raise BidsValidationError(
            f"{record.dataset}/{record.participant_id}: header declares {n_channels_in_header} channels, "
            f"expected {EXPECTED_CHANNEL_COUNT}"
        )

    sidecar = json.loads(record.eeg_json_path.read_text(encoding="utf-8"))
    native_sfreq = float(sidecar["SamplingFrequency"])
    if not any(abs(native_sfreq - r) <= SAMPLING_RATE_RTOL * r for r in AUDITED_SAMPLING_RATES_HZ):
        raise BidsValidationError(
            f"{record.dataset}/{record.participant_id}: unexpected sampling frequency {native_sfreq}"
        )

    for key, expected in EXPECTED_TYPE_COUNTS.items():
        if int(sidecar[key]) != expected:
            raise BidsValidationError(
                f"{record.dataset}/{record.participant_id}: {key}={sidecar[key]}, expected {expected}"
            )

    return sidecar


def assert_participant_uniqueness(all_records: list[SubjectRecord]) -> None:
    seen = {}
    for r in all_records:
        key = r.participant_id
        if key in seen:
            raise BidsValidationError(
                f"Duplicate participant_id {key!r} across {seen[key]} and {r.dataset} - "
                "expected disjoint participant tables per source README"
            )
        seen[key] = r.dataset
