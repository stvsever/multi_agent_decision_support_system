"""Orchestrates Stages 0-7 (discovery through epoching) for one subject."""

from pathlib import Path

from pipeline_v1 import bad_channels, filtering, ica_cleaning, io_bids, loading
from pipeline_v1.epoching import build_epochs
from pipeline_v1.qc import SubjectQC, evaluate_inclusion_gates


def preprocess_subject(dataset: str, dataset_path: Path, participant_id: str) -> dict:
    """Run the full preprocessing pipeline for one subject. Never raises for expected
    data-quality issues; QC.exclusion_status/reason records the outcome instead.
    """
    qc = SubjectQC(dataset_id=dataset, source_participant_id_private=participant_id)

    record = io_bids.discover_subject(dataset, dataset_path, participant_id)
    sidecar = io_bids.validate_subject_record(record)
    qc.recording_duration_s = float(sidecar["RecordingDuration"])

    raw, channels_table = loading.load_and_reconcile(record)
    unit_audit = loading.audit_and_fix_units(raw)

    branches = filtering.build_branches(raw)

    bad_result = bad_channels.detect_bad_channels(branches["main"])

    raw_clean, ica_qc = ica_cleaning.fit_and_clean(
        branches["main"], branches["ica_fit"], bad_result["bad_channels"]
    )
    for key, value in ica_qc.items():
        setattr(qc, key, value)

    epoch_result = build_epochs(raw_clean, dataset, bad_result["bad_channels"])
    for key, value in epoch_result["qc"].items():
        if hasattr(qc, key):
            setattr(qc, key, value)
        else:
            qc.extra[key] = value
    qc.usable_fraction = (
        qc.usable_duration_s / qc.recording_duration_s
        if qc.usable_duration_s and qc.recording_duration_s
        else None
    )
    qc.extra["unit_audit"] = unit_audit
    qc.extra["bad_channel_reasons"] = bad_result["reasons"]

    included, reason = evaluate_inclusion_gates(qc)
    qc.exclusion_status = "included" if included else "excluded"
    qc.exclusion_reason = reason

    return {
        "record": record,
        "channels_table": channels_table,
        "raw_interp_continuous": epoch_result["raw_interp_continuous"],
        "epochs_interp": epoch_result["epochs_interp"],
        "epochs_no_interp": epoch_result["epochs_no_interp"],
        "microstate_raw": branches["microstate"],
        "bad_cortical_channels": epoch_result["bad_cortical_channels"],
        "qc": qc,
    }
