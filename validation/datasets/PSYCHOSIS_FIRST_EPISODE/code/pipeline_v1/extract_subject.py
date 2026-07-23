"""Runs preprocessing + all 8 feature groups for one subject, returns one flat row."""

import traceback
from pathlib import Path

from pipeline_v1.features.aperiodic import aperiodic_and_alpha_peak_features
from pipeline_v1.features.connectivity import compute_connectivity_matrices, connectivity_features
from pipeline_v1.features.entropy import entropy_features
from pipeline_v1.features.fractal import fractal_features
from pipeline_v1.features.graph import graph_features
from pipeline_v1.features.spectral import spectral_features
from pipeline_v1.preprocessing import preprocess_subject
from pipeline_v1.schema import canonical_feature_names

_, ALL_FEATURE_NAMES = canonical_feature_names()


def extract_subject(dataset: str, dataset_path: Path, participant_id: str) -> dict:
    """Never raises for expected issues; returns a dict with status/error info instead."""
    try:
        result = preprocess_subject(dataset, dataset_path, participant_id)
    except Exception as exc:
        return {
            "participant_id": participant_id,
            "dataset": dataset,
            "status": "failed",
            "error": f"{exc}\n{traceback.format_exc()}",
            "row": None,
            "qc_row": None,
        }

    qc = result["qc"]

    if qc.exclusion_status == "excluded":
        row = {"participant_id": participant_id, "dataset": dataset, **{name: None for name in ALL_FEATURE_NAMES}}
        return {
            "participant_id": participant_id,
            "dataset": dataset,
            "status": "excluded",
            "error": qc.exclusion_reason,
            "row": row,
            "qc_row": qc.as_row(),
        }

    try:
        features = {}
        features.update(spectral_features(result["epochs_interp"], dataset))
        b_c_features, bc_qc = aperiodic_and_alpha_peak_features(result["epochs_interp"], dataset)
        features.update(b_c_features)
        for key, value in bc_qc.items():
            setattr(qc, key, value)
        features.update(entropy_features(result["epochs_interp"], dataset))
        features.update(fractal_features(result["epochs_interp"], result["raw_interp_continuous"], dataset))

        matrices, valid_names = compute_connectivity_matrices(result["epochs_no_interp"], dataset)
        features.update(connectivity_features(matrices, valid_names))
        features.update(graph_features(matrices, valid_names))

        missing = set(ALL_FEATURE_NAMES) - set(features)
        extra = set(features) - set(ALL_FEATURE_NAMES)
        if missing or extra:
            raise AssertionError(f"Feature key mismatch - missing={missing}, extra={extra}")

    except Exception as exc:
        return {
            "participant_id": participant_id,
            "dataset": dataset,
            "status": "failed",
            "error": f"{exc}\n{traceback.format_exc()}",
            "row": None,
            "qc_row": qc.as_row(),
        }

    row = {"participant_id": participant_id, "dataset": dataset, **{name: features[name] for name in ALL_FEATURE_NAMES}}
    return {
        "participant_id": participant_id,
        "dataset": dataset,
        "status": "ok",
        "error": None,
        "row": row,
        "qc_row": qc.as_row(),
    }
