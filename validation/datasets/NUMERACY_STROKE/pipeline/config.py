"""
Dataset-specific configuration for NUMERACY_STROKE (OpenNeuro ds006533).

This is the ONLY place NUMERACY_STROKE-specific knowledge lives. Reusable
lesion-mask/atlas helpers live in ``validation/common/lesion.py`` and stay
dataset-agnostic.

Prediction task: precise vs. approximate numeracy performance
(``precise_numeracy``, ``approximate_numeracy``) in left-hemisphere chronic
stroke survivors, from demographics + whole-brain lesion-overlap features.
"""

from __future__ import annotations

from pathlib import Path

DATASET_NAME = "NUMERACY_STROKE"
DATASET_LABEL = "Precise vs. Approximate Numeracy in Stroke Participants (OpenNeuro ds006533)"
ACCESSION = "ds006533"
LICENSE = "CC0"

ROOT = Path(__file__).resolve().parents[1]                    # datasets/NUMERACY_STROKE
DATA_RAW = ROOT / "data" / "raw" / ACCESSION
DERIVATIVES_DIR = DATA_RAW / "derivatives"
LESION_MASKS_DIR = DERIVATIVES_DIR / "lesion_masks"
PARTICIPANTS_TSV = DATA_RAW / "participants.tsv"
PROCESSED_DIR = ROOT / "data" / "processed"
ATLAS_CACHE_DIR = ROOT / "data" / "_cache" / "atlases"        # gitignored downloads

# ---------------------------------------------------------------- targets
TARGET_COLUMNS = ["approximate_numeracy", "precise_numeracy"]

# ------------------------------------------------------------- atlas params
N_ROIS = 1000              # Schaefer-2018 cortical parcels ("high-res")
YEO_NETWORKS = 7
RESOLUTION_MM = 1
TIAN_SCALE = "S4"           # subcortex, 54 bilateral parcels
CEREBELLAR_ATLAS = "NettekovenAsym128"  # cerebellum, up to 128 parcels

# Reference subject used to define the shared MNI grid the atlas is resampled
# onto (all subjects already share this exact grid - see pipeline README).
REFERENCE_SUBJECT = "sub-001"


def all_subjects() -> list[str]:
    """Every participant_id in participants.tsv, in file order."""
    import pandas as pd
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    return df["participant_id"].astype(str).tolist()


def lesion_mask_path(subject: str) -> Path:
    return LESION_MASKS_DIR / subject / "anat" / f"{subject}_lesion-mask.nii.gz"
