"""Frozen pipeline configuration and shared paths for the resting-EEG pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# validation/datasets/PSYCHOSIS_FIRST_EPISODE
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "pipeline_config.json"

RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed" / "eeg_fep_rest_v2"
SUBJECT_ROOT = PROCESSED_ROOT / "per_subject"
RESULTS_ROOT = ROOT / "results"
FIGURE_ROOT = RESULTS_ROOT / "figures"

DATASET_URLS = {
    "ds003944": "https://github.com/OpenNeuroDatasets/ds003944.git",
    "ds003947": "https://github.com/OpenNeuroDatasets/ds003947.git",
}

# Feature-family band definitions (Hz). Upper edges are exclusive except the last.
BANDS: dict[str, tuple[float, float]] = {
    "delta_1_4_hz": (1.0, 4.0),
    "theta_4_8_hz": (4.0, 8.0),
    "alpha_8_13_hz": (8.0, 13.0),
    "beta_13_30_hz": (13.0, 30.0),
    "low_gamma_30_45_hz": (30.0, 45.0),
}
BAND_NAMES = list(BANDS)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
