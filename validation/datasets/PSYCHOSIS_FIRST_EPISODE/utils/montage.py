"""Harmonized montage for the two OpenNeuro first-episode-psychosis EEG datasets.

Both recordings (ds003944, ds003947) were collected on an Elekta Neuromag system
with an EasyCap layout, but the two studies wired slightly different 61-electrode
subsets. Their scalp electrodes intersect in exactly 49 cortical positions (plus
the M2 mastoid, which we drop). Computing every regional, global, connectivity and
graph feature on this single shared 49-channel montage is what makes the values
comparable across the two acquisitions rather than confounded by layout.

The .vhdr headers carry only generic labels (EEG001 ... EEG064); the true 10-10
names live in the *_channels.tsv sidecars, row-aligned to the binary. The loader
applies that positional mapping (see utils.io), after which every recording speaks
standard 10-10 names and this module can address electrodes by name.
"""

from __future__ import annotations

import numpy as np

# Case reconciliation from source labels to MNE standard_1005 spelling.
CASE_FIXES = {
    "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz",
    "OZ": "Oz", "AFZ": "AFz", "FCZ": "FCz",
    "CPZ": "CPz", "POZ": "POz", "IZ": "Iz",
}

# Non-cortical channels that must never enter a cortical feature or the reference.
NON_CORTICAL = {"VEOG", "HEOG", "EOG", "Misc", "MISC", "ECG", "EMG",
                "M1", "M2", "A1", "A2", "TP9", "TP10"}

# The 49 electrodes present in BOTH datasets (mastoids excluded). Derived
# empirically from every *_channels.tsv in both datasets; all 49 carry
# standard_1005 coordinates.
COMMON_CORTICAL = [
    "Fp1", "Fp2", "AF7", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FC5", "FC1", "FC2", "FC6", "FT7", "FT8",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T7", "T8", "TP7",
    "CP3", "CP1", "CP2", "CP4", "TP8", "P7", "P5", "P3", "Pz", "P4",
    "P6", "P8", "PO7", "PO3", "PO4", "PO8", "O1", "Oz", "O2",
]
assert len(COMMON_CORTICAL) == 49 and len(set(COMMON_CORTICAL)) == 49

# Ten lateralized cortical regions of interest, balanced left/right, defined once
# over the shared montage. Midline electrodes (Fz, Cz, Pz, Oz) contribute to the
# global scope only, so left/right asymmetry contrasts stay strictly lateralized.
REGIONS: dict[str, list[str]] = {
    "frontal_left":   ["Fp1", "AF7", "AF3", "F7", "F5", "F3", "F1"],
    "frontal_right":  ["Fp2", "AF4", "F2", "F4", "F6", "F8"],
    "central_left":   ["FC5", "FC1", "C5", "C3", "C1"],
    "central_right":  ["FC6", "FC2", "C6", "C4", "C2"],
    "temporal_left":  ["FT7", "T7", "TP7"],
    "temporal_right": ["FT8", "T8", "TP8"],
    "parietal_left":  ["CP3", "CP1", "P7", "P5", "P3"],
    "parietal_right": ["CP4", "CP2", "P4", "P6", "P8"],
    "occipital_left": ["PO7", "PO3", "O1"],
    "occipital_right":["PO8", "PO4", "O2"],
}
MIDLINE = ["Fz", "Cz", "Pz", "Oz"]

# Ordered scope and node vocabularies used throughout feature naming.
SCOPES = ["global", *REGIONS.keys()]
NODES = list(REGIONS.keys())
ASYMMETRY_REGIONS = ["frontal", "central", "temporal", "parietal", "occipital"]

# Sanity: the ten regions plus four midline electrodes tile the 49 exactly once.
_covered = [ch for region in REGIONS.values() for ch in region] + MIDLINE
assert sorted(_covered) == sorted(COMMON_CORTICAL), "region map must tile the 49 electrodes"
assert len(_covered) == 49


def normalize_name(name: str) -> str:
    """Map a source electrode label to standard_1005 spelling."""
    stripped = str(name).strip()
    return CASE_FIXES.get(stripped.upper(), stripped)


def scope_indices(names: list[str], scope: str) -> list[int]:
    """Return positional indices of the channels that belong to a scope.

    ``global`` returns every provided channel; a region returns only its members
    that are actually present (all present when the montage is the shared 49).
    """
    if scope == "global":
        return list(range(len(names)))
    members = set(REGIONS[scope])
    return [i for i, name in enumerate(names) if name in members]


# --------------------------------------------------------------------------- #
# Canonical A-D microstate ordering.
#
# pycrostates returns four data-driven maps in an arbitrary order and sign. We
# relabel them to the canonical Koenig A-D scheme from each map's topographic
# orientation, which is exactly what those four archetypes encode:
#   A  left-posterior  to right-anterior diagonal   (gradient axis ~ +45 deg)
#   B  right-posterior to left-anterior  diagonal   (gradient axis ~ -45 deg)
#   C  anterior-posterior, near midline             (gradient axis ~ vertical)
#   D  fronto-central, radial/central               (poorly fit by any linear axis)
# The rule below is deterministic and montage-based, so the class_a..class_d
# columns carry their conventional meaning rather than arbitrary labels.
# --------------------------------------------------------------------------- #
def canonical_microstate_order(maps: np.ndarray, positions: np.ndarray) -> list[int]:
    """Order four microstate maps into canonical A, B, C, D.

    Parameters
    ----------
    maps : (4, n_channels) template topographies (sign arbitrary).
    positions : (n_channels, 2) electrode coordinates, columns [x_right, y_anterior].

    Returns
    -------
    list[int]
        Indices into ``maps`` giving the A, B, C, D order.
    """
    x = positions[:, 0].astype(float)
    y = positions[:, 1].astype(float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    design = np.column_stack([x, y, np.ones_like(x)])

    angles: list[float] = []
    linear_r2: list[float] = []
    for m in maps:
        m = np.asarray(m, dtype=float)
        m = m - m.mean()
        beta, *_ = np.linalg.lstsq(design, m, rcond=None)
        fitted = design @ beta
        ss_res = float(np.sum((m - fitted) ** 2))
        ss_tot = float(np.sum(m ** 2)) + 1e-12
        linear_r2.append(1.0 - ss_res / ss_tot)
        # Gradient orientation as an axis in [0, 180) degrees; sign of the map
        # (and hence of the gradient) is arbitrary, so fold to a half circle.
        angle = np.degrees(np.arctan2(beta[1], beta[0])) % 180.0
        angles.append(angle)

    remaining = list(range(4))
    # D = the least linearly oriented (most radial / central) topography.
    d = min(remaining, key=lambda i: linear_r2[i])
    remaining.remove(d)
    # C = the most anterior-posterior (axis nearest vertical, 90 deg).
    c = min(remaining, key=lambda i: abs(angles[i] - 90.0))
    remaining.remove(c)
    # A vs B by the diagonal sign of the remaining two: A leans toward +45, B -45.
    # A signed diagonal score in (-90, 90]: positive -> A, negative -> B.
    def diagonal(i: int) -> float:
        a = angles[i]
        return a if a <= 90.0 else a - 180.0
    a, b = sorted(remaining, key=diagonal, reverse=True)
    return [a, b, c, d]
