"""Group G: functional connectivity (225) - spec section 7.7.

Debiased squared weighted phase-lag index (wpli2_debiased) between the 10
frozen region node signals (built via regions.fit_node_spatial_weights on
original, non-interpolated good channels only - never on epochs_interp).

Note: with fixed 4 s epochs, mne-connectivity itself warns that frequencies
near the bottom of the delta band (~1 Hz) fall short of the 5-cycles-per-
epoch guideline for a fully reliable spectral estimate (5 Hz would need a
5 s window). This is an inherent tradeoff of the spec's fixed 4 s epoch
design (shared with every other feature group), not a bug here - the delta-
band edges should be read with that caveat rather than "fixed" by locally
lengthening epochs just for this group.

`compute_connectivity_matrices` is the shared computation also consumed by
Group H (graph theory), so the (relatively cheap, but non-trivial) multitaper
connectivity estimate is only computed once per subject.
"""

from itertools import combinations

import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs

from pipeline_v1 import regions

BANDS = {
    "delta_1_4_hz": (1.0, 4.0),
    "theta_4_8_hz": (4.0, 8.0),
    "alpha_8_13_hz": (8.0, 13.0),
    "beta_13_30_hz": (13.0, 30.0),
    "low_gamma_30_45_hz": (30.0, 45.0),
}
EDGES = list(combinations(regions.NODES, 2))  # fixed order, matches spec section 7.7.2


def _build_node_epochs(epochs_no_interp: mne.Epochs, dataset: str) -> tuple[np.ndarray | None, dict]:
    data = epochs_no_interp.get_data(picks="eeg")  # (n_epochs, n_channels, n_times), volts
    ch_names = [epochs_no_interp.ch_names[i] for i in mne.pick_types(epochs_no_interp.info, eeg=True)]

    all_time_concat = data.transpose(1, 0, 2).reshape(len(ch_names), -1)  # (n_channels, n_epochs*n_times)
    weights = regions.fit_node_spatial_weights(all_time_concat, ch_names, dataset)

    valid_nodes = [n for n in regions.NODES if weights[n] is not None]
    if len(valid_nodes) < 2:
        return None, weights

    n_epochs, _, n_times = data.shape
    node_epochs = np.full((n_epochs, len(regions.NODES), n_times), np.nan)
    for ei in range(n_epochs):
        node_signals = regions.apply_node_spatial_weights(data[ei], ch_names, dataset, weights)
        for ni, node in enumerate(regions.NODES):
            if node_signals[node] is not None:
                node_epochs[ei, ni, :] = node_signals[node]

    return node_epochs, weights


def compute_connectivity_matrices(
    epochs_no_interp: mne.Epochs, dataset: str
) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    """Returns ({band: (n_valid, n_valid) symmetric matrix, zero diagonal}, valid_node_names)
    or (None, []) if fewer than 2 nodes have enough good channels for connectivity."""
    sfreq = epochs_no_interp.info["sfreq"]
    node_epochs, weights = _build_node_epochs(epochs_no_interp, dataset)
    if node_epochs is None:
        return None, []

    valid_node_idx = {n: i for i, n in enumerate(regions.NODES) if weights[n] is not None}
    valid_names = list(valid_node_idx.keys())
    valid_data = node_epochs[:, [valid_node_idx[n] for n in valid_names], :]

    fmin = tuple(b[0] for b in BANDS.values())
    fmax = tuple(b[1] for b in BANDS.values())
    con = spectral_connectivity_epochs(
        valid_data,
        method="wpli2_debiased",
        mode="multitaper",
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        mt_adaptive=True,
        verbose=False,
    )
    dense = con.get_data(output="dense")  # (n_valid, n_valid, n_bands), lower-triangle filled
    dense_sym = dense + dense.transpose(1, 0, 2)  # symmetrize (diagonal stays 0)

    matrices = {band: dense_sym[:, :, band_i] for band_i, band in enumerate(BANDS)}
    return matrices, valid_names


def connectivity_features(matrices: dict[str, np.ndarray] | None, valid_names: list[str]) -> dict:
    """Builds the 225 flat Group G features from compute_connectivity_matrices' output."""
    local_idx = {n: i for i, n in enumerate(valid_names)}
    features = {}
    for band in BANDS:
        for node_a, node_b in EDGES:
            key = f"G_functional_connectivity__wpli2_debiased__{band}__{node_a}_to_{node_b}"
            if matrices is not None and node_a in local_idx and node_b in local_idx:
                features[key] = float(matrices[band][local_idx[node_a], local_idx[node_b]])
            else:
                features[key] = None
    return features
