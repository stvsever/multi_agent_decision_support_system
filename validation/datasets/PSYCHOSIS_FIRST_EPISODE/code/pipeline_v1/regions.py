"""Canonical 10-node + global scalp-region mapping for ds003944 and ds003947.

Also provides the two region-reduction methods used downstream:
  - `median_region_value`: for univariate (spectral/entropy/fractal/...) features -
    compute the measure per channel first, then take the median across a region's
    channels. Avoids phase cancellation and prevents electrode-dense regions from
    dominating.
  - `fit_node_spatial_weights` / `apply_node_spatial_weights`: for connectivity -
    fit one first-principal-component spatial weight vector per node from all
    retained time in a subject (original, non-interpolated good channels only),
    sign-oriented to the region mean, then freeze and apply identically to every
    epoch. Never refit per epoch - that would make epochs spatially incomparable.


ds003944 and ds003947 use genuinely different 61-channel 10-10 layouts (not
just different casing) - verified directly against both datasets' real
channels.tsv files, not assumed from either dataset's README. The channel
lists below are hardcoded per dataset for that reason; do not merge them.

TP9 is excluded from ds003944 only, because that dataset's eeg.json declares
it as the recording reference (EEGReference: "TP9") and it is present as a
(flat/near-flat) recorded channel. ds003947's eeg.json also declares TP9 as
the reference, but ds003947 never recorded a TP9 channel at all, so there is
nothing to exclude there. This is why ds003944's temporal_left node has one
fewer channel (3) than temporal_right (4): TP7 remains, but the would-be TP9
member of that node is gone.
"""

import numpy as np

SCOPES = [
    "global",
    "frontal_left", "frontal_right",
    "central_left", "central_right",
    "temporal_left", "temporal_right",
    "parietal_left", "parietal_right",
    "occipital_left", "occipital_right",
]

NODES = SCOPES[1:]

HOMOLOGOUS_PAIRS = {
    "frontal": ("frontal_left", "frontal_right"),
    "central": ("central_left", "central_right"),
    "temporal": ("temporal_left", "temporal_right"),
    "parietal": ("parietal_left", "parietal_right"),
    "occipital": ("occipital_left", "occipital_right"),
}

# Per-dataset node -> channel-name membership (canonical MNE standard_1005 casing).
REGION_CHANNELS = {
    "ds003944": {
        "frontal_left": ["Fp1", "AF7", "AF3", "F7", "F5", "F3", "F1"],
        "frontal_right": ["Fp2", "AF4", "AF6", "F2", "F4", "F6", "F8"],
        "central_left": ["FT9", "FT7", "FC5", "FC1", "C5", "C3", "C1"],
        "central_right": ["FC2", "FC6", "FT8", "FT10", "C2", "C4", "C6"],
        "temporal_left": ["T9", "T7", "TP7"],
        "temporal_right": ["T8", "T10", "TP8", "TP10"],
        "parietal_left": ["CP3", "CP1", "P7", "P5", "P3", "P1"],
        "parietal_right": ["CP2", "CP4", "P2", "P4", "P6", "P8"],
        "occipital_left": ["PO7", "PO3", "O1"],
        "occipital_right": ["PO4", "PO8", "O2"],
    },
    "ds003947": {
        "frontal_left": ["Fp1", "AF7", "AF3", "F7", "F5", "F3", "F1"],
        "frontal_right": ["Fp2", "AF4", "AF8", "F2", "F4", "F6", "F8"],
        "central_left": ["FT7", "FC5", "FC3", "FC1", "C5", "C3", "C1"],
        "central_right": ["FC2", "FC4", "FC6", "FT8", "C2", "C4", "C6"],
        "temporal_left": ["T7", "TP7", "P9"],
        "temporal_right": ["T8", "TP8", "P10"],
        "parietal_left": ["CP5", "CP3", "CP1", "P7", "P5", "P3"],
        "parietal_right": ["CP2", "CP4", "CP6", "P4", "P6", "P8"],
        "occipital_left": ["PO9", "PO7", "PO3", "O1"],
        "occipital_right": ["PO4", "PO8", "PO10", "O2"],
    },
}

# Midline channels: contribute to `global` only, never duplicated into a
# hemisphere (duplication would artificially inflate homologous connectivity).
MIDLINE_CHANNELS = {
    "ds003944": ["Fpz", "Fz", "Cz", "Pz", "Oz", "Iz"],
    "ds003947": ["AFz", "Fz", "FCz", "Cz", "Pz", "Oz"],
}

# Present in the recording but never used as a cortical feature channel.
EXCLUDED_CHANNELS = {
    "ds003944": ["TP9", "M2", "VEOG", "ECG", "Misc"],
    "ds003947": ["M2", "VEOG", "ECG", "Misc"],
}


def cortical_channels(dataset: str) -> list[str]:
    """All channels eligible for univariate (global/regional) feature computation."""
    region_chans = [ch for chans in REGION_CHANNELS[dataset].values() for ch in chans]
    return region_chans + MIDLINE_CHANNELS[dataset]


def channel_to_node(dataset: str) -> dict[str, str]:
    """Map each lateral channel to its single node (does not include midline)."""
    mapping = {}
    for node, chans in REGION_CHANNELS[dataset].items():
        for ch in chans:
            mapping[ch] = node
    return mapping


def scope_channels(dataset: str, scope: str) -> list[str]:
    """Channels contributing to a given univariate scope. `global` = all cortical channels."""
    if scope == "global":
        return cortical_channels(dataset)
    return REGION_CHANNELS[dataset][scope]


def median_region_value(per_channel_values: dict[str, float], dataset: str, scope: str) -> float | None:
    """Median of a per-channel measure across a scope's available channels."""
    chans = scope_channels(dataset, scope)
    values = [per_channel_values[ch] for ch in chans if ch in per_channel_values]
    if not values:
        return None
    return float(np.median(values))


def fit_node_spatial_weights(
    data: np.ndarray, ch_names: list[str], dataset: str, min_channels: int = 2
) -> dict[str, np.ndarray | None]:
    """Fit one frozen PC1 spatial weight vector per node from continuous good-channel data.

    `data` is (n_channels, n_times) restricted to original (non-interpolated) good
    channels only, `ch_names` its channel order. Returns {node: weight_vector or
    None if fewer than `min_channels` good channels are available for that node}.
    """
    name_to_idx = {name: i for i, name in enumerate(ch_names)}
    weights = {}
    for node, node_channels in REGION_CHANNELS[dataset].items():
        idx = [name_to_idx[ch] for ch in node_channels if ch in name_to_idx]
        if len(idx) < min_channels:
            weights[node] = None
            continue
        sub = data[idx, :]
        sub_centered = sub - sub.mean(axis=1, keepdims=True)
        # PC1 spatial weights via SVD of the (channels x time) matrix.
        u, _, _ = np.linalg.svd(sub_centered, full_matrices=False)
        w = u[:, 0]
        region_mean = sub_centered.mean(axis=0)
        pc1_projection = w @ sub_centered
        if np.corrcoef(pc1_projection, region_mean)[0, 1] < 0:
            w = -w
        full_weight = np.zeros(len(node_channels))
        for local_i, ch in enumerate(node_channels):
            if ch in name_to_idx:
                full_weight[local_i] = w[idx.index(name_to_idx[ch])]
        weights[node] = full_weight
    return weights


def apply_node_spatial_weights(
    epoch_data: np.ndarray, ch_names: list[str], dataset: str, weights: dict[str, np.ndarray | None]
) -> dict[str, np.ndarray | None]:
    """Apply frozen per-node weights to one epoch's (n_channels, n_times) data."""
    name_to_idx = {name: i for i, name in enumerate(ch_names)}
    node_signals = {}
    for node, node_channels in REGION_CHANNELS[dataset].items():
        w = weights.get(node)
        if w is None:
            node_signals[node] = None
            continue
        idx = [name_to_idx[ch] for ch in node_channels if ch in name_to_idx]
        w_present = np.array([w[i] for i, ch in enumerate(node_channels) if ch in name_to_idx])
        node_signals[node] = w_present @ epoch_data[idx, :]
    return node_signals
