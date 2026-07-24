"""Stage 7: final reference, interpolation, epoching, and rejection.

Produces two parallel epoch sets:
  - `epochs_interp`: bad channels interpolated, average-referenced. Used for
    univariate (A/B/C/D/E) and microstate (F) features.
  - `epochs_no_interp`: bad channels dropped (never interpolated), average
    referenced over remaining good channels. Used to build connectivity (G)
    node signals, per spec: "connectivity node signals use original good
    channels only."
"""

import mne
from autoreject import AutoReject

from pipeline_v1 import regions

EPOCH_DURATION_S = 4.0
RANDOM_STATE = 97


def _restrict_to_cortical(raw: mne.io.Raw, dataset: str) -> mne.io.Raw:
    cortical = regions.cortical_channels(dataset)
    keep = [ch for ch in raw.ch_names if ch in cortical]
    return raw.copy().pick(picks=keep)


def build_epochs(raw_clean_main: mne.io.Raw, dataset: str, bad_channels: list[str]) -> dict:
    cortical = regions.cortical_channels(dataset)
    bad_cortical = [ch for ch in bad_channels if ch in cortical]

    # --- interpolated copy (univariate / microstate features) ---
    raw_interp = _restrict_to_cortical(raw_clean_main, dataset)
    raw_interp.info["bads"] = [ch for ch in bad_cortical if ch in raw_interp.ch_names]
    if raw_interp.info["bads"]:
        raw_interp.interpolate_bads(reset_bads=True, verbose=False)
    raw_interp.set_eeg_reference("average", projection=False, verbose=False)

    epochs_interp = mne.make_fixed_length_epochs(raw_interp, duration=EPOCH_DURATION_S, preload=True, verbose=False)
    n_epochs_total = len(epochs_interp)

    ar = AutoReject(n_interpolate=[1, 2], consensus=[0.3, 0.5], random_state=RANDOM_STATE, verbose=False)
    epochs_interp_clean, reject_log = ar.fit_transform(epochs_interp, return_log=True)
    n_epochs_retained = len(epochs_interp_clean)

    # --- non-interpolated copy (connectivity node input) ---
    raw_no_interp = _restrict_to_cortical(raw_clean_main, dataset)
    good_only = [ch for ch in raw_no_interp.ch_names if ch not in bad_cortical]
    raw_no_interp = raw_no_interp.pick(picks=good_only)
    raw_no_interp.set_eeg_reference("average", projection=False, verbose=False)

    epochs_no_interp = mne.make_fixed_length_epochs(
        raw_no_interp, duration=EPOCH_DURATION_S, preload=True, verbose=False
    )
    # Apply the same epoch-level accept/reject decisions as the interpolated copy,
    # so both epoch sets refer to the same underlying time windows.
    good_epoch_mask = ~reject_log.bad_epochs
    if len(epochs_no_interp) == len(good_epoch_mask):
        epochs_no_interp = epochs_no_interp[good_epoch_mask]

    node_channel_counts = {
        node: sum(1 for ch in chans if ch in good_only) for node, chans in regions.REGION_CHANNELS[dataset].items()
    }

    qc = {
        "n_eeg_channels_original": len(cortical),
        "n_bad_channels": len(bad_cortical),
        "bad_channel_fraction": len(bad_cortical) / len(cortical) if cortical else None,
        "n_interpolated_channels": len(bad_cortical),
        "n_epochs_total": n_epochs_total,
        "n_epochs_retained": n_epochs_retained,
        "bad_epoch_fraction": 1.0 - (n_epochs_retained / n_epochs_total) if n_epochs_total else None,
        "usable_duration_s": n_epochs_retained * EPOCH_DURATION_S,
        "connectivity_node_coverage_count": sum(1 for c in node_channel_counts.values() if c >= 2),
        "node_channel_counts": node_channel_counts,
    }

    return {
        "epochs_interp": epochs_interp_clean,
        "epochs_no_interp": epochs_no_interp,
        "raw_interp_continuous": raw_interp,  # continuous, bad-channel-interpolated, avg-referenced - for DFA (Group E)
        "bad_cortical_channels": bad_cortical,
        "qc": qc,
    }
