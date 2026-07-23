"""Group E: fractal and scale-free dynamics (22) - spec section 7.5.

Higuchi FD: per-epoch on standardized epochs, channel-then-region median
(same pattern as Group D). DFA: computed once on the full continuous clean
signal per channel (never per-epoch, never across concatenation boundaries -
this pipeline has exactly one continuous clean segment per subject, so that
constraint is automatically satisfied), then channel-then-region median.
"""

import antropy as ant
import mne
import numpy as np

from pipeline_v1 import regions
from pipeline_v1.features.entropy import _robust_zscore

HIGUCHI_KMAX = 10


def fractal_features(epochs: mne.Epochs, raw_continuous: mne.io.Raw, dataset: str) -> dict:
    epoch_data = epochs.get_data(picks="eeg") * 1e6
    epoch_ch_names = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True)]
    n_epochs = epoch_data.shape[0]

    higuchi_per_channel = {}
    for ci, ch in enumerate(epoch_ch_names):
        vals = [ant.higuchi_fd(_robust_zscore(epoch_data[ei, ci, :]), kmax=HIGUCHI_KMAX) for ei in range(n_epochs)]
        higuchi_per_channel[ch] = float(np.nanmedian(vals))

    cont_data = raw_continuous.get_data(picks="eeg") * 1e6
    cont_ch_names = [raw_continuous.ch_names[i] for i in mne.pick_types(raw_continuous.info, eeg=True)]
    dfa_per_channel = {}
    for ci, ch in enumerate(cont_ch_names):
        try:
            dfa_per_channel[ch] = float(ant.detrended_fluctuation(cont_data[ci, :]))
        except Exception:
            dfa_per_channel[ch] = np.nan

    features = {}
    for scope in regions.SCOPES:
        features[f"E_fractal_features__higuchi_fractal_dimension__{scope}"] = regions.median_region_value(
            higuchi_per_channel, dataset, scope
        )
        features[f"E_fractal_features__detrended_fluctuation_analysis_exponent__{scope}"] = (
            regions.median_region_value(dfa_per_channel, dataset, scope)
        )
    return features
