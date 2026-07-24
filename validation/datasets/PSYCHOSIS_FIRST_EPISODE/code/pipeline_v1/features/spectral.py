"""Group A: spectral features (168) - spec section 7.1.

4 s multitaper PSD per epoch (time-bandwidth NW=3 -> bandwidth=1.5 Hz for 4 s
epochs -> 2*NW-1=5 tapers), band power integrated per channel per epoch, then
aggregated epoch-median-then-region-median throughout (per spec section 8's
general aggregation rule, applied consistently rather than only to entropy).
"""

import mne
import numpy as np

from pipeline_v1 import regions

BANDS = {
    "delta_1_4_hz": (1.0, 4.0),
    "theta_4_8_hz": (4.0, 8.0),
    "alpha_8_13_hz": (8.0, 13.0),
    "beta_13_30_hz": (13.0, 30.0),
    "low_gamma_30_45_hz": (30.0, 45.0),
}
RATIOS = {
    "theta_over_alpha": ("theta_4_8_hz", "alpha_8_13_hz"),
    "theta_over_beta": ("theta_4_8_hz", "beta_13_30_hz"),
    "alpha_over_delta": ("alpha_8_13_hz", "delta_1_4_hz"),
}
ASYMMETRY_REGIONS = ["frontal", "central", "temporal", "parietal", "occipital"]

MULTITAPER_BANDWIDTH_HZ = 1.5  # NW=3 for 4 s epochs -> 5 tapers
EPS = 1e-15  # stabilizer for log/ratio of near-zero power


def _band_powers_per_epoch(epochs: mne.Epochs) -> tuple[dict, np.ndarray, list[str]]:
    """Returns ({band: (n_epochs, n_channels) power in uV^2}, freqs, ch_names)."""
    spectrum = epochs.compute_psd(
        method="multitaper", fmin=1.0, fmax=45.0, bandwidth=MULTITAPER_BANDWIDTH_HZ, picks="eeg", verbose=False
    )
    psd, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs), V^2/Hz
    psd_uv2 = psd * 1e12  # V^2/Hz -> uV^2/Hz

    band_power = {}
    for band, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs < fmax) if band != "low_gamma_30_45_hz" else (freqs >= fmin) & (freqs <= fmax)
        band_power[band] = np.trapezoid(psd_uv2[:, :, mask], freqs[mask], axis=2)  # (n_epochs, n_channels)
    return band_power, freqs, spectrum.ch_names


def spectral_features(epochs: mne.Epochs, dataset: str) -> dict:
    band_power, _, ch_names = _band_powers_per_epoch(epochs)
    total_power = sum(band_power.values())  # (n_epochs, n_channels), bands are contiguous over 1-45 Hz

    # --- per-channel, epoch-median-aggregated values, used to build every family ---
    log_abs_power_ch = {}  # {band: {channel: value}}
    rel_power_ch = {}
    ratio_log_ch = {band_pair: {} for band_pair in RATIOS}  # keyed by ratio name

    for band in BANDS:
        log_abs_power_ch[band] = {}
        rel_power_ch[band] = {}
        for ci, ch in enumerate(ch_names):
            abs_power_epochs = band_power[band][:, ci]
            log_abs_power_ch[band][ch] = float(np.median(np.log10(abs_power_epochs + EPS)))
            rel_power_epochs = abs_power_epochs / (total_power[:, ci] + EPS)
            rel_power_ch[band][ch] = float(np.median(rel_power_epochs))

    for ratio_name, (num_band, den_band) in RATIOS.items():
        for ci, ch in enumerate(ch_names):
            num = band_power[num_band][:, ci]
            den = band_power[den_band][:, ci]
            log_ratio_epochs = np.log((num + EPS) / (den + EPS))
            ratio_log_ch[ratio_name][ch] = float(np.median(log_ratio_epochs))

    # --- region-median aggregation into the 11 scopes ---
    features = {}
    for band in BANDS:
        for scope in regions.SCOPES:
            features[f"A_spectral_features__log10_absolute_power_uv2__{band}__{scope}"] = (
                regions.median_region_value(log_abs_power_ch[band], dataset, scope)
            )
            features[f"A_spectral_features__relative_power_fraction_of_1_45_hz__{band}__{scope}"] = (
                regions.median_region_value(rel_power_ch[band], dataset, scope)
            )

    for ratio_name in RATIOS:
        for scope in regions.SCOPES:
            features[f"A_spectral_features__natural_log_power_ratio__{ratio_name}__{scope}"] = (
                regions.median_region_value(ratio_log_ch[ratio_name], dataset, scope)
            )

    for band in BANDS:
        for region in ASYMMETRY_REGIONS:
            left_scope, right_scope = regions.HOMOLOGOUS_PAIRS[region]
            left_val = features[f"A_spectral_features__log10_absolute_power_uv2__{band}__{left_scope}"]
            right_val = features[f"A_spectral_features__log10_absolute_power_uv2__{band}__{right_scope}"]
            asym = (right_val - left_val) if (left_val is not None and right_val is not None) else None
            features[f"A_spectral_features__log10_power_asymmetry_right_minus_left__{band}__{region}"] = asym

    return features
