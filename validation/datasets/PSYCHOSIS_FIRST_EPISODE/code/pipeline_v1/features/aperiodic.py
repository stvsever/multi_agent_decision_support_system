"""Groups B (33) + C (22): peak alpha frequency and aperiodic 1/f slope.

One shared specparam fit per scope (spec sections 7.2/7.3): fit on the
2-40 Hz region-median PSD curve, fixed aperiodic mode. Group C reads the
aperiodic exponent/offset directly; Group B searches the fitted periodic
peaks within 7-14 Hz. specparam operates in log10(power) space, so a peak's
`height` parameter *is* its elevation above the aperiodic line in log10-power
units; multiplying by 10 gives dB (verified against a synthetic test signal:
log-space peak component peaks exactly at the fitted height, linear-space
component recovers the true injected linear power).
"""

import mne
import numpy as np
from specparam import SpectralModel

from pipeline_v1 import regions
from pipeline_v1.features.spectral import MULTITAPER_BANDWIDTH_HZ

ALPHA_SEARCH_RANGE_HZ = (7.0, 14.0)
FIT_FREQ_RANGE_HZ = (2.0, 40.0)
MIN_PEAK_PROMINENCE_DB = 3.0  # min_peak_height=0.3 log10-units == 3 dB


def _channel_median_psd(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-channel epoch-median PSD curve in uV^2/Hz. Returns (freqs, psd[chan, freq], ch_names)."""
    spectrum = epochs.compute_psd(
        method="multitaper", fmin=1.0, fmax=45.0, bandwidth=MULTITAPER_BANDWIDTH_HZ, picks="eeg", verbose=False
    )
    psd, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs), V^2/Hz
    psd_uv2 = psd * 1e12
    median_psd = np.median(psd_uv2, axis=0)  # (n_channels, n_freqs)
    return freqs, median_psd, spectrum.ch_names


def _region_median_curve(freqs: np.ndarray, median_psd: np.ndarray, ch_names: list[str], dataset: str, scope: str) -> np.ndarray | None:
    chans = regions.scope_channels(dataset, scope)
    idx = [ch_names.index(ch) for ch in chans if ch in ch_names]
    if not idx:
        return None
    return np.median(median_psd[idx, :], axis=0)


def aperiodic_and_alpha_peak_features(epochs: mne.Epochs, dataset: str) -> tuple[dict, dict]:
    """Returns (features_dict, qc_dict)."""
    freqs, median_psd, ch_names = _channel_median_psd(epochs)

    features = {}
    r_squared_values = []
    error_values = []
    n_alpha_peaks_found = 0

    for scope in regions.SCOPES:
        curve = _region_median_curve(freqs, median_psd, ch_names, dataset, scope)
        b_prefix = f"B_peak_alpha_frequency"
        c_prefix = f"C_spectral_slope_1_f"

        if curve is None:
            features[f"{b_prefix}__strongest_alpha_peak_frequency_hz__{scope}"] = None
            features[f"{b_prefix}__alpha_center_of_gravity_hz__{scope}"] = None
            features[f"{b_prefix}__alpha_peak_prominence_db__{scope}"] = None
            features[f"{c_prefix}__aperiodic_exponent__{scope}"] = None
            features[f"{c_prefix}__aperiodic_offset_log10_power__{scope}"] = None
            continue

        sm = SpectralModel(
            aperiodic_mode="fixed",
            peak_width_limits=(0.5, 4.0),
            min_peak_height=MIN_PEAK_PROMINENCE_DB / 10.0,
            verbose=False,
        )
        sm.fit(freqs, curve, freq_range=FIT_FREQ_RANGE_HZ)

        offset, exponent = sm.results.get_params("aperiodic")
        features[f"{c_prefix}__aperiodic_exponent__{scope}"] = float(exponent)
        features[f"{c_prefix}__aperiodic_offset_log10_power__{scope}"] = float(offset)
        r_squared_values.append(float(sm.results.get_metrics("gof")))
        error_values.append(float(sm.results.get_metrics("error")))

        peak_params = sm.results.get_params("peak")
        alpha_peaks = (
            [p for p in peak_params if ALPHA_SEARCH_RANGE_HZ[0] <= p[0] <= ALPHA_SEARCH_RANGE_HZ[1]]
            if len(peak_params)
            else []
        )

        if not alpha_peaks:
            features[f"{b_prefix}__strongest_alpha_peak_frequency_hz__{scope}"] = None
            features[f"{b_prefix}__alpha_center_of_gravity_hz__{scope}"] = None
            features[f"{b_prefix}__alpha_peak_prominence_db__{scope}"] = None
            continue

        n_alpha_peaks_found += 1
        strongest = max(alpha_peaks, key=lambda p: p[1])
        features[f"{b_prefix}__strongest_alpha_peak_frequency_hz__{scope}"] = float(strongest[0])
        features[f"{b_prefix}__alpha_peak_prominence_db__{scope}"] = float(strongest[1] * 10.0)

        peak_component_linear = sm.results.model.get_component("peak", space="linear")
        fit_freqs = sm.data.freqs
        mask = (fit_freqs >= ALPHA_SEARCH_RANGE_HZ[0]) & (fit_freqs <= ALPHA_SEARCH_RANGE_HZ[1])
        weights = np.clip(peak_component_linear[mask], 0, None)
        cog = float(np.sum(fit_freqs[mask] * weights) / np.sum(weights)) if np.sum(weights) > 0 else None
        features[f"{b_prefix}__alpha_center_of_gravity_hz__{scope}"] = cog

    qc = {
        "aperiodic_fit_r2_median": float(np.nanmedian(r_squared_values)) if r_squared_values else None,
        "aperiodic_fit_error_median": float(np.nanmedian(error_values)) if error_values else None,
        "alpha_peak_detected_scope_count": n_alpha_peaks_found,
    }
    return features, qc
