"""Preprocessing: clean each recording and harmonize it onto the shared montage.

Stages, per recording:
  1. load microvolt data with true electrode names (utils.io);
  2. crop to a common analysable window, band-pass 0.5-45 Hz, resample to 250 Hz;
  3. flag bad cortical channels (flat, variance outlier, low neighbour
     correlation, residual line noise);
  4. remove ocular and cardiac components with picard ICA guided by VEOG/ECG;
  5. HARMONIZE: interpolate the flagged channels, restrict to the 49 electrodes
     shared by both datasets, and re-reference to their average;
  6. cut fixed 4 s epochs and score each for amplitude and muscle artefact.

Stages 1-4 are identical in spirit to the audited v1 cleaning, so a completed v1
checkpoint (ICA already applied on the native montage) can be reused for stages
5-6 without recomputing ICA. Stage 5 is the scientific upgrade over v1, which
kept dataset-specific channel sets and never interpolated.
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from . import io as eeg_io
from .config import SUBJECT_ROOT
from .montage import COMMON_CORTICAL, NON_CORTICAL


# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #
def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad <= np.finfo(float).eps:
        return np.zeros_like(values)
    return 0.67448975 * (values - median) / mad


def _line_noise_ratios(raw: mne.io.BaseRaw) -> dict[str, float]:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    names = [raw.ch_names[i] for i in picks]
    data_uv = raw.get_data(picks=picks) * 1e6
    sfreq = float(raw.info["sfreq"])
    freqs, psd = welch(data_uv, fs=sfreq, nperseg=min(int(4 * sfreq), data_uv.shape[1]),
                       noverlap=0, axis=-1)
    line = (freqs >= 58.0) & (freqs <= 62.0)
    broad = (freqs >= 1.0) & (freqs <= 100.0)
    ratios = np.sum(psd[:, line], axis=1) / np.maximum(np.sum(psd[:, broad], axis=1),
                                                       np.finfo(float).eps)
    return {name: float(v) for name, v in zip(names, ratios)}


# --------------------------------------------------------------------------- #
# Stage 3: bad channel detection
# --------------------------------------------------------------------------- #
def detect_bad_channels(raw: mne.io.BaseRaw, config: dict[str, Any],
                        line_ratios: dict[str, float]) -> pd.DataFrame:
    params = config["preprocessing"]
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    names = [raw.ch_names[i] for i in picks]
    data_uv = raw.get_data(picks=picks) * 1e6
    std = np.std(data_uv, axis=1)
    p2p = np.ptp(data_uv, axis=1)
    var_z = robust_z(np.log10(np.maximum(std, 1e-12)))

    centered = data_uv - np.median(data_uv, axis=1, keepdims=True)
    decim = max(1, int(round(raw.info["sfreq"] / 100.0)))
    corr = np.corrcoef(centered[:, ::decim])
    pos = np.array([raw.info["chs"][i]["loc"][:3] for i in picks])
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    peer_corr = np.full(len(picks), np.nan)
    for i in range(len(picks)):
        finite = np.flatnonzero(np.isfinite(dist[i]))
        if len(finite):
            nearest = finite[np.argsort(dist[i, finite])[: min(6, len(finite))]]
            peer_corr[i] = np.nanmedian(corr[i, nearest])

    line = np.array([line_ratios[name] for name in names])
    line_z = robust_z(np.log10(np.maximum(line, 1e-15)))
    is_flat = std < float(params["flat_channel_std_uv"])
    is_var = np.abs(var_z) > float(params["bad_channel_robust_z"])
    is_lowcorr = peer_corr < float(params["bad_channel_min_median_correlation"])
    is_line = (line_z > float(params["bad_channel_robust_z"])) & (
        line > float(params["bad_channel_min_line_noise_fraction"]))
    is_bad = is_flat | is_var | is_lowcorr | is_line

    reasons = []
    for i in range(len(names)):
        tags = []
        if is_flat[i]:
            tags.append("flat")
        if is_var[i]:
            tags.append("variance_outlier")
        if is_lowcorr[i]:
            tags.append("low_correlation")
        if is_line[i]:
            tags.append("line_noise_outlier")
        reasons.append("|".join(tags))
    return pd.DataFrame({
        "channel": names, "std_uv": std, "peak_to_peak_uv": p2p,
        "variance_robust_z": var_z, "median_peer_correlation": peer_corr,
        "line_noise_ratio": line, "line_noise_robust_z": line_z,
        "is_bad": is_bad, "bad_reasons": reasons,
    })


# --------------------------------------------------------------------------- #
# Stage 4: ICA cleaning
# --------------------------------------------------------------------------- #
def fit_and_apply_ica(raw: mne.io.BaseRaw, config: dict[str, Any],
                      seed: int) -> dict[str, Any]:
    params = config["preprocessing"]
    ica_raw = raw.copy().filter(l_freq=float(params["ica_highpass_hz"]), h_freq=None,
                                picks="eeg", method="fir", phase="zero",
                                fir_design="firwin", verbose="ERROR")
    ica = mne.preprocessing.ICA(
        n_components=float(params["ica_variance_fraction"]),
        method=str(params["ica_method"]),
        fit_params={"ortho": False, "extended": True},
        random_state=seed, max_iter=500, verbose="ERROR")
    ica.fit(ica_raw, picks="eeg", decim=int(params["ica_decimation"]),
            reject_by_annotation=True, verbose="ERROR")

    scores_out: dict[str, Any] = {"eog": [], "ecg": [], "errors": []}
    candidates: dict[int, float] = {}
    if "VEOG" in raw.ch_names:
        try:
            _, scores = ica.find_bads_eog(raw, ch_name="VEOG", threshold=3.0)
            selected = [i for i, s in enumerate(scores) if abs(float(s)) >= 0.35]
            scores_out["eog"] = [{"component": int(i), "score": float(scores[i])}
                                 for i in selected]
            for i in selected:
                candidates[int(i)] = max(candidates.get(int(i), 0.0), abs(float(scores[i])))
        except Exception as exc:
            scores_out["errors"].append(f"EOG: {type(exc).__name__}: {exc}")
    if "ECG" in raw.ch_names:
        try:
            _, scores = ica.find_bads_ecg(raw, ch_name="ECG", method="correlation",
                                          threshold=0.30)
            selected = [i for i, s in enumerate(scores) if abs(float(s)) >= 0.30]
            scores_out["ecg"] = [{"component": int(i), "score": float(scores[i])}
                                 for i in selected]
            for i in selected:
                candidates[int(i)] = max(candidates.get(int(i), 0.0), abs(float(scores[i])))
        except Exception as exc:
            scores_out["errors"].append(f"ECG: {type(exc).__name__}: {exc}")

    exclude = [i for i, _ in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)]
    max_remove = int(params["ica_max_components_removed"])
    if len(exclude) > max_remove:
        exclude = exclude[:max_remove]
        scores_out["errors"].append("ICA candidate list truncated to configured maximum")
    ica.exclude = exclude
    ica.apply(raw, exclude=exclude, verbose="ERROR")
    return {"n_components": int(ica.n_components_), "excluded": exclude,
            "detection": scores_out}


# --------------------------------------------------------------------------- #
# Stage 5: harmonize onto the shared 49-channel montage
# --------------------------------------------------------------------------- #
def harmonize_to_common(raw: mne.io.BaseRaw, config: dict[str, Any]) -> mne.io.BaseRaw:
    """Interpolate bad channels, keep the shared 49, and average-reference them.

    ``raw`` must be an ICA-cleaned recording on its native cortical montage with
    ``raw.info['bads']`` marking channels to interpolate.
    """
    branch = raw.copy()
    if branch.info["bads"]:
        branch.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
    missing = [ch for ch in COMMON_CORTICAL if ch not in branch.ch_names]
    if missing:
        raise RuntimeError(f"Recording is missing shared electrodes: {missing}")
    branch.pick(COMMON_CORTICAL)
    branch.reorder_channels(COMMON_CORTICAL)
    branch.set_eeg_reference("average", projection=False, verbose="ERROR")
    return branch


# --------------------------------------------------------------------------- #
# Stage 6: fixed-length epochs + per-epoch QC
# --------------------------------------------------------------------------- #
def epoch_quality(raw: mne.io.BaseRaw, config: dict[str, Any]) -> pd.DataFrame:
    params = config["preprocessing"]
    duration = float(params["epoch_duration_s"])
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True,
                                          reject_by_annotation=True, verbose="ERROR")
    picks = mne.pick_types(epochs.info, eeg=True, exclude="bads")
    data_uv = epochs.get_data(picks=picks, copy=False) * 1e6
    p2p = np.max(np.ptp(data_uv, axis=-1), axis=1)
    rms = np.sqrt(np.mean(np.square(data_uv), axis=(1, 2)))
    p2p_z = robust_z(np.log10(np.maximum(p2p, 1e-12)))
    rms_z = robust_z(np.log10(np.maximum(rms, 1e-12)))

    sfreq = float(epochs.info["sfreq"])
    spectrum = np.abs(np.fft.rfft(data_uv, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(data_uv.shape[-1], 1.0 / sfreq)
    muscle = (freqs >= 30.0) & (freqs <= 45.0)
    broad = (freqs >= 1.0) & (freqs <= 45.0)
    muscle_ratio = np.sum(spectrum[:, :, muscle], axis=(1, 2)) / np.maximum(
        np.sum(spectrum[:, :, broad], axis=(1, 2)), np.finfo(float).eps)
    muscle_z = robust_z(np.log10(np.maximum(muscle_ratio, 1e-15)))

    bad_amp = p2p > float(params["epoch_peak_to_peak_uv"])
    bad_robust = (np.abs(p2p_z) > float(params["epoch_robust_z"])) | (
        np.abs(rms_z) > float(params["epoch_robust_z"]))
    bad_muscle = muscle_z > float(params["epoch_robust_z"])
    retained = ~(bad_amp | bad_robust | bad_muscle)

    reasons = []
    for i in range(len(epochs)):
        tags = []
        if bad_amp[i]:
            tags.append("absolute_amplitude")
        if bad_robust[i]:
            tags.append("robust_amplitude")
        if bad_muscle[i]:
            tags.append("muscle_ratio")
        reasons.append("|".join(tags))
    return pd.DataFrame({
        "epoch_index": np.arange(len(epochs)),
        "onset_s": epochs.events[:, 0] / sfreq,
        "duration_s": duration,
        "peak_to_peak_uv": p2p, "rms_uv": rms,
        "peak_to_peak_robust_z": p2p_z, "rms_robust_z": rms_z,
        "muscle_power_fraction": muscle_ratio, "muscle_robust_z": muscle_z,
        "is_retained": retained, "rejection_reason": reasons,
    })


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _clean_native(record: dict[str, Any], config: dict[str, Any],
                  reuse_checkpoint: Path | None) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """Produce an ICA-cleaned recording on its native cortical montage.

    Either reuse a completed native ICA checkpoint (fast) or run stages 1-4.
    """
    qc: dict[str, Any] = {}
    if reuse_checkpoint is not None and reuse_checkpoint.exists():
        raw = mne.io.read_raw_fif(reuse_checkpoint, preload=True, verbose="ERROR")
        qc["source"] = "reused_v1_ica_checkpoint"
        # v1 stored an average reference over the native set; bads remain marked.
        qc["native_channel_count"] = len(eeg_io.cortical_eeg_names(raw))
        qc["bad_channels"] = list(raw.info["bads"])
        return raw, qc

    params = config["preprocessing"]
    raw, sidecar = eeg_io.load_raw(record)
    qc["source"] = "from_raw"
    qc["native_sampling_frequency_hz"] = float(raw.info["sfreq"])
    qc["source_duration_s"] = float(raw.times[-1])
    source_end = min(float(params["common_source_duration_s"]), float(raw.times[-1]))
    raw.crop(tmin=float(params["trim_start_s"]),
             tmax=source_end - float(params["trim_end_s"]), include_tmax=False)
    line_ratios = _line_noise_ratios(raw)
    raw.filter(l_freq=float(params["main_highpass_hz"]),
               h_freq=float(params["main_lowpass_hz"]), picks="data",
               method="fir", phase="zero", fir_design="firwin", verbose="ERROR")
    raw.resample(float(params["analysis_sampling_hz"]), npad="auto", verbose="ERROR")

    channel_table = detect_bad_channels(raw, config, line_ratios)
    cortical = set(eeg_io.cortical_eeg_names(raw))
    bad = channel_table.loc[channel_table["is_bad"] & channel_table["channel"].isin(cortical),
                            "channel"].tolist()
    non_cortical_eeg = [n for n in raw.ch_names
                        if raw.get_channel_types(picks=[n])[0] == "eeg" and n in NON_CORTICAL]
    # Keep only cortical EEG, then average-reference so ICA sees clean data.
    raw.pick([n for n in raw.ch_names if n in cortical or n in ("VEOG", "ECG")])
    raw.info["bads"] = [b for b in bad if b in raw.ch_names]
    ica_info = fit_and_apply_ica(raw, config, seed=int(config["random_seed"]))
    raw.pick(sorted(cortical, key=raw.ch_names.index))
    raw.info["bads"] = [b for b in bad if b in raw.ch_names]
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    qc["native_channel_count"] = len(cortical)
    qc["bad_channels"] = raw.info["bads"]
    qc["dropped_non_cortical"] = non_cortical_eeg
    qc["ica"] = ica_info
    qc["channel_table"] = channel_table
    return raw, qc


def run_subject(record: dict[str, Any], config: dict[str, Any],
                overwrite: bool = False, reuse_root: Path | None = None,
                logger=None) -> dict[str, Any]:
    """Full per-subject preprocessing to a harmonized checkpoint and QC record."""
    started = time.time()
    out_dir = SUBJECT_ROOT / record["recording_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_path = out_dir / "preprocessing_qc.json"
    clean_path = out_dir / "clean_raw.fif"
    if qc_path.exists() and clean_path.exists() and not overwrite:
        return json.loads(qc_path.read_text(encoding="utf-8"))

    params = config["preprocessing"]
    qc: dict[str, Any] = {
        "dataset_id": record["dataset_id"],
        "participant_id": record["participant_id"],
        "recording_id": record["recording_id"],
        "source_type_audit_only": record.get("source_type", "n/a"),
        "status": "failed",
        "failure_reason": "",
    }
    raw = None
    try:
        reuse_checkpoint = None
        if reuse_root is not None:
            candidate = reuse_root / record["recording_id"] / "clean_raw.fif"
            reuse_checkpoint = candidate if candidate.exists() else None
        native, native_qc = _clean_native(record, config, reuse_checkpoint)
        channel_table = native_qc.pop("channel_table", None)
        if channel_table is not None:
            channel_table.to_csv(out_dir / "channel_qc.csv", index=False)
        elif reuse_checkpoint is not None:
            # Carry the audited channel-quality provenance from the reused checkpoint.
            source_qc = reuse_checkpoint.with_name("channel_qc.csv")
            if source_qc.exists():
                import shutil
                shutil.copyfile(source_qc, out_dir / "channel_qc.csv")
        qc.update(native_qc)

        harmonized = harmonize_to_common(native, config)
        qc["harmonized_channel_count"] = len(harmonized.ch_names)
        qc["interpolated_channel_count"] = len(
            [b for b in native_qc.get("bad_channels", []) if b in COMMON_CORTICAL])
        qc["analysis_sampling_frequency_hz"] = float(harmonized.info["sfreq"])
        qc["analysis_duration_s"] = float(harmonized.times[-1])

        epoch_table = epoch_quality(harmonized, config)
        epoch_table.to_csv(out_dir / "epoch_qc.csv", index=False)
        retained = int(epoch_table["is_retained"].sum())
        total = int(len(epoch_table))
        usable = retained * float(params["epoch_duration_s"])
        n_bad_cortical = len([b for b in native_qc.get("bad_channels", [])
                              if b in COMMON_CORTICAL])
        qc["epoch_count_total"] = total
        qc["epoch_count_retained"] = retained
        qc["bad_epoch_fraction"] = 1.0 - retained / max(total, 1)
        qc["usable_duration_s"] = usable
        qc["bad_channel_fraction"] = n_bad_cortical / len(COMMON_CORTICAL)

        reasons = []
        if usable < float(params["minimum_clean_duration_s"]):
            reasons.append("insufficient_clean_duration")
        if retained < int(params["minimum_retained_epochs"]):
            reasons.append("insufficient_retained_epochs")
        if qc["bad_channel_fraction"] > float(params["maximum_bad_channel_fraction"]):
            reasons.append("excessive_bad_channels")
        if qc["bad_epoch_fraction"] > float(params["maximum_bad_epoch_fraction"]):
            reasons.append("excessive_bad_epochs")
        qc["feature_eligible"] = not reasons
        qc["exclusion_reasons"] = reasons

        harmonized.save(clean_path, overwrite=True, verbose="ERROR")
        qc["status"] = "processed"
        qc["runtime_s"] = time.time() - started
        qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True, default=str),
                           encoding="utf-8")
        if logger:
            logger.info("%s: %d/%d epochs, %d interp, eligible=%s",
                        record["recording_id"], retained, total,
                        qc["interpolated_channel_count"], qc["feature_eligible"])
        return qc
    except Exception as exc:
        qc["failure_reason"] = f"{type(exc).__name__}: {exc}"
        qc["traceback"] = traceback.format_exc()
        qc["runtime_s"] = time.time() - started
        qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True, default=str),
                           encoding="utf-8")
        if logger:
            logger.exception("Failed %s", record["recording_id"])
        return qc
    finally:
        if raw is not None:
            raw.close()
        gc.collect()
