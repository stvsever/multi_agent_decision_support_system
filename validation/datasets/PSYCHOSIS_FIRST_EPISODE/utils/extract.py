"""Batch driver: build group microstate templates, then extract 836 features/subject."""

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

from . import features as F
from .config import SUBJECT_ROOT, PROCESSED_ROOT, load_config
from .montage import COMMON_CORTICAL, canonical_microstate_order

TEMPLATE_DIR = PROCESSED_ROOT / "microstate_templates"


# --------------------------------------------------------------------------- #
# Loading harmonized checkpoints
# --------------------------------------------------------------------------- #
def load_harmonized(record: dict[str, Any], cfg: dict[str, Any]):
    """Return (raw, retained_epoch_data_volts, continuous_volts) for one subject."""
    subject_dir = SUBJECT_ROOT / record["recording_id"]
    raw = mne.io.read_raw_fif(subject_dir / "clean_raw.fif", preload=True, verbose="ERROR")
    raw.reorder_channels(COMMON_CORTICAL)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=float(cfg["preprocessing"]["epoch_duration_s"]),
        preload=True, reject_by_annotation=True, verbose="ERROR")
    epoch_qc = pd.read_csv(subject_dir / "epoch_qc.csv")
    if len(epoch_qc) != len(epochs):
        raise RuntimeError("epoch QC does not align with reconstructed epochs")
    retained = epoch_qc["is_retained"].astype(bool).to_numpy()
    data = epochs.get_data(copy=True)[retained]        # (n_ep, 49, n_t) volts
    continuous = raw.get_data()                         # (49, n_samples) volts
    return raw, data, continuous


def _eligible(record: dict[str, Any]) -> bool:
    qc = record.get("preprocessing_qc", {})
    return qc.get("status") == "processed" and bool(qc.get("feature_eligible", False))


def load_records(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from . import io as eeg_io
    records = eeg_io.discover_records()
    for record in records:
        qc_path = SUBJECT_ROOT / record["recording_id"] / "preprocessing_qc.json"
        record["preprocessing_qc"] = (
            json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {})
        record["subject_dir"] = SUBJECT_ROOT / record["recording_id"]
    return records


# --------------------------------------------------------------------------- #
# Group microstate templates (fit once, reused for every subject)
# --------------------------------------------------------------------------- #
def build_group_microstate_templates(records: list[dict[str, Any]], cfg: dict[str, Any],
                                     logger=None, overwrite: bool = False):
    from pycrostates.cluster import ModKMeans
    from pycrostates.io import ChData, read_cluster
    from pycrostates.preprocessing import extract_gfp_peaks

    model_path = TEMPLATE_DIR / "group_modkmeans.fif"
    meta_path = TEMPLATE_DIR / "group_templates.json"
    if model_path.exists() and meta_path.exists() and not overwrite:
        return read_cluster(model_path)

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    cap = int(cfg["features"]["microstate_gfp_peaks_per_subject"])
    lo, hi = cfg["features"]["microstate_filter_hz"]
    pooled: list[np.ndarray] = []
    info = None
    per_subject: dict[str, int] = {}
    eligible = [r for r in records if _eligible(r)]
    for k, record in enumerate(eligible, 1):
        try:
            raw, _, _ = load_harmonized(record, cfg)
            mraw = raw.copy().filter(lo, hi, picks="eeg", verbose="ERROR")
            gp = extract_gfp_peaks(mraw, min_peak_distance=2, verbose="ERROR")
            maps = gp.get_data()
            gfp = maps.std(0)
            if maps.shape[1] > cap:
                keep = np.argsort(gfp)[-cap:]
                maps = maps[:, keep]
            pooled.append(maps)
            info = gp.info
            per_subject[record["recording_id"]] = int(maps.shape[1])
            raw.close()
        except Exception as exc:
            if logger:
                logger.warning("template contribution failed %s: %s",
                               record["recording_id"], exc)
        if logger and k % 20 == 0:
            logger.info("microstate peaks pooled from %d/%d subjects", k, len(eligible))
        gc.collect()
    all_maps = np.concatenate(pooled, axis=1)
    # Cap total to keep the modified k-means tractable and deterministic.
    rng = np.random.default_rng(int(cfg["random_seed"]))
    total_cap = 60000
    if all_maps.shape[1] > total_cap:
        sel = rng.choice(all_maps.shape[1], size=total_cap, replace=False)
        all_maps = all_maps[:, sel]

    modk = ModKMeans(n_clusters=int(cfg["features"]["microstate_n_states"]),
                     n_init=50, max_iter=300, random_state=int(cfg["random_seed"]))
    modk.fit(ChData(all_maps.astype(np.float64), info), n_jobs=1, verbose="ERROR")

    positions = info.get_montage().get_positions()["ch_pos"]
    xy = np.array([[positions[c][0], positions[c][1]] for c in COMMON_CORTICAL])
    order = canonical_microstate_order(modk.cluster_centers_, xy)
    modk.reorder_clusters(order=order)
    modk.rename_clusters(new_names=["A", "B", "C", "D"])
    modk.save(model_path)

    meta = {
        "n_states": int(cfg["features"]["microstate_n_states"]),
        "channels": COMMON_CORTICAL,
        "class_order": ["A", "B", "C", "D"],
        "labeling": "Koenig A-D by topographic orientation (utils.montage.canonical_microstate_order)",
        "global_explained_variance": float(modk.GEV_),
        "pooled_map_count": int(all_maps.shape[1]),
        "contributing_subjects": len(per_subject),
        "maps_per_subject": per_subject,
        "filter_hz": [lo, hi],
        "random_seed": int(cfg["random_seed"]),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(TEMPLATE_DIR / "group_templates.npz",
                        maps=modk.cluster_centers_, channels=np.array(COMMON_CORTICAL))
    if logger:
        logger.info("group microstate templates: GEV=%.3f from %d maps, %d subjects",
                    modk.GEV_, all_maps.shape[1], len(per_subject))
    return modk


# --------------------------------------------------------------------------- #
# Per-subject extraction
# --------------------------------------------------------------------------- #
def extract_subject(record: dict[str, Any], modk, groups: dict[str, list[str]],
                    names: list[str], cfg: dict[str, Any], overwrite: bool = False,
                    logger=None) -> tuple[dict[str, Any], dict[str, Any]]:
    subject_dir = record["subject_dir"]
    checkpoint = subject_dir / "features_836.json"
    qc_checkpoint = subject_dir / "feature_qc.json"
    if checkpoint.exists() and qc_checkpoint.exists() and not overwrite:
        return (json.loads(checkpoint.read_text(encoding="utf-8")),
                json.loads(qc_checkpoint.read_text(encoding="utf-8")))

    row: dict[str, Any] = {"recording_id": record["recording_id"]}
    row.update({n: float("nan") for n in names})
    qc: dict[str, Any] = {
        "dataset_id": record["dataset_id"],
        "participant_id": record["participant_id"],
        "recording_id": record["recording_id"],
        "status": "not_eligible",
        "group_status": {k: "not_run" for k in groups},
        "group_errors": {},
    }
    if not _eligible(record):
        qc["reason"] = record["preprocessing_qc"].get("failure_reason") or "preprocessing_exclusion"
        checkpoint.write_text(json.dumps(row), encoding="utf-8")
        qc_checkpoint.write_text(json.dumps(qc, indent=2, sort_keys=True), encoding="utf-8")
        return row, qc

    started = time.time()
    raw = None
    matrices: dict[str, np.ndarray] = {}
    try:
        raw, epoch_data, continuous = load_harmonized(record, cfg)
        sfreq = float(raw.info["sfreq"])
        qc["retained_epoch_count"] = int(epoch_data.shape[0])

        blocks = [
            (["A_spectral", "B_alpha_peak", "C_aperiodic"],
             lambda: F.extract_spectral(epoch_data, sfreq, cfg)),
            (["D_entropy", "E_fractal"],
             lambda: F.extract_complexity(epoch_data, continuous, sfreq, cfg)),
            (["F_microstates"],
             lambda: F.extract_microstates(raw, modk, cfg)),
        ]
        for keys, fn in blocks:
            try:
                row.update(fn())
                for k in keys:
                    qc["group_status"][k] = "complete"
            except Exception as exc:
                qc["group_errors"]["|".join(keys)] = f"{type(exc).__name__}: {exc}"
                for k in keys:
                    qc["group_status"][k] = "failed"
                if logger:
                    logger.exception("%s block failed for %s", keys, record["recording_id"])
        try:
            gh, matrices = F.extract_connectivity_graph(epoch_data, sfreq, cfg,
                                                        record["recording_id"])
            row.update(gh)
            g_missing = any(not np.isfinite(float(row[n])) for n in groups["G_connectivity"])
            qc["group_status"]["G_connectivity"] = (
                "complete_with_missing_edges" if g_missing else "complete")
            h_missing = any(not np.isfinite(float(row[n])) for n in groups["H_graph"])
            qc["group_status"]["H_graph"] = (
                "complete_with_unavailable_bands" if h_missing else "complete")
        except Exception as exc:
            qc["group_errors"]["GH"] = f"{type(exc).__name__}: {exc}"
            qc["group_status"]["G_connectivity"] = "failed"
            qc["group_status"]["H_graph"] = "failed"
            if logger:
                logger.exception("GH block failed for %s", record["recording_id"])

        missing = [n for n in names if not np.isfinite(float(row[n]))]
        qc["feature_count_finite"] = len(names) - len(missing)
        qc["feature_count_missing"] = len(missing)
        qc["missing_feature_names"] = missing
        qc["status"] = "complete" if not qc["group_errors"] else "partial"
        qc["runtime_s"] = time.time() - started
        assert set(row) == {"recording_id", *names}
        checkpoint.write_text(json.dumps(row), encoding="utf-8")
        qc_checkpoint.write_text(json.dumps(qc, indent=2, sort_keys=True), encoding="utf-8")
        if matrices:
            np.savez_compressed(subject_dir / "connectivity_matrices.npz", **matrices)
        if logger:
            logger.info("%s: status=%s finite=%d/%d %.1fs", record["recording_id"],
                        qc["status"], qc["feature_count_finite"], len(names), qc["runtime_s"])
        return row, qc
    except Exception as exc:
        qc["status"] = "failed"
        qc["reason"] = f"{type(exc).__name__}: {exc}"
        qc["traceback"] = traceback.format_exc()
        qc["runtime_s"] = time.time() - started
        checkpoint.write_text(json.dumps(row), encoding="utf-8")
        qc_checkpoint.write_text(json.dumps(qc, indent=2, sort_keys=True), encoding="utf-8")
        if logger:
            logger.exception("feature extraction failed for %s", record["recording_id"])
        return row, qc
    finally:
        if raw is not None:
            raw.close()
        gc.collect()
