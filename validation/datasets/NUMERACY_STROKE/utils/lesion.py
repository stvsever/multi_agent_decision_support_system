"""
Lesion mask feature extraction (structural stroke-lesion modality).

Turns a subject's binary lesion mask into per-region "affected voxel ratio"
features across a whole-brain parcellation assembled from three atlases:

* cortex     - Schaefer-2018, Yeo-7 networks (reuses connectome.load_atlas)
* subcortex  - Tian scale IV (S4), 54 bilateral parcels
* cerebellum - Nettekoven-2024, asymmetric 128-region parcellation

All three ship in the same MNI152 stereotaxic family as this dataset's
already-normalized lesion masks (confirmed: identical affine/shape to the
dataset's own T1w and standard template), so each is resampled once
(nearest-neighbor - parcel IDs are categorical, not continuous) onto the
lesion masks' exact grid. No per-subject registration is needed.

ROI IDs are merged into one label volume with non-overlapping ranges: cortex
1..n_rois, subcortex next 54, cerebellum next up to 128. On any boundary-voxel
overlap between atlases, the more anatomically specific one wins (cerebellum
> subcortex > cortex).
"""

from __future__ import annotations

import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from validation.common import connectome as conn

S3_BASE = "https://s3.amazonaws.com/openneuro.org"
TIAN_BASE = (
    "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
    "Group-Parcellation/3T/Subcortex-Only"
)
CEREBELLAR_BASE = (
    "https://raw.githubusercontent.com/DiedrichsenLab/cerebellar_atlases/master/"
    "Nettekoven_2024"
)

N_TIAN_S4 = 54


# --------------------------------------------------------------- download

def _lesion_url(subject: str, accession: str, name: str) -> str:
    return f"{S3_BASE}/{accession}/derivatives/lesion_masks/{subject}/anat/{name}"


def download_lesion_files(subject: str, accession: str, dest_dir: Path) -> Dict[str, Optional[Path]]:
    """Download the binary lesion mask for one subject, replacing a broken symlink."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Optional[Path]] = {}
    name = f"{subject}_lesion-mask.nii.gz"
    dst = dest_dir / name
    real = dst.exists() and not dst.is_symlink() and dst.stat().st_size > 0
    if not real:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        try:
            urllib.request.urlretrieve(_lesion_url(subject, accession, name), dst)
        except Exception:
            out["lesion_mask"] = None
            return out
    out["lesion_mask"] = dst
    return out


def download_many(subjects: List[str], accession: str, dest_root: Path, workers: int = 8) -> Dict[str, Dict]:
    """Bulk, idempotent download across subjects. dest_root is .../derivatives/lesion_masks."""
    results: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(download_lesion_files, s, accession, dest_root / s / "anat"): s
            for s in subjects
        }
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    return results


def fetch_tian_s4(cache_dir: Path) -> Tuple[Path, Path]:
    """Download the Tian scale-IV (S4) subcortical atlas NIfTI + label file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    nii = cache_dir / "Tian_Subcortex_S4_3T_1mm.nii.gz"
    labels = cache_dir / "Tian_Subcortex_S4_3T_label.txt"
    if not (nii.exists() and nii.stat().st_size > 0):
        urllib.request.urlretrieve(f"{TIAN_BASE}/Tian_Subcortex_S4_3T_1mm.nii.gz", nii)
    if not (labels.exists() and labels.stat().st_size > 0):
        urllib.request.urlretrieve(f"{TIAN_BASE}/Tian_Subcortex_S4_3T_label.txt", labels)
    return nii, labels


def fetch_cerebellar_atlas(cache_dir: Path) -> Tuple[Path, Path]:
    """Download the Nettekoven-2024 asymmetric 128-region cerebellar atlas (MNI space)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    nii = cache_dir / "atl-NettekovenAsym128_space-MNI_dseg.nii"
    lut = cache_dir / "atl-NettekovenAsym128.lut"
    if not (nii.exists() and nii.stat().st_size > 0):
        urllib.request.urlretrieve(
            f"{CEREBELLAR_BASE}/atl-NettekovenAsym128_space-MNI_dseg.nii", nii
        )
    if not (lut.exists() and lut.stat().st_size > 0):
        urllib.request.urlretrieve(f"{CEREBELLAR_BASE}/atl-NettekovenAsym128.lut", lut)
    return nii, lut


# ----------------------------------------------------------- label parsing

def _parse_tian_labels(path: Path) -> Dict[int, str]:
    names = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    return {i + 1: name for i, name in enumerate(names)}


def _parse_cerebellar_lut(path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx == 0:
            continue
        out[idx] = parts[-1]
    return out


# ------------------------------------------------------------ atlas build

def build_combined_atlas(
    reference_img,
    cache_dir: Path,
    n_rois: int = 1000,
    yeo_networks: int = 7,
    resolution_mm: int = 1,
) -> Tuple[np.ndarray, Dict[int, str], Dict[int, str]]:
    """Resample & merge cortex + subcortex + cerebellum into one label volume.

    Returns (combined_labels [reference grid shape], {roi_id: name},
    {roi_id: region_type}) where region_type is "cortex"/"subcortex"/
    "cerebellum". ROI ids with zero voxels after resampling are dropped.
    """
    from nilearn.image import resample_to_img
    import nibabel as nib

    roi_names: Dict[int, str] = {}
    roi_region: Dict[int, str] = {}
    combined = np.zeros(reference_img.shape, dtype=np.int32)

    # --- cortex (Schaefer) ---
    cortex_atlas = conn.load_atlas(n_rois=n_rois, yeo_networks=yeo_networks, resolution_mm=resolution_mm)
    cortex_img = resample_to_img(
        cortex_atlas.maps, reference_img, interpolation="nearest",
        force_resample=True, copy_header=True,
    )
    cortex_data = np.rint(np.asarray(cortex_img.dataobj)).astype(np.int32)
    cortex_label_names = [l.decode() if isinstance(l, bytes) else l for l in cortex_atlas.labels]
    for i in range(1, n_rois + 1):
        roi_names[i] = cortex_label_names[i - 1] if i - 1 < len(cortex_label_names) else f"Schaefer_{i}"
        roi_region[i] = "cortex"
    mask = cortex_data > 0
    combined[mask] = cortex_data[mask]

    # --- subcortex (Tian S4) ---
    tian_nii, tian_labels_path = fetch_tian_s4(cache_dir / "tian")
    tian_img = nib.load(tian_nii)
    tian_resampled = resample_to_img(
        tian_img, reference_img, interpolation="nearest",
        force_resample=True, copy_header=True,
    )
    tian_data = np.rint(np.asarray(tian_resampled.dataobj)).astype(np.int32)
    tian_names = _parse_tian_labels(tian_labels_path)
    subcortex_offset = n_rois
    for tid, name in tian_names.items():
        roi_names[subcortex_offset + tid] = name
        roi_region[subcortex_offset + tid] = "subcortex"
    mask = tian_data > 0
    combined[mask] = subcortex_offset + tian_data[mask]

    # --- cerebellum (Nettekoven) ---
    cereb_nii, cereb_lut_path = fetch_cerebellar_atlas(cache_dir / "cerebellar")
    cereb_img = nib.load(cereb_nii)
    cereb_resampled = resample_to_img(
        cereb_img, reference_img, interpolation="nearest",
        force_resample=True, copy_header=True,
    )
    cereb_data = np.rint(np.asarray(cereb_resampled.dataobj)).astype(np.int32)
    cereb_names = _parse_cerebellar_lut(cereb_lut_path)
    cerebellum_offset = subcortex_offset + N_TIAN_S4
    for cid, name in cereb_names.items():
        roi_names[cerebellum_offset + cid] = name
        roi_region[cerebellum_offset + cid] = "cerebellum"
    mask = cereb_data > 0
    combined[mask] = cerebellum_offset + cereb_data[mask]

    # Drop declared ROI ids that ended up with zero voxels (resampling/overlap loss).
    present_ids = set(np.unique(combined).tolist()) - {0}
    roi_names = {rid: name for rid, name in roi_names.items() if rid in present_ids}
    roi_region = {rid: region for rid, region in roi_region.items() if rid in present_ids}

    return combined, roi_names, roi_region


def roi_voxel_counts(combined_atlas: np.ndarray, roi_names: Dict[int, str]) -> Dict[int, int]:
    """Total voxel count per ROI in the (fixed, subject-independent) combined atlas."""
    max_id = int(combined_atlas.max())
    counts = np.bincount(combined_atlas.ravel(), minlength=max_id + 1)
    return {rid: int(counts[rid]) for rid in roi_names if rid <= max_id}


# --------------------------------------------------------- per-subject

def extract_subject_lesion_features(
    lesion_path: Path,
    combined_atlas: np.ndarray,
    roi_total_voxels: Dict[int, int],
) -> Dict[str, float]:
    """Per-ROI affected-voxel-ratio features plus whole-brain totals for one subject."""
    import nibabel as nib

    img = nib.load(lesion_path)
    data = np.asarray(img.dataobj) > 0
    zooms = img.header.get_zooms()[:3]
    voxel_vol_mm3 = float(zooms[0]) * float(zooms[1]) * float(zooms[2])

    max_id = int(combined_atlas.max())
    lesioned_counts = np.bincount(
        combined_atlas.ravel(), weights=data.ravel().astype(np.float64), minlength=max_id + 1
    )

    feats: Dict[str, float] = {}
    for roi_id, n_roi in roi_total_voxels.items():
        feats[f"lesion_ratio_p{roi_id}"] = round(float(lesioned_counts[roi_id]) / n_roi, 6)

    total_voxels = int(data.sum())
    feats["lesion_total_voxels"] = total_voxels
    feats["lesion_total_volume_mm3"] = round(total_voxels * voxel_vol_mm3, 2)
    return feats


def feature_specs(roi_names: Dict[int, str], roi_region: Dict[int, str]) -> Dict[str, Dict[str, Any]]:
    """Feature specs for every generated lesion column, same shape as freesurfer.feature_specs()."""
    source_by_region = {
        "cortex": "Schaefer-2018 cortical parcellation (Yeo-7 networks)",
        "subcortex": "Tian scale-IV (S4) subcortical parcellation",
        "cerebellum": "Nettekoven-2024 asymmetric 128-region cerebellar parcellation",
    }
    specs: Dict[str, Dict[str, Any]] = {}
    for roi_id, name in roi_names.items():
        region = roi_region.get(roi_id, "unknown")
        specs[f"lesion_ratio_p{roi_id}"] = {
            "label": f"Lesion overlap: {name}",
            "description": f"Fraction of the {name} parcel's voxels marked as lesioned.",
            "stat_type": "numeric",
            "units": "proportion",
            "group": f"lesion_{region}",
            "subdomain_hint": source_by_region.get(region, region),
        }
    specs["lesion_total_voxels"] = {
        "label": "Total lesioned voxels", "stat_type": "numeric", "units": "voxels",
        "group": "lesion_summary", "subdomain_hint": "whole_brain",
    }
    specs["lesion_total_volume_mm3"] = {
        "label": "Total lesion volume", "stat_type": "numeric", "units": "mm^3",
        "group": "lesion_summary", "subdomain_hint": "whole_brain",
    }
    return specs
