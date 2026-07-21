"""
Functional connectome feature extraction (movie-watching fMRI).

Raw voxel connectivity has far too many edges to hand to the engine, so this is
a real dimensionality-reduction pipeline: fMRIPrep-preprocessed BOLD is parcellated
with the Schaefer-2018 atlas (100 parcels, labelled by the 7 Yeo networks), denoised
with fMRIPrep confounds, and reduced to **network-level functional connectivity**:
the 7 within-network and 21 between-network mean correlations (28 features total).

These nest under ``BRAIN -> connectomics`` in the ontology (alongside the FreeSurfer
morphometry branch), split into within-network and between-network connectivity. The
full parcel-level correlation matrix is returned too, purely for notebook visualisation.

Resolution knob: ``feature_specs`` / ``network_fc`` are parametrised by the network
set, so re-extracting from BOLD with a finer atlas (e.g. Schaefer-200 labelled by the
17 Yeo sub-networks -> 17 within + 136 between = 153 features) is a drop-in change. The
cached derivatives here are 7-network, so 7 networks is the default.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

S3_BASE = "https://s3.amazonaws.com/openneuro.org"

# 7 Yeo networks in Schaefer label order, with readable names.
YEO7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
YEO7_LABELS = {
    "Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "Dorsal attention",
    "SalVentAttn": "Salience / ventral attention", "Limbic": "Limbic",
    "Cont": "Frontoparietal control", "Default": "Default mode",
}
CONFOUND_COLUMNS = [
    "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
    "csf", "white_matter", "global_signal", "framewise_displacement",
]
_ATLAS_CACHE: Dict[Tuple[int, int], Any] = {}


def load_atlas(n_rois: int = 100, yeo_networks: int = 7, resolution_mm: int = 2):
    key = (n_rois, yeo_networks)
    if key not in _ATLAS_CACHE:
        from nilearn.datasets import fetch_atlas_schaefer_2018
        _ATLAS_CACHE[key] = fetch_atlas_schaefer_2018(
            n_rois=n_rois, yeo_networks=yeo_networks, resolution_mm=resolution_mm
        )
    return _ATLAS_CACHE[key]


def atlas_networks(atlas) -> List[str]:
    """Return the Yeo network of each parcel (in atlas order)."""
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]
    nets = []
    for l in labels:
        parts = l.split("_")
        nets.append(parts[2] if len(parts) > 2 else "Other")
    return nets


def _func_url(subject: str, accession: str, name: str) -> str:
    return f"{S3_BASE}/{accession}/derivatives/fmriprep/{subject}/func/{name}"


def download_func(subject: str, accession: str, cache_dir: Path) -> Dict[str, Optional[Path]]:
    """Download preprocessed BOLD, confounds, and brain mask for one subject."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    space = "space-MNI152NLin2009cAsym"
    files = {
        "bold": f"{subject}_task-moviewatching_{space}_desc-preproc_bold.nii.gz",
        "confounds": f"{subject}_task-moviewatching_desc-confounds_regressors.tsv",
        "mask": f"{subject}_task-moviewatching_{space}_desc-brain_mask.nii.gz",
    }
    out: Dict[str, Optional[Path]] = {}
    for key, name in files.items():
        dst = cache_dir / name
        if not (dst.exists() and dst.stat().st_size > 0):
            try:
                urllib.request.urlretrieve(_func_url(subject, accession, name), dst)
            except Exception:
                out[key] = None
                continue
        out[key] = dst
    return out


def parcel_timeseries(paths: Dict[str, Optional[Path]], atlas, t_r: float = 2.2) -> np.ndarray:
    """Confound-denoised parcel timeseries (TRs x parcels)."""
    from nilearn.maskers import NiftiLabelsMasker
    cdf = pd.read_csv(paths["confounds"], sep="\t")
    keep = [c for c in CONFOUND_COLUMNS if c in cdf.columns]
    conf = cdf[keep].fillna(0.0).values
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps, standardize="zscore_sample", detrend=True,
        low_pass=0.1, high_pass=0.01, t_r=t_r, mask_img=paths.get("mask"),
        resampling_target="data", verbose=0,
    )
    return masker.fit_transform(str(paths["bold"]), confounds=conf)


def network_fc(ts: np.ndarray, networks: List[str]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Reduce parcel timeseries to network-level FC.

    Returns (feature_dict, network_fc_7x7, parcel_fc_full). Network FC is the mean
    Fisher-z correlation of parcel pairs within/between each network pair.
    """
    nets = np.array(networks)
    parcel_fc = np.corrcoef(ts.T)
    parcel_z = np.arctanh(np.clip(parcel_fc, -0.999, 0.999))

    feats: Dict[str, float] = {}
    net_mat = np.full((len(YEO7), len(YEO7)), np.nan)
    for i, a in enumerate(YEO7):
        ia = np.where(nets == a)[0]
        for j, b in enumerate(YEO7):
            if j < i:
                continue
            ib = np.where(nets == b)[0]
            if len(ia) == 0 or len(ib) == 0:
                continue
            block = parcel_z[np.ix_(ia, ib)]
            if a == b:
                # within-network: upper triangle only (exclude self-correlations)
                iu = np.triu_indices(len(ia), k=1)
                vals = block[iu]
            else:
                vals = block.flatten()
            if vals.size == 0:
                continue
            val = float(np.tanh(np.nanmean(vals)))  # back to correlation units
            net_mat[i, j] = net_mat[j, i] = val
            fid = f"fc_{a.lower()}_within" if a == b else f"fc_{a.lower()}_{b.lower()}"
            feats[fid] = round(val, 4)
    return feats, net_mat, parcel_fc


def extract_subject_fc(paths, atlas, networks, t_r: float = 2.2):
    ts = parcel_timeseries(paths, atlas, t_r=t_r)
    return network_fc(ts, networks)


# Ontology path prefixes (shared BRAIN domain with morphometry; deterministic).
from .freesurfer import _BRAIN  # noqa: E402  (shared brain domain segment)
_CONN = {"id": "connectomics", "label": "Connectomics",
         "definition": "Functional connectivity of the brain from movie-watching fMRI (Yeo/Schaefer intrinsic networks)."}
_WITHIN = {"id": "within_network", "label": "Within-network FC",
           "definition": "Mean functional connectivity among parcels of the same intrinsic network."}
_BETWEEN = {"id": "between_network", "label": "Between-network FC",
            "definition": "Mean functional connectivity between pairs of intrinsic networks."}


def feature_specs(networks: List[str] = YEO7, labels: Dict[str, str] = YEO7_LABELS) -> Dict[str, Dict[str, Any]]:
    """Feature specs (with deep ontology ``path``) for network-level FC.

    Parametrised by the network set so a finer atlas re-extraction reuses this
    verbatim. Self-explanatory labels, no redundant text.
    """
    specs: Dict[str, Dict[str, Any]] = {}
    for i, a in enumerate(networks):
        for j, b in enumerate(networks):
            if j < i:
                continue
            if a == b:
                fid = f"fc_{a.lower()}_within"
                label = f"{labels[a]} within-network FC"
                path = [_BRAIN, _CONN, _WITHIN]
            else:
                fid = f"fc_{a.lower()}_{b.lower()}"
                label = f"{labels[a]} - {labels[b]} FC"
                path = [_BRAIN, _CONN, _BETWEEN]
            specs[fid] = {"label": label, "stat_type": "numeric", "units": "Pearson r",
                          "group": "brain_connectome", "path": path}
    return specs
