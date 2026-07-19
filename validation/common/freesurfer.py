"""
FreeSurfer morphometry feature extraction (structural brain modality).

Parses the small per-subject FreeSurfer stats tables (aseg.stats, lh/rh.aparc.stats)
into a curated, human-labeled morphometric feature set that slots into the ontology
as a BRAIN_MORPHOMETRY domain. We deliberately reduce the raw ~250 region metrics to
a clean, interpretable set:

* global measures (intracranial volume, gray/white volumes, mean thickness),
* subcortical volumes for key bilateral structures,
* cortical thickness aggregated to lobes (surface-area weighted).

Keeping labels meaningful matters because the engine reasons over feature text.
"""

from __future__ import annotations

import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

S3_BASE = "https://s3.amazonaws.com/openneuro.org"

# Desikan-Killiany cortical region -> lobe (standard FreeSurfer lobe grouping).
DK_LOBES: Dict[str, str] = {
    # frontal
    "superiorfrontal": "frontal", "rostralmiddlefrontal": "frontal",
    "caudalmiddlefrontal": "frontal", "parsopercularis": "frontal",
    "parstriangularis": "frontal", "parsorbitalis": "frontal",
    "lateralorbitofrontal": "frontal", "medialorbitofrontal": "frontal",
    "precentral": "frontal", "paracentral": "frontal", "frontalpole": "frontal",
    # parietal
    "superiorparietal": "parietal", "inferiorparietal": "parietal",
    "supramarginal": "parietal", "postcentral": "parietal", "precuneus": "parietal",
    # temporal
    "superiortemporal": "temporal", "middletemporal": "temporal",
    "inferiortemporal": "temporal", "bankssts": "temporal", "fusiform": "temporal",
    "transversetemporal": "temporal", "entorhinal": "temporal",
    "temporalpole": "temporal", "parahippocampal": "temporal",
    # occipital
    "lateraloccipital": "occipital", "lingual": "occipital",
    "cuneus": "occipital", "pericalcarine": "occipital",
    # cingulate
    "rostralanteriorcingulate": "cingulate", "caudalanteriorcingulate": "cingulate",
    "posteriorcingulate": "cingulate", "isthmuscingulate": "cingulate",
    # insula
    "insula": "insula",
}
LOBES = ["frontal", "parietal", "temporal", "occipital", "cingulate", "insula"]

# Subcortical structures kept (matched by substring, bilateral).
SUBCORTICAL = [
    ("hippocampus", "Hippocampus"),
    ("amygdala", "Amygdala"),
    ("thalamus", "Thalamus"),
    ("caudate", "Caudate"),
    ("putamen", "Putamen"),
    ("pallidum", "Pallidum"),
    ("accumbens", "Accumbens-area"),
    ("ventraldc", "VentralDC"),
]

# Global measures pulled from the aseg.stats "# Measure" header lines.
GLOBAL_MEASURES = {
    "EstimatedTotalIntraCranialVol": ("fs_etiv", "Estimated total intracranial volume", "mm^3"),
    "BrainSegVol": ("fs_brainseg_vol", "Total brain segmentation volume", "mm^3"),
    "TotalGrayVol": ("fs_total_gray_vol", "Total gray matter volume", "mm^3"),
    "CortexVol": ("fs_cortex_vol", "Total cortical gray matter volume", "mm^3"),
    "CerebralWhiteMatterVol": ("fs_cerebral_wm_vol", "Total cerebral white matter volume", "mm^3"),
    "SubCortGrayVol": ("fs_subcort_gray_vol", "Total subcortical gray matter volume", "mm^3"),
}


def download_stats(subject: str, accession: str, dest_dir: Path) -> Dict[str, Optional[Path]]:
    """Download the three FreeSurfer stats files for a subject. Returns paths (or None)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Optional[Path]] = {}
    for key, fname in [("aseg", "aseg.stats"), ("lh", "lh.aparc.stats"), ("rh", "rh.aparc.stats")]:
        dst = dest_dir / f"{subject}_{fname}"
        if not (dst.exists() and dst.stat().st_size > 0):
            url = f"{S3_BASE}/{accession}/derivatives/freesurfer/{subject}/stats/{fname}"
            try:
                urllib.request.urlretrieve(url, dst)
            except Exception:
                out[key] = None
                continue
        out[key] = dst
    return out


def download_many(subjects: List[str], accession: str, dest_dir: Path, workers: int = 8) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_stats, s, accession, dest_dir): s for s in subjects}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    return results


def _parse_measures(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("# Measure"):
            continue
        # "# Measure BrainSeg, BrainSegVol, ..., 1322462.0, mm^3"
        parts = [p.strip() for p in line[len("# Measure"):].split(",")]
        if len(parts) >= 4:
            key = parts[1]
            try:
                out[key] = float(parts[-2])
            except ValueError:
                continue
    return out


def _parse_aseg_volumes(text: str) -> Dict[str, float]:
    """Return {StructName: Volume_mm3} for segmentation rows."""
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split()
        if len(cols) >= 5:
            try:
                vol = float(cols[3])
            except ValueError:
                continue
            out[cols[4]] = vol
    return out


def _parse_aparc(text: str) -> List[Tuple[str, float, float]]:
    """Return [(region, surf_area, thickness_avg)] from an aparc.stats table."""
    rows: List[Tuple[str, float, float]] = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split()
        # StructName NumVert SurfArea GrayVol ThickAvg ThickStd ...
        if len(cols) >= 5:
            try:
                rows.append((cols[0], float(cols[2]), float(cols[4])))
            except ValueError:
                continue
    return rows


def extract_subject_features(paths: Dict[str, Optional[Path]]) -> Dict[str, float]:
    """Extract the curated morphometry feature dict for one subject."""
    feats: Dict[str, float] = {}
    if not paths.get("aseg"):
        return feats
    aseg_text = Path(paths["aseg"]).read_text()
    measures = _parse_measures(aseg_text)
    volumes = _parse_aseg_volumes(aseg_text)

    for meas_key, (fid, _label, _unit) in GLOBAL_MEASURES.items():
        if meas_key in measures:
            feats[fid] = measures[meas_key]

    for slug, needle in SUBCORTICAL:
        for hemi_pref, hemi in [("Left", "lh"), ("Right", "rh")]:
            match = next(
                (v for k, v in volumes.items() if k.startswith(hemi_pref) and needle.split("-")[0].lower() in k.lower()),
                None,
            )
            if match is not None:
                feats[f"fs_vol_{hemi}_{slug}"] = match

    # Mean cortical thickness (aparc measures) and lobe-level thickness.
    for hemi, key in [("lh", "lh"), ("rh", "rh")]:
        if not paths.get(key):
            continue
        text = Path(paths[key]).read_text()
        m = _parse_measures(text)
        if "MeanThickness" in m:
            feats[f"fs_mean_thickness_{hemi}"] = m["MeanThickness"]
        lobe_area: Dict[str, float] = {l: 0.0 for l in LOBES}
        lobe_wsum: Dict[str, float] = {l: 0.0 for l in LOBES}
        for region, area, thick in _parse_aparc(text):
            lobe = DK_LOBES.get(region.lower())
            if lobe:
                lobe_area[lobe] += area
                lobe_wsum[lobe] += area * thick
        for lobe in LOBES:
            if lobe_area[lobe] > 0:
                feats[f"fs_thk_{hemi}_{lobe}"] = round(lobe_wsum[lobe] / lobe_area[lobe], 4)
    return feats


def feature_specs() -> Dict[str, Dict[str, Any]]:
    """Feature specs for every morphometry feature.

    Leaf features carry a self-explanatory label and no redundant description
    (the ontology stores interpretable definitions only at parent nodes).
    """
    specs: Dict[str, Dict[str, Any]] = {}
    for _mkey, (fid, label, unit) in GLOBAL_MEASURES.items():
        specs[fid] = {"label": label, "stat_type": "numeric", "units": unit,
                      "group": "brain_morphometry", "subdomain_hint": "global_brain_measures"}
    for slug, _needle in SUBCORTICAL:
        for hemi in ("lh", "rh"):
            side = "Left" if hemi == "lh" else "Right"
            specs[f"fs_vol_{hemi}_{slug}"] = {
                "label": f"{side} {slug} volume", "stat_type": "numeric", "units": "mm^3",
                "group": "brain_morphometry", "subdomain_hint": "subcortical_volumes"}
    for hemi in ("lh", "rh"):
        side = "Left" if hemi == "lh" else "Right"
        specs[f"fs_mean_thickness_{hemi}"] = {
            "label": f"{side} hemisphere mean cortical thickness", "stat_type": "numeric",
            "units": "mm", "group": "brain_morphometry", "subdomain_hint": "cortical_thickness"}
        for lobe in LOBES:
            specs[f"fs_thk_{hemi}_{lobe}"] = {
                "label": f"{side} {lobe} lobe cortical thickness", "stat_type": "numeric",
                "units": "mm", "group": "brain_morphometry", "subdomain_hint": "cortical_thickness"}
    return specs
