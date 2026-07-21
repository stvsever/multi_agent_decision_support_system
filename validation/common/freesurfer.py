"""
FreeSurfer morphometry feature extraction (brain morphometry modality).

Parses the per-subject FreeSurfer stats tables (aseg.stats, lh/rh.aparc.stats)
into a high-resolution, human-labeled morphometric feature set that slots into the
ontology under ``BRAIN -> morphometry``. Unlike a lobe-level summary, this exposes
the full Desikan-Killiany atlas resolution that the pre-processing already contains:

* global volumetric summaries (intracranial volume, gray/white volumes, ...),
* subcortical volumes for key bilateral structures,
* per-region cortical THICKNESS, SURFACE AREA and GRAY-MATTER VOLUME for all 34
  Desikan-Killiany regions per hemisphere, each nested under its lobe.

Every feature carries an explicit ontology ``path`` so the hierarchy is clean,
deep and reproducible without depending on an LLM to group brain regions. Keeping
labels meaningful matters because the engine reasons over feature text.
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
LOBE_LABEL = {l: f"{l.capitalize()} lobe" for l in LOBES}

# Human-readable Desikan-Killiany region labels.
REGION_LABELS: Dict[str, str] = {
    "superiorfrontal": "superior frontal", "rostralmiddlefrontal": "rostral middle frontal",
    "caudalmiddlefrontal": "caudal middle frontal", "parsopercularis": "pars opercularis",
    "parstriangularis": "pars triangularis", "parsorbitalis": "pars orbitalis",
    "lateralorbitofrontal": "lateral orbitofrontal", "medialorbitofrontal": "medial orbitofrontal",
    "precentral": "precentral", "paracentral": "paracentral", "frontalpole": "frontal pole",
    "superiorparietal": "superior parietal", "inferiorparietal": "inferior parietal",
    "supramarginal": "supramarginal", "postcentral": "postcentral", "precuneus": "precuneus",
    "superiortemporal": "superior temporal", "middletemporal": "middle temporal",
    "inferiortemporal": "inferior temporal", "bankssts": "banks of superior temporal sulcus",
    "fusiform": "fusiform", "transversetemporal": "transverse temporal", "entorhinal": "entorhinal",
    "temporalpole": "temporal pole", "parahippocampal": "parahippocampal",
    "lateraloccipital": "lateral occipital", "lingual": "lingual", "cuneus": "cuneus",
    "pericalcarine": "pericalcarine", "rostralanteriorcingulate": "rostral anterior cingulate",
    "caudalanteriorcingulate": "caudal anterior cingulate", "posteriorcingulate": "posterior cingulate",
    "isthmuscingulate": "isthmus cingulate", "insula": "insula",
}

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

# Ontology path prefixes (deep, deterministic; no LLM needed for brain structure).
_BRAIN = {"id": "BRAIN", "label": "Brain",
          "definition": "Features derived from brain imaging: structure (morphometry) and function (connectomics)."}
_MORPH = {"id": "morphometry", "label": "Morphometry",
          "definition": "Structural morphometry of the brain from T1-weighted MRI (FreeSurfer): regional volumes, cortical thickness and surface area."}
_CORT_THK = {"id": "cortical_thickness", "label": "Cortical Thickness",
             "definition": "Mean cortical thickness of Desikan-Killiany regions."}
_CORT_AREA = {"id": "cortical_area", "label": "Cortical Surface Area",
              "definition": "White-surface area of Desikan-Killiany cortical regions."}
_CORT_VOL = {"id": "cortical_volume", "label": "Cortical Gray-Matter Volume",
             "definition": "Gray-matter volume of Desikan-Killiany cortical regions."}
_SUBCORT = {"id": "subcortical_volumes", "label": "Subcortical Volumes",
            "definition": "Volumes of key bilateral subcortical structures."}
_GLOBAL = {"id": "global_volumetrics", "label": "Global Volumetric Summaries",
           "definition": "Whole-brain volumetric summaries."}


def _lobe_seg(lobe: str) -> Dict[str, str]:
    return {"id": lobe, "label": LOBE_LABEL[lobe], "definition": f"{LOBE_LABEL[lobe]} regions."}


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
            try:
                value = float(parts[-2])
            except ValueError:
                continue
            # FreeSurfer header rows carry both a measure name and a short key.
            # eTIV is written as "EstimatedTotalIntraCranialVol, eTIV", while
            # other consumers may address either form. Preserve both aliases.
            for key in parts[:2]:
                if key:
                    out[key] = value
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


def _parse_aparc(text: str) -> List[Tuple[str, float, float, float]]:
    """Return [(region, surf_area, gray_vol, thickness_avg)] from an aparc.stats table."""
    rows: List[Tuple[str, float, float, float]] = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split()
        # StructName NumVert SurfArea GrayVol ThickAvg ThickStd ...
        if len(cols) >= 5:
            try:
                rows.append((cols[0], float(cols[2]), float(cols[3]), float(cols[4])))
            except ValueError:
                continue
    return rows


def extract_subject_features(paths: Dict[str, Optional[Path]]) -> Dict[str, float]:
    """Extract the high-resolution morphometry feature dict for one subject."""
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

    # Per-region cortical thickness / surface area / gray-matter volume + mean thickness.
    for hemi, key in [("lh", "lh"), ("rh", "rh")]:
        if not paths.get(key):
            continue
        text = Path(paths[key]).read_text()
        m = _parse_measures(text)
        if "MeanThickness" in m:
            feats[f"fs_mean_thickness_{hemi}"] = m["MeanThickness"]
        for region, area, gray_vol, thick in _parse_aparc(text):
            if region.lower() not in DK_LOBES:
                continue
            r = region.lower()
            feats[f"fs_thk_{hemi}_{r}"] = round(thick, 4)
            feats[f"fs_area_{hemi}_{r}"] = round(area, 1)
            feats[f"fs_gmv_{hemi}_{r}"] = round(gray_vol, 1)
    return feats


def feature_specs() -> Dict[str, Dict[str, Any]]:
    """Feature specs (with deep ontology ``path``) for every morphometry feature.

    Leaf features carry a self-explanatory label and no redundant description
    (the ontology stores interpretable definitions only at parent nodes).
    """
    specs: Dict[str, Dict[str, Any]] = {}

    def add(fid, label, units, path):
        specs[fid] = {"label": label, "stat_type": "numeric", "units": units,
                      "group": "brain_morphometry", "path": path}

    for _mkey, (fid, label, unit) in GLOBAL_MEASURES.items():
        add(fid, label, unit, [_BRAIN, _MORPH, _GLOBAL])

    for slug, _needle in SUBCORTICAL:
        for hemi in ("lh", "rh"):
            side = "Left" if hemi == "lh" else "Right"
            add(f"fs_vol_{hemi}_{slug}", f"{side} {slug} volume", "mm^3",
                [_BRAIN, _MORPH, _SUBCORT])

    for hemi in ("lh", "rh"):
        side = "Left" if hemi == "lh" else "Right"
        add(f"fs_mean_thickness_{hemi}", f"{side} hemisphere mean cortical thickness", "mm",
            [_BRAIN, _MORPH, _CORT_THK, {"id": "hemispheric_mean", "label": "Hemispheric mean",
                                         "definition": "Whole-hemisphere mean cortical thickness."}])
        for region, lobe in DK_LOBES.items():
            rlab = REGION_LABELS.get(region, region)
            add(f"fs_thk_{hemi}_{region}", f"{side} {rlab} thickness", "mm",
                [_BRAIN, _MORPH, _CORT_THK, _lobe_seg(lobe)])
            add(f"fs_area_{hemi}_{region}", f"{side} {rlab} surface area", "mm^2",
                [_BRAIN, _MORPH, _CORT_AREA, _lobe_seg(lobe)])
            add(f"fs_gmv_{hemi}_{region}", f"{side} {rlab} gray-matter volume", "mm^3",
                [_BRAIN, _MORPH, _CORT_VOL, _lobe_seg(lobe)])
    return specs
