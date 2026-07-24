#!/usr/bin/env python3
"""
Step 03 - build fine and coarse master ontologies plus the coarse group-level
lesion feature table.

The ontology is an ARBITRARY-DEPTH abstract hierarchy on top of the flat feature
table, emitted in the engine's native `children` schema via the current, unmodified
`validation.common.ontology.build_ontology_tree`. Every feature carries an explicit
`path` (domain first), so placement is fully deterministic - no LLM judgment and no
spend. The atlas hierarchy (cortex -> Yeo network -> parcel; subcortex -> Tian
structure -> parcel; cerebellum -> Nettekoven domain -> subregion -> parcel) already
exists by construction, so we reuse it as abstract intermediate levels instead of a
flat per-ROI list. This mirrors the deterministic `path` ingestion the brain
modalities in AOMIC_ID1000 already use.

Abstract structure produced (both granularities share the same top domains):

  Demographics and Background -> Participant characteristics / Imaging acquisition
  Clinical Profile           -> Language function / Whole-brain lesion load
  Brain Lesion Topography    -> Cerebral cortex (Schaefer 2018, Yeo 7 networks)
                                  -> <network> -> parcel        (fine)
                                  -> <network aggregate>        (coarse)
                             -> Subcortical nuclei (Tian S4)
                                  -> <structure> -> parcel      (fine)
                                  -> <structure aggregate>      (coarse)
                             -> Cerebellum (Nettekoven 2024)
                                  -> functional domain -> subregion -> parcel (fine)
                                  -> functional domain -> subregion aggregate (coarse)

Note: `_feature_specs.json`'s cortical labels are shifted by one ROI (a pre-existing
quirk in lesion.py's label indexing - ROI p1 is labeled "Background"); the one
affected ROI falls into an "Unassigned cortex" network rather than a real Yeo
network. This affects only the display name, never which voxels belong to which ROI.

Writes:
  ontology/subclass_structure_fine.json / .owl
  ontology/subclass_structure_coarse.json / .owl
  data/processed/_all_subjects_features_coarse.csv   group-level lesion ratios
  data/processed/_group_feature_specs.json           label/description/units per
                                                      coarse column

If the coarse feature table and its specs already exist on disk, the atlas is not
rebuilt (so the ontology can be regenerated without nibabel or network access).
"""

import json
import re
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import config
from validation.common import ontology as onto

ONTOLOGY_DIR = config.ROOT / "ontology"
PREVALENCE_MIN_SUBJECTS = 10  # fine-tier ROI filter, matches inspection.ipynb's threshold

ONTOLOGY_CONTEXT = (
    "NUMERACY_STROKE (OpenNeuro ds006533): 105 left-hemisphere chronic stroke "
    "survivors assessed on approximate and precise numeracy. Features are "
    "demographics, aphasia severity, whole-brain lesion volume, and per-region "
    "lesion overlap across a combined cortical (Schaefer-2018/Yeo-7), "
    "subcortical (Tian scale-IV), and cerebellar (Nettekoven-2024) parcellation. "
    "Predictors are organised into demographics/background, a clinical profile, and "
    "a brain-lesion-topography domain resolved down to individual parcels."
)

TABULAR_SPECS = {
    "age": {"label": "Age", "description": "Age of participant in years.",
            "stat_type": "numeric", "units": "years"},
    "education_years": {"label": "Education (years)",
                         "description": "Years of formal education completed.",
                         "stat_type": "numeric", "units": "years"},
    "image_modality": {"label": "Imaging modality",
                        "description": "Research-grade MR, clinical MR, or clinical CT acquisition.",
                        "stat_type": "nominal", "units": None},
    "aphasia_quotient": {"label": "Aphasia severity (WAB-R quotient)",
                          "description": "Western Aphasia Battery-Revised aphasia quotient "
                                          "(0-100 on the raw scale; higher is less impaired). "
                                          "Stored rank-based inverse-normal transformed.",
                          "stat_type": "numeric", "units": "score"},
    "lesion_volume": {"label": "Whole-brain lesion volume",
                       "description": "Proportion of the left hemisphere that is lesioned "
                                       "(log-transformed).",
                       "stat_type": "numeric", "units": "proportion"},
}
T1_COLS = ["age", "education_years", "image_modality"]

YEO7 = {"Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"}

DOMAIN_BY_GROUP = {
    "lesion_cortex": "LESION_CORTEX",
    "lesion_subcortex": "LESION_SUBCORTEX",
    "lesion_cerebellum": "LESION_CEREBELLUM",
    "lesion_summary": "LESION_SUMMARY",
}

# ---- abstract-level display names (factual, no over-claiming of function) ----
YEO_FULL = {
    "Vis": "Visual network", "SomMot": "Somatomotor network",
    "DorsAttn": "Dorsal attention network", "SalVentAttn": "Salience / ventral attention network",
    "Limbic": "Limbic network", "Cont": "Frontoparietal control network",
    "Default": "Default mode network", "Other": "Unassigned cortex",
}
TIAN_FULL = {
    "HIP": "Hippocampus", "THA": "Thalamus", "PUT": "Putamen", "CAU": "Caudate nucleus",
    "lAMY": "Amygdala (lateral)", "mAMY": "Amygdala (medial)", "AMY": "Amygdala",
    "NAc": "Nucleus accumbens", "GP": "Globus pallidus",
    "pGP": "Globus pallidus (posterior)", "aGP": "Globus pallidus (anterior)",
    "Other": "Other subcortex",
}
CEREB_DOMAIN_FULL = {
    "M": "Cerebellar functional domain M", "A": "Cerebellar functional domain A",
    "D": "Cerebellar functional domain D", "S": "Cerebellar functional domain S",
    "T": "Cerebellar functional domain T",
}


def _cortex_subgroup(name: str) -> str:
    parts = name.split("_")
    return parts[2] if len(parts) > 2 and parts[2] in YEO7 else "Other"


def _subcortex_subgroup(name: str) -> str:
    return name.split("-")[0] if "-" in name else "Other"


def _cerebellum_subgroup(name: str) -> str:
    m = re.match(r"^([A-Za-z]+\d+)", name)
    return m.group(1) if m else "Other"


SUBGROUP_FN = {
    "lesion_cortex": _cortex_subgroup,
    "lesion_subcortex": _subcortex_subgroup,
    "lesion_cerebellum": _cerebellum_subgroup,
}


def roi_subdomain(spec: dict) -> str:
    fn = SUBGROUP_FN.get(spec.get("group"))
    if fn is None:
        return "whole_brain"
    name = spec["label"].replace("Lesion overlap: ", "")
    return fn(name)


def _seg(sid: str, label: str, definition: str = "") -> dict:
    return {"id": sid, "label": label, "definition": definition}


# --------------------------------------------------------------------------- #
# Abstract path assignment (domain first). This is the whole point of step 03:
# turn a flat feature id into a place in a clean, deep, reproducible hierarchy.
# --------------------------------------------------------------------------- #
_DEMO = _seg("DEMOGRAPHICS_AND_BACKGROUND", "Demographics and Background",
             "Participant demographics and imaging-acquisition covariates.")
_CLIN = _seg("CLINICAL_PROFILE", "Clinical Profile",
             "Behavioural and whole-brain clinical severity measures.")
_TOPO = _seg("LESION_TOPOGRAPHY", "Brain Lesion Topography",
             "Per-region proportion of tissue lesioned across a combined cortical, "
             "subcortical and cerebellar parcellation.")


def assign_path(feat: dict) -> list:
    """Return the abstract ontology path (segments, domain first) for a feature."""
    fid = feat["id"]

    # ---- tabular demographics / clinical ----
    if fid in ("age", "education_years"):
        return [_DEMO, _seg("participant_characteristics", "Participant characteristics")]
    if fid == "image_modality":
        return [_DEMO, _seg("imaging_acquisition", "Imaging acquisition")]
    if fid == "aphasia_quotient":
        return [_CLIN, _seg("language_function", "Language function")]
    if fid in ("lesion_volume", "lesion_total_voxels", "lesion_total_volume_mm3"):
        return [_CLIN, _seg("whole_brain_lesion_load", "Whole-brain lesion load",
                            "Aggregate lesion burden over the whole brain.")]

    group = feat.get("domain") or DOMAIN_BY_GROUP.get(feat.get("group"), "")
    subdomain = feat.get("subdomain") or feat.get("subdomain_hint") or "Other"
    is_aggregate = fid.startswith("lesion_ratio_group_")

    if group == "LESION_CORTEX":
        base = [_TOPO, _seg("cerebral_cortex", "Cerebral cortex (Schaefer 2018, Yeo 7 networks)")]
        net = YEO_FULL.get(subdomain, f"{subdomain} network")
        if is_aggregate:  # coarse: network aggregate is a leaf directly under cortex
            return base
        return base + [_seg(f"net_{subdomain.lower()}", net)]

    if group == "LESION_SUBCORTEX":
        base = [_TOPO, _seg("subcortical_nuclei", "Subcortical nuclei (Tian scale-IV)")]
        struct = TIAN_FULL.get(subdomain, subdomain)
        if is_aggregate:
            return base
        return base + [_seg(f"struct_{subdomain.lower()}", struct)]

    if group == "LESION_CEREBELLUM":
        base = [_TOPO, _seg("cerebellum", "Cerebellum (Nettekoven 2024 functional atlas)")]
        letter = (subdomain[:1] or "X").upper()
        dom = CEREB_DOMAIN_FULL.get(letter, f"Cerebellar functional domain {letter}")
        dom_seg = _seg(f"cbdom_{letter.lower()}", dom,
                       "One of the Nettekoven 2024 cerebellar functional domains "
                       "(letter-coded); subregions are numbered within each domain.")
        if is_aggregate:  # coarse: subregion aggregate leaf under its domain letter
            return base + [dom_seg]
        return base + [dom_seg, _seg(f"cbsub_{subdomain.lower()}", f"Subregion {subdomain}")]

    # whole-brain summary aggregates or anything unmapped
    return [_CLIN, _seg("whole_brain_lesion_load", "Whole-brain lesion load")]


def build_tabular_features() -> list:
    return [
        {"id": col, "label": meta["label"], "definition": meta["description"],
         "stat_type": meta["stat_type"], "units": meta["units"]}
        for col, meta in TABULAR_SPECS.items()
    ]


def build_fine_lesion_features(specs: dict, prevalence: pd.Series) -> list:
    feats = []
    for col, spec in specs.items():
        domain = DOMAIN_BY_GROUP.get(spec.get("group"))
        if domain is None:
            continue
        if col.startswith("lesion_ratio_p") and prevalence.get(col, 0) <= PREVALENCE_MIN_SUBJECTS:
            continue
        subdomain = roi_subdomain(spec) if col.startswith("lesion_ratio_p") else "whole_brain"
        feats.append({
            "id": col, "label": spec["label"], "definition": spec.get("description", ""),
            "stat_type": spec.get("stat_type", "numeric"), "units": spec.get("units"),
            "domain": domain, "subdomain": subdomain,
        })
    return feats


def build_group_aggregates(specs: dict, roi_total_voxels: dict, ratio_df: pd.DataFrame):
    """Voxel-weighted group-level lesion ratio per (domain, subgroup), across subjects.

    Returns (new_columns: {col_name: pd.Series}, feature_dicts: list).
    """
    members = defaultdict(list)  # (domain, subgroup) -> [(col, voxels)]
    for col, spec in specs.items():
        if not col.startswith("lesion_ratio_p"):
            continue
        domain = DOMAIN_BY_GROUP.get(spec.get("group"))
        if domain is None:
            continue
        roi_id = int(col.replace("lesion_ratio_p", ""))
        voxels = roi_total_voxels.get(roi_id)
        if not voxels:
            continue
        subgroup = roi_subdomain(spec)
        members[(domain, subgroup)].append((col, voxels))

    new_cols: dict = {}
    feats: list = []
    for (domain, subgroup), cols_voxels in members.items():
        cols = [c for c, _ in cols_voxels]
        weights = np.array([v for _, v in cols_voxels], dtype=float)
        total_voxels = weights.sum()
        ratios = ratio_df[cols].to_numpy(dtype=float)
        lesioned = ratios * weights[np.newaxis, :]
        group_ratio = lesioned.sum(axis=1) / total_voxels
        col_name = f"lesion_ratio_group_{domain}_{subgroup}".lower()
        new_cols[col_name] = pd.Series(group_ratio, index=ratio_df.index)
        region_label = domain.replace("LESION_", "").title()
        feats.append({
            "id": col_name,
            "label": f"Lesion overlap: {subgroup} ({region_label})",
            "definition": f"Voxel-weighted lesion overlap ratio aggregated across the "
                           f"{subgroup} {region_label.lower()} group ({len(cols)} ROIs).",
            "stat_type": "numeric", "units": "proportion",
            "domain": domain, "subdomain": subgroup,
        })
    return new_cols, feats


def build_ontology(features: list) -> dict:
    """Assign abstract paths to every feature, then build the children-schema tree."""
    for feat in features:
        feat["path"] = assign_path(feat)
    tree = onto.build_ontology_tree(features, config.DATASET_NAME, ONTOLOGY_CONTEXT, llm=None)
    return tree


def _summary(tree: dict) -> str:
    return (f"{len(tree['domains'])} domains, {tree['n_features']} leaves, "
            f"max depth {tree['repair_stats']['max_depth']}")


def _load_or_build_coarse(lesion_specs: dict, raw_df: pd.DataFrame, transformed_df: pd.DataFrame):
    """Return (group_specs, group_feats). Reuse on-disk coarse table if present, else
    build it from the atlas (requires nibabel + cached atlas downloads)."""
    coarse_csv = config.PROCESSED_DIR / "_all_subjects_features_coarse.csv"
    group_specs_path = config.PROCESSED_DIR / "_group_feature_specs.json"
    if coarse_csv.exists() and group_specs_path.exists():
        group_specs = json.loads(group_specs_path.read_text())
        group_feats = [
            {"id": cid, "label": meta["label"], "definition": meta.get("description", ""),
             "stat_type": meta.get("stat_type", "numeric"), "units": meta.get("units"),
             "domain": (meta.get("group") or "").upper(), "subdomain": meta.get("subdomain_hint", "Other")}
            for cid, meta in group_specs.items()
        ]
        print(f"[03] Reusing existing coarse feature table ({len(group_feats)} coarse features)")
        return group_specs, group_feats

    print("[03] Building combined atlas to get ROI voxel counts for group aggregation ...")
    import nibabel as nib
    from validation.datasets.NUMERACY_STROKE.utils import lesion
    reference_img = nib.load(config.lesion_mask_path(config.REFERENCE_SUBJECT))
    combined, roi_names, _ = lesion.build_combined_atlas(
        reference_img, config.ATLAS_CACHE_DIR,
        n_rois=config.N_ROIS, yeo_networks=config.YEO_NETWORKS, resolution_mm=config.RESOLUTION_MM,
    )
    roi_total_voxels = lesion.roi_voxel_counts(combined, roi_names)
    ratio_df = raw_df.set_index("participant_id")
    group_cols, group_feats = build_group_aggregates(lesion_specs, roi_total_voxels, ratio_df)

    coarse_df = transformed_df[["participant_id"] + list(TABULAR_SPECS) +
                                ["lesion_total_voxels", "lesion_total_volume_mm3"]].copy().set_index("participant_id")
    for col_name, series in group_cols.items():
        coarse_df[col_name] = series
    coarse_df.reset_index().to_csv(coarse_csv, index=False)

    group_specs = {f["id"]: {"label": f["label"], "description": f["definition"],
                              "stat_type": f["stat_type"], "units": f["units"],
                              "group": f["domain"].lower(), "subdomain_hint": f["subdomain"]}
                   for f in group_feats}
    for col in ("lesion_total_voxels", "lesion_total_volume_mm3"):
        if col in lesion_specs:
            group_specs[col] = lesion_specs[col]
    group_specs_path.write_text(json.dumps(group_specs, indent=2))
    return group_specs, group_feats


def main() -> None:
    with open(config.PROCESSED_DIR / "_feature_specs.json") as f:
        lesion_specs = json.load(f)
    raw_df = pd.read_csv(config.PROCESSED_DIR / "_all_subjects_features.csv")
    transformed_df = pd.read_csv(config.PROCESSED_DIR / "_all_subjects_features_transformed.csv")

    lesion_ratio_cols = [c for c in raw_df.columns if c.startswith("lesion_ratio_p")]
    prevalence = (raw_df[lesion_ratio_cols] > 0).sum(axis=0)

    ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)

    # ---- fine ontology (per-parcel leaves) ----
    fine_features = build_tabular_features() + build_fine_lesion_features(lesion_specs, prevalence)
    print(f"[03] Fine granularity: {len(fine_features)} features "
          f"(ROIs affected in >{PREVALENCE_MIN_SUBJECTS} subjects, plus tabular + whole-brain totals)")
    fine_ontology = build_ontology(fine_features)
    onto.write_subclass_json(fine_ontology, ONTOLOGY_DIR / "subclass_structure_fine.json")
    onto.write_owl(fine_ontology, ONTOLOGY_DIR / "subclass_structure_fine.owl")
    print(f"[03] Fine ontology: {_summary(fine_ontology)}")

    # ---- coarse ontology (network / structure / subregion aggregate leaves) ----
    _, group_feats = _load_or_build_coarse(lesion_specs, raw_df, transformed_df)
    whole_brain_feats = [f for f in build_fine_lesion_features(lesion_specs, prevalence)
                          if f["subdomain"] == "whole_brain"]
    coarse_features = build_tabular_features() + group_feats + whole_brain_feats
    print(f"[03] Coarse granularity: {len(coarse_features)} features")
    coarse_ontology = build_ontology(coarse_features)
    onto.write_subclass_json(coarse_ontology, ONTOLOGY_DIR / "subclass_structure_coarse.json")
    onto.write_owl(coarse_ontology, ONTOLOGY_DIR / "subclass_structure_coarse.owl")
    print(f"[03] Coarse ontology: {_summary(coarse_ontology)}")

    print("[03] Wrote ontology/subclass_structure_{fine,coarse}.json/.owl")


if __name__ == "__main__":
    main()
