#!/usr/bin/env python3
"""
Step 03 - build fine and coarse master ontologies (LLM-labeled, deterministically
structured) plus the coarse group-level lesion feature table.

Structure (domain/subdomain assignment) is deterministic, derived from data already
on disk (validation/datasets/NUMERACY_STROKE/data/processed/_feature_specs.json's
`group` field + a parse of each ROI's label string) - it needs no judgment call,
since it's just the atlas hierarchy (cortex -> Yeo network, subcortex -> Tian
structure, cerebellum -> Nettekoven domain) that already exists by construction.
A single LLM call per granularity (fine/coarse) then writes human-readable parent
labels/definitions for those domains/subdomains, via the existing
validation.common.ontology.build_labeled_ontology - unmodified, same mechanism
AOMIC_ID1000 already uses.

Note: `_feature_specs.json`'s cortical labels are shifted by one ROI (a
pre-existing quirk in lesion.py's label indexing - ROI p1 is labeled
"Background" instead of its real Schaefer parcel name, and the true final
parcel's name is simply never used). This doesn't affect which voxels belong to
which ROI id (only the display name), so the one affected ROI just falls into an
"Other" cortical subdomain here rather than a real Yeo network.

Writes:
  ontology/subclass_structure_fine.json / .owl
  ontology/subclass_structure_coarse.json / .owl
  data/processed/_all_subjects_features_coarse.csv   group-level lesion ratios
                                                      + transformed tabular columns
  data/processed/_group_feature_specs.json           label/description/units per
                                                      coarse column
"""

import json
import re
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import config
from validation.datasets.NUMERACY_STROKE.utils import lesion
from validation.common import ontology as onto
from validation.common.llm import OntologyLLM

ONTOLOGY_DIR = config.ROOT / "ontology"
ONTOLOGY_MODEL = "google/gemini-3.1-flash-lite"
PREVALENCE_MIN_SUBJECTS = 10  # fine-tier ROI filter, matches inspection.ipynb's threshold

ONTOLOGY_CONTEXT = (
    "NUMERACY_STROKE (OpenNeuro ds006533): 105 left-hemisphere chronic stroke "
    "survivors assessed on approximate and precise numeracy. Features are "
    "demographics, aphasia severity, whole-brain lesion volume, and per-region "
    "lesion overlap across a combined cortical (Schaefer-2018/Yeo-7), "
    "subcortical (Tian scale-IV), and cerebellar (Nettekoven-2024) parcellation. "
    "Keep the three lesion regions (cortex/subcortex/cerebellum) as separate "
    "domains from demographics/clinical covariates."
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
                          "description": "Western Aphasia Battery-Revised severity score "
                                          "(rank-based inverse normal transformed).",
                          "stat_type": "numeric", "units": "score"},
    "lesion_volume": {"label": "Whole-brain lesion volume",
                       "description": "Proportion of the left hemisphere that is lesioned "
                                       "(log-transformed).",
                       "stat_type": "numeric", "units": "proportion"},
}

YEO7 = {"Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"}

DOMAIN_BY_GROUP = {
    "lesion_cortex": "LESION_CORTEX",
    "lesion_subcortex": "LESION_SUBCORTEX",
    "lesion_cerebellum": "LESION_CEREBELLUM",
    "lesion_summary": "LESION_SUMMARY",
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


def build_tabular_features() -> list:
    return [
        {"id": col, "label": meta["label"], "definition": meta["description"],
         "stat_type": meta["stat_type"], "units": meta["units"],
         "domain": "DEMOGRAPHICS_AND_CLINICAL", "subdomain": "general"}
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


def main() -> None:
    with open(config.PROCESSED_DIR / "_feature_specs.json") as f:
        lesion_specs = json.load(f)
    raw_df = pd.read_csv(config.PROCESSED_DIR / "_all_subjects_features.csv")
    transformed_df = pd.read_csv(config.PROCESSED_DIR / "_all_subjects_features_transformed.csv")

    lesion_ratio_cols = [c for c in raw_df.columns if c.startswith("lesion_ratio_p")]
    prevalence = (raw_df[lesion_ratio_cols] > 0).sum(axis=0)

    llm = OntologyLLM(model=ONTOLOGY_MODEL, temperature=0.2)
    ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)

    # ---- fine ----
    fine_features = build_tabular_features() + build_fine_lesion_features(lesion_specs, prevalence)
    print(f"[03] Fine granularity: {len(fine_features)} features "
          f"(ROIs affected in >{PREVALENCE_MIN_SUBJECTS} subjects, plus tabular + whole-brain totals)")
    print(f"[03] Building fine ontology with {ONTOLOGY_MODEL} ...")
    fine_ontology = onto.build_labeled_ontology(fine_features, config.DATASET_NAME, ONTOLOGY_CONTEXT, llm)
    onto.write_subclass_json(fine_ontology, ONTOLOGY_DIR / "subclass_structure_fine.json")
    onto.write_owl(fine_ontology, ONTOLOGY_DIR / "subclass_structure_fine.owl")
    print(f"[03] Fine ontology: {len(fine_ontology['domains'])} domains, "
          f"{sum(len(d['subdomains']) for d in fine_ontology['domains'])} subdomains, "
          f"{fine_ontology['n_features']} leaves")

    # ---- coarse: group-level aggregates ----
    print("[03] Building combined atlas once to get ROI voxel counts for group aggregation ...")
    import nibabel as nib
    reference_img = nib.load(config.lesion_mask_path(config.REFERENCE_SUBJECT))
    combined, roi_names, roi_region = lesion.build_combined_atlas(
        reference_img, config.ATLAS_CACHE_DIR,
        n_rois=config.N_ROIS, yeo_networks=config.YEO_NETWORKS, resolution_mm=config.RESOLUTION_MM,
    )
    roi_total_voxels = lesion.roi_voxel_counts(combined, roi_names)

    ratio_df = raw_df.set_index("participant_id")
    group_cols, group_feats = build_group_aggregates(lesion_specs, roi_total_voxels, ratio_df)
    print(f"[03] Coarse granularity: {len(group_feats)} group-level lesion aggregates")

    coarse_df = transformed_df[["participant_id"] + list(TABULAR_SPECS) +
                                ["lesion_total_voxels", "lesion_total_volume_mm3"]].copy()
    coarse_df = coarse_df.set_index("participant_id")
    for col_name, series in group_cols.items():
        coarse_df[col_name] = series
    coarse_df = coarse_df.reset_index()
    coarse_df.to_csv(config.PROCESSED_DIR / "_all_subjects_features_coarse.csv", index=False)

    group_specs = {f["id"]: {"label": f["label"], "description": f["definition"],
                              "stat_type": f["stat_type"], "units": f["units"],
                              "group": f["domain"].lower(), "subdomain_hint": f["subdomain"]}
                   for f in group_feats}
    # whole-brain summary columns are shared verbatim between fine and coarse -
    # carry their specs over so they're resolvable from _group_feature_specs.json too.
    for col in ("lesion_total_voxels", "lesion_total_volume_mm3"):
        if col in lesion_specs:
            group_specs[col] = lesion_specs[col]
    with open(config.PROCESSED_DIR / "_group_feature_specs.json", "w") as f:
        json.dump(group_specs, f, indent=2)

    whole_brain_feats = [f for f in build_fine_lesion_features(lesion_specs, prevalence)
                          if f["subdomain"] == "whole_brain"]
    coarse_features = build_tabular_features() + group_feats + whole_brain_feats
    print(f"[03] Building coarse ontology with {ONTOLOGY_MODEL} ...")
    coarse_ontology = onto.build_labeled_ontology(coarse_features, config.DATASET_NAME, ONTOLOGY_CONTEXT, llm)
    onto.write_subclass_json(coarse_ontology, ONTOLOGY_DIR / "subclass_structure_coarse.json")
    onto.write_owl(coarse_ontology, ONTOLOGY_DIR / "subclass_structure_coarse.owl")
    print(f"[03] Coarse ontology: {len(coarse_ontology['domains'])} domains, "
          f"{sum(len(d['subdomains']) for d in coarse_ontology['domains'])} subdomains, "
          f"{coarse_ontology['n_features']} leaves")

    print(f"[03] Wrote ontology/subclass_structure_{{fine,coarse}}.json/.owl, "
          f"_all_subjects_features_coarse.csv, _group_feature_specs.json")


if __name__ == "__main__":
    main()
