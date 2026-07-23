#!/usr/bin/env python3
"""
Step 04 - project each cumulative tier onto its ontology and emit COMPASS files.

Tiers mirror AOMIC_ID1000's cumulative-feature-group naming (T1_demographics,
T2_personality, ... in validation/datasets/AOMIC_ID1000/pipeline/config.py):

  T1_demographics   age, education_years, image_modality
  T2_aphasia        T1 + aphasia_quotient, lesion_volume (single clinical/
                    behavioral scalars from participants.tsv, not yet the
                    atlas-derived imaging features)
  T3_lesion_fine    T2 + prevalence-filtered per-ROI lesion features (step 03)
  T3_lesion_coarse  T2 + network/structure-level lesion aggregates (step 03)

Each tier is built for both targets (approximate_numeracy, precise_numeracy)
and both cohort modes:

  all_shared  - all 105 subjects, real sub-XXX ids, z-scored against the full
                cohort (this is the user's own dataset, not a blinded benchmark).
  blinded     - a seeded, target-blind 20-subject evaluation subset (predictor-
                completeness filtered) renamed to eval-XXXX, z-scored against the
                disjoint ~85-subject reference cohort - mirrors AOMIC_ID1000's
                leakage-safe design, scaled down for this dataset's n=105.

4 tiers x 2 targets x 2 cohort modes = 16 tier directories. Every participant
gets the four COMPASS files (data_overview.json, hierarchical_deviation_map.json,
multimodal_data.json, non_numerical_data.txt) via the existing, unmodified
validation.common.{deviation,compass_writer,tiers} - same mechanism
AOMIC_ID1000's 03_build_compass_inputs.py already uses.

FA1/FA2 factor scores are excluded from every tier as predictors: per
participants.json they're derived from the same math/language task battery as
both DVs (confirmed via the notebook's correlation matrix: r=0.85-0.97 with one
DV or the other), i.e. circular, same reasoning AOMIC uses to exclude its IST
subscales. aphasia_quotient stays a predictor (r=0.26 vs 0.78 with the two DVs -
an asymmetry that's itself a finding, not a reason to model it as a third
target).

Writes:
  compass_inputs/<tier_id>/<participant_id>/{data_overview.json,
    hierarchical_deviation_map.json, multimodal_data.json, non_numerical_data.txt}
  compass_inputs/tiers.json                 summary of all 16 tiers
  results/subset_<target>.json              blinded-tier ground truth (per target)
"""

import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import config
from validation.common import compass_writer, deviation, tiers as tiermod

RESULTS_DIR = config.ROOT / "results"
INPUTS_DIR = config.ROOT / "compass_inputs"
ONTOLOGY_DIR = config.ROOT / "ontology"

RANDOM_SEED = 42
BLINDED_SUBSET_SIZE = 20
PREVALENCE_MIN_SUBJECTS = 10  # must match 03_build_ontology.py's fine-tier filter

TARGETS = ["approximate_numeracy", "precise_numeracy"]
TARGET_LABELS = {
    "approximate_numeracy": "Z-score of performance on a computer-based dot-comparison task "
                            "(the non-symbolic Approximate Number System)",
    "precise_numeracy": "composite Z-score across several precise, symbolic numeracy tasks "
                        "(WAB number items, number writing/dictation, calculation)",
}

# Only the TRANSFORMED feature tier is ingested (see the dataset README): the two
# scalar clinical covariates below are already standardized on disk, so their units
# are labelled honestly as standardized transforms rather than their raw native
# scale. The per-region lesion-overlap features remain raw proportions (0-1).
TABULAR_SPECS = {
    "age": {"label": "Age", "stat_type": "numeric", "units": "years"},
    "education_years": {"label": "Education (years)", "stat_type": "numeric", "units": "years"},
    "image_modality": {"label": "Imaging modality", "stat_type": "nominal", "units": None},
    "aphasia_quotient": {"label": "Aphasia severity (WAB-R quotient, standardized)",
                          "stat_type": "numeric",
                          "units": "standardized units (rank inverse-normal; higher = less aphasia)"},
    "lesion_volume": {"label": "Whole-brain lesion volume (standardized)", "stat_type": "numeric",
                       "units": "standardized units (log lesion fraction of the left hemisphere)"},
}
T1_COLS = ["age", "education_years", "image_modality"]
T2_COLS = T1_COLS + ["aphasia_quotient", "lesion_volume"]


def load_all_frames():
    """One merged frame with every column any tier could need, plus per-granularity specs."""
    tabular = pd.read_csv(config.PROCESSED_DIR / "_all_subjects_features_transformed.csv")
    raw = pd.read_csv(config.PROCESSED_DIR / "_all_subjects_features.csv")
    coarse = pd.read_csv(config.PROCESSED_DIR / "_all_subjects_features_coarse.csv")

    lesion_cols_all = [c for c in raw.columns if c.startswith("lesion_ratio_p")]
    prevalence = (raw[lesion_cols_all] > 0).sum(axis=0)
    fine_lesion_cols = [c for c in lesion_cols_all if prevalence[c] > PREVALENCE_MIN_SUBJECTS]
    fine_lesion_cols += ["lesion_total_voxels", "lesion_total_volume_mm3"]
    # whole-brain totals are identical in both CSVs (carried over verbatim in step 03) -
    # keep them only via fine_lesion_cols/raw to avoid a merge column collision.
    coarse_lesion_cols = [
        c for c in coarse.columns
        if c not in ["participant_id"] + list(TABULAR_SPECS) + ["lesion_total_voxels", "lesion_total_volume_mm3"]
    ]

    with open(config.PROCESSED_DIR / "_feature_specs.json") as f:
        fine_specs_all = json.load(f)
    with open(config.PROCESSED_DIR / "_group_feature_specs.json") as f:
        coarse_specs_all = json.load(f)

    merged = (
        tabular[["participant_id"] + list(TABULAR_SPECS)]
        .merge(raw[["participant_id"] + TARGETS + fine_lesion_cols], on="participant_id")
        .merge(coarse[["participant_id"] + coarse_lesion_cols], on="participant_id")
    )
    specs = {
        **TABULAR_SPECS,
        **{c: fine_specs_all[c] for c in fine_lesion_cols if c in fine_specs_all},
        **{c: coarse_specs_all[c] for c in coarse_lesion_cols if c in coarse_specs_all},
    }
    return merged, fine_lesion_cols, coarse_lesion_cols, specs


def select_blinded_subset(merged_df: pd.DataFrame, target: str) -> list:
    eligible = merged_df.dropna(subset=[target])
    ids = sorted(eligible["participant_id"].astype(str).tolist())
    rng = np.random.default_rng(RANDOM_SEED)
    n = min(BLINDED_SUBSET_SIZE, len(ids))
    return sorted(rng.choice(ids, size=n, replace=False).tolist())


SCALE_NOTE = (
    "SCALE GUIDE (how to read this record): the two numeracy phenotypes are already "
    "population Z-scores (0 = the stroke-cohort average, +/-1 = one SD; higher = better "
    "numeracy). Predictors are the standardized TRANSFORMED tier only: age and education "
    "are raw years; aphasia_quotient is a standardized WAB-R quotient (higher = less "
    "aphasia); whole-brain lesion volume is a standardized log lesion fraction; every "
    "per-region 'lesion overlap' feature is a raw proportion from 0 (spared) to 1 (fully "
    "lesioned). The (High/Low/...) qualifier on each feature is this participant's cohort "
    "deviation, not part of the raw value."
)


def build_target_note(target: str, reference_df: pd.DataFrame) -> str:
    ref_target = pd.to_numeric(reference_df[target], errors="coerce").dropna()
    mean, sd = float(ref_target.mean()), float(ref_target.std(ddof=0))
    return (
        f"Prediction target: {target}, a {TARGET_LABELS[target]}. In the reference split, "
        f"mean={mean:.3f}, sd={sd:.3f} native Z-score units. Predict one numeric value on "
        f"that native scale. No {'/'.join(t for t in TARGETS if t != target)} value is a "
        f"reliable proxy - this dataset's whole point is their differential relationship to "
        f"language/lesion features.\n\n" + SCALE_NOTE
    )


def write_tier(tier_id, target, cohort_mode, ontology, merged_df, predictor_cols, specs):
    allowed = set(predictor_cols)
    projected = tiermod.project_ontology(ontology, allowed)
    if projected["n_features"] != len(predictor_cols):
        missing = allowed - set(projected["column_index"])
        raise SystemExit(f"[04] {tier_id}: ontology projection missing columns: {sorted(missing)}")

    if cohort_mode == "blinded":
        chosen = select_blinded_subset(merged_df, target)
        reference_df = merged_df[~merged_df["participant_id"].astype(str).isin(chosen)]
        write_ids = chosen
    else:
        reference_df = merged_df
        write_ids = sorted(merged_df["participant_id"].astype(str).tolist())

    target_note = build_target_note(target, reference_df)
    ref = deviation.ReferenceModel(specs, mode="cohort").fit(reference_df[predictor_cols])

    tier_dir = INPUTS_DIR / tier_id
    tier_dir.mkdir(parents=True, exist_ok=True)
    expected = set(
        (f"eval-{i:04d}" for i in range(1, len(write_ids) + 1)) if cohort_mode == "blinded" else write_ids
    )
    for old_dir in tier_dir.iterdir():
        if old_dir.is_dir() and old_dir.name not in expected:
            import shutil
            shutil.rmtree(old_dir)

    records = []
    for index, source_id in enumerate(write_ids, 1):
        row = merged_df.loc[merged_df["participant_id"].astype(str) == source_id].iloc[0]
        participant_id = f"eval-{index:04d}" if cohort_mode == "blinded" else source_id
        encoded = ref.encode_participant(row)
        payloads = compass_writer.build_participant_payloads(
            participant_id=participant_id, ontology=projected, encoded=encoded,
            target_note=target_note, reference_mode="cohort",
        )
        if cohort_mode == "blinded":
            serialized = json.dumps(payloads)
            if source_id in serialized:
                raise AssertionError(f"source participant id leaked into payload for {participant_id}")
        compass_writer.write_participant(tier_dir / participant_id, payloads)
        gt = row.get(target)
        records.append({
            "participant_id": participant_id,
            "source_participant_id": source_id if cohort_mode == "blinded" else None,
            "ground_truth": None if pd.isna(gt) else round(float(gt), 4),
        })
    return records


def main() -> None:
    with open(ONTOLOGY_DIR / "subclass_structure_fine.json") as f:
        fine_ontology = json.load(f)
    with open(ONTOLOGY_DIR / "subclass_structure_coarse.json") as f:
        coarse_ontology = json.load(f)

    merged_df, fine_lesion_cols, coarse_lesion_cols, specs = load_all_frames()

    tier_levels = [
        ("T1_demographics", T1_COLS, fine_ontology),
        ("T2_aphasia", T2_COLS, fine_ontology),
        ("T3_lesion_fine", T2_COLS + fine_lesion_cols, fine_ontology),
        # brain-only lesion tier at the end: per-parcel lesion overlap only, no demographics/clinical.
        ("T4_lesion_brain_only", fine_lesion_cols, fine_ontology),
    ]
    _ = (coarse_lesion_cols, coarse_ontology)  # loaded for parity; coarse tier is no longer built

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    tier_meta = []
    for target in TARGETS:
        target_short = "approx" if target == "approximate_numeracy" else "precise"
        for level_name, predictor_cols, ontology in tier_levels:
            for cohort_mode in ("all_shared", "blinded"):
                tier_id = f"{target_short}_{level_name}_{cohort_mode.replace('_', '')}"
                level_specs = {c: specs[c] for c in predictor_cols}
                records = write_tier(
                    tier_id, target, cohort_mode, ontology, merged_df, predictor_cols, level_specs,
                )
                print(f"[04] {tier_id}: {len(records)} participants, {len(predictor_cols)} features "
                      f"-> compass_inputs/{tier_id}/")
                if cohort_mode == "blinded":
                    subset_path = RESULTS_DIR / f"subset_{target}.json"
                    with open(subset_path, "w") as f:
                        json.dump({
                            "dataset": config.DATASET_LABEL, "target": target,
                            "seed": RANDOM_SEED, "n_evaluation": len(records),
                            "n_reference": len(merged_df) - len(records),
                            "participants": records,
                        }, f, indent=2)
                tier_meta.append({
                    "id": tier_id, "target": target, "level": level_name,
                    "cohort_mode": cohort_mode, "n_participants": len(records),
                    "n_features": len(predictor_cols),
                })

    with open(INPUTS_DIR / "tiers.json", "w") as f:
        json.dump({"tiers": tier_meta}, f, indent=2)
    print(f"[04] Built {len(tier_meta)} tiers.")


if __name__ == "__main__":
    main()
