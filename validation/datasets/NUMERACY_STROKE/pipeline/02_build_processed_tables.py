#!/usr/bin/env python3
"""
Step 02 - build two processed feature CSVs per subject: raw and transformed.

For every subject: every participants.tsv column (demographics, aphasia
quotient, factor scores, whole-brain lesion_volume, and the two DV/target
columns approximate_numeracy/precise_numeracy) plus one affected-voxel-ratio
feature per ROI of the combined cortex+subcortex+cerebellum parcellation
(see validation/common/lesion.py). Writes:

  data/processed/sub-XXX/raw/sub-XXX_features_raw.csv
      one row per subject, everything as extracted/carried over unchanged.
  data/processed/sub-XXX/transformed/sub-XXX_features_transformed.csv
      same columns, except aphasia_quotient/approximate_numeracy/
      precise_numeracy/lesion_volume are replaced by their recommended
      transform (validation/common/transforms.RECOMMENDED_TRANSFORM),
      fit once across this whole cohort (there is no train/test split at
      this stage - see inspection.ipynb for why each transform was chosen
      and the caveat on refitting per split for actual model training).
  data/processed/_all_subjects_features.csv                  raw, combined
  data/processed/_all_subjects_features_transformed.csv      transformed, combined
  data/processed/_feature_specs.json      label/description/units per
                                           generated lesion column
"""

import json

import _bootstrap  # noqa: F401
import nibabel as nib
import numpy as np
import pandas as pd

import config
from validation.datasets.NUMERACY_STROKE.utils import lesion
from validation.datasets.NUMERACY_STROKE.utils.transforms import RECOMMENDED_TRANSFORM, apply_recommended_transform


def _build_transformed_df(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Apply each column's recommended transform, fit once across the whole cohort.

    NaNs are left untouched (transformed on the non-null subset only, then
    reinserted at their original positions) since rank_int/yeojohnson/log all
    require complete input.
    """
    transformed_df = combined_df.copy()
    for col, kind in RECOMMENDED_TRANSFORM.items():
        if col not in transformed_df.columns:
            continue
        present = transformed_df[col].notna()
        transformed_values, params = apply_recommended_transform(col, transformed_df.loc[present, col])
        transformed_df.loc[present, col] = transformed_values
        print(f"[02] Transformed '{col}' with '{kind}'"
              + (f" (fitted lambda={params['lmbda']:.3f})" if "lmbda" in params else ""))
    return transformed_df


def main() -> None:
    df = pd.read_csv(config.PARTICIPANTS_TSV, sep="\t", na_values=["n/a", "N/A", ""])
    participant_cols = list(df.columns)

    ref_path = config.lesion_mask_path(config.REFERENCE_SUBJECT)
    if not ref_path.exists():
        raise SystemExit(
            f"[02] Reference lesion mask missing: {ref_path}. Run 01_download_lesion_data.py first."
        )
    reference_img = nib.load(ref_path)

    print("[02] Building combined cortex+subcortex+cerebellum atlas ...")
    combined, roi_names, roi_region = lesion.build_combined_atlas(
        reference_img, config.ATLAS_CACHE_DIR,
        n_rois=config.N_ROIS, yeo_networks=config.YEO_NETWORKS, resolution_mm=config.RESOLUTION_MM,
    )
    roi_total_voxels = lesion.roi_voxel_counts(combined, roi_names)
    nominal_max = config.N_ROIS + 54 + 128
    print(f"[02] Combined atlas: {len(roi_names)}/{nominal_max} nominal ROIs survived resampling "
          f"({sum(1 for r in roi_region.values() if r == 'cortex')} cortex, "
          f"{sum(1 for r in roi_region.values() if r == 'subcortex')} subcortex, "
          f"{sum(1 for r in roi_region.values() if r == 'cerebellum')} cerebellum).")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    specs = lesion.feature_specs(roi_names, roi_region)
    with open(config.PROCESSED_DIR / "_feature_specs.json", "w") as f:
        json.dump(specs, f, indent=2)

    lesion_feature_cols = list(specs.keys())
    ordered_cols = participant_cols + lesion_feature_cols
    all_rows = []
    processed, skipped = [], {}
    overlap_counts = []

    for subject in config.all_subjects():
        lesion_path = config.lesion_mask_path(subject)
        if not lesion_path.exists():
            skipped[subject] = "missing lesion mask (download step failed/incomplete)"
            continue

        participant_row = df.loc[df["participant_id"] == subject]
        if participant_row.empty:
            skipped[subject] = "not present in participants.tsv"
            continue

        feats = lesion.extract_subject_lesion_features(lesion_path, combined, roi_total_voxels)
        row = participant_row.iloc[0].to_dict()
        row.update(feats)

        row_df = pd.DataFrame([[row.get(c) for c in ordered_cols]], columns=ordered_cols)

        # stale layout from a previous run, if any
        old_flat_path = config.PROCESSED_DIR / subject / f"{subject}_features.csv"
        if old_flat_path.exists():
            old_flat_path.unlink()

        all_rows.append(row_df)
        processed.append(subject)
        n_nonzero = sum(1 for c in lesion_feature_cols if c.startswith("lesion_ratio_p") and row.get(c, 0) > 0)
        overlap_counts.append(n_nonzero)
        print(f"[02] {subject}: {n_nonzero} ROIs with nonzero lesion overlap")

    combined_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=ordered_cols)
    combined_df.to_csv(config.PROCESSED_DIR / "_all_subjects_features.csv", index=False)

    print("[02] Fitting recommended transforms across the full cohort ...")
    transformed_df = _build_transformed_df(combined_df)
    transformed_df.to_csv(config.PROCESSED_DIR / "_all_subjects_features_transformed.csv", index=False)

    for subject in processed:
        raw_dir = config.PROCESSED_DIR / subject / "raw"
        transformed_dir = config.PROCESSED_DIR / subject / "transformed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        transformed_dir.mkdir(parents=True, exist_ok=True)

        subj_raw = combined_df.loc[combined_df["participant_id"] == subject]
        subj_transformed = transformed_df.loc[transformed_df["participant_id"] == subject]
        subj_raw.to_csv(raw_dir / f"{subject}_features_raw.csv", index=False)
        subj_transformed.to_csv(transformed_dir / f"{subject}_features_transformed.csv", index=False)

    print(f"\n[02] Processed {len(processed)}/{len(config.all_subjects())} subjects.")
    if skipped:
        print(f"[02] Skipped {len(skipped)}: {skipped}")
    if overlap_counts:
        print(f"[02] Mean ROIs with nonzero overlap per subject: {np.mean(overlap_counts):.1f}")
    for col in config.TARGET_COLUMNS:
        n_missing = combined_df[col].isna().sum() if col in combined_df.columns else len(processed)
        print(f"[02] Missing target '{col}': {n_missing}/{len(processed)}")
    demo_cols = [c for c in participant_cols if c not in config.TARGET_COLUMNS + ["participant_id"]]
    for col in demo_cols:
        n_missing = combined_df[col].isna().sum() if col in combined_df.columns else len(processed)
        if n_missing:
            print(f"[02] Missing '{col}': {n_missing}/{len(processed)}")
    print(f"[02] Wrote data/processed/<subject>/raw/<subject>_features_raw.csv and "
          f"data/processed/<subject>/transformed/<subject>_features_transformed.csv "
          f"for {len(processed)} subjects, plus the two combined CSVs and _feature_specs.json")


if __name__ == "__main__":
    main()
