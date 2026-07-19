#!/usr/bin/env python3
"""
Step 10 - extract FreeSurfer morphometry features (structural brain modality).

Downloads the small FreeSurfer stats tables for a reference cohort (the run subset
plus additional subjects, for stable z-scores) and parses them into a curated,
labeled morphometric feature table. Writes:

  brain/freesurfer/morphometry_features.csv   participant_id + ~35 features
  brain/freesurfer/morphometry_specs.json     feature label/description/type/units

Stats files are tiny text; the raw downloads are cached under brain/_cache/freesurfer.
"""

import json

import _bootstrap  # noqa: F401
import pandas as pd

import config
from validation.common import freesurfer as fs


def main() -> None:
    df = pd.read_csv(config.PARTICIPANTS_TSV, sep="\t", na_values=["n/a", "N/A", ""])
    subset = config.select_subset_ids(df)
    ref_ids = config.brain_reference_ids(df, config.BRAIN_MORPH_REF_SIZE, include=subset)
    print(f"[10] Downloading FreeSurfer stats for {len(ref_ids)} subjects "
          f"(reference cohort incl. {len(subset)} run-subset) ...")

    cache = config.BRAIN_CACHE_DIR / "freesurfer"
    downloaded = fs.download_many(ref_ids, config.ACCESSION, cache, workers=10)

    rows = []
    for pid in ref_ids:
        paths = downloaded.get(pid, {})
        feats = fs.extract_subject_features(paths)
        if feats:
            feats["participant_id"] = pid
            rows.append(feats)
    table = pd.DataFrame(rows).set_index("participant_id").sort_index()

    specs = fs.feature_specs()
    # Keep only spec'd feature columns that were actually extracted, in spec order.
    cols = [c for c in specs if c in table.columns]
    table = table[cols]

    config.FREESURFER_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.FREESURFER_DIR / "morphometry_features.csv")
    kept_specs = {c: specs[c] for c in cols}
    with open(config.FREESURFER_DIR / "morphometry_specs.json", "w") as f:
        json.dump(kept_specs, f, indent=2)

    subset_present = table.reindex(subset).notna().mean(axis=1).mean() if subset else 0
    print(f"[10] Extracted {len(cols)} morphometry features for {len(table)} subjects.")
    print(f"[10] Run-subset mean feature coverage: {subset_present*100:.0f}%")
    print(f"[10] Subdomains: {sorted(set(s['subdomain_hint'] for s in kept_specs.values()))}")
    print(f"[10] Wrote brain/freesurfer/morphometry_features.csv and morphometry_specs.json")


if __name__ == "__main__":
    main()
