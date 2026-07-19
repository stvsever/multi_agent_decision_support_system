#!/usr/bin/env python3
"""
Step 01 - deterministic exploration of the raw feature structure.

Profiles every candidate predictor (type, missingness, distribution) plus the
target, across the tabular phenotype and any extracted brain features, and writes
``ontology/feature_manifest.json``. No LLM calls here.
"""

import json

import _bootstrap  # noqa: F401

import config
from validation.common import manifest as manifest_mod


def main() -> None:
    df = config.load_merged_frame()
    specs = config.all_feature_specs()
    manifest = manifest_mod.build_manifest(
        df=df,
        dataset_name=config.DATASET_LABEL,
        target=config.TARGET,
        feature_specs=specs,
        excluded=config.EXCLUDED_COLUMNS,
    )
    config.ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.ONTOLOGY_DIR / "feature_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    groups = config.feature_groups()
    print(f"[01] Participants: {manifest['n_participants']}")
    print(f"[01] Target: {manifest['target']['label']} "
          f"(mean {manifest['target']['mean']}, sd {manifest['target']['std']}, "
          f"range {manifest['target']['minimum']}-{manifest['target']['maximum']})")
    print(f"[01] Predictors profiled: {manifest['n_predictors']} across {len(groups)} groups")
    for gid, cols in groups.items():
        print(f"      {gid:20} {len(cols):3} features")
    print(f"[01] Wrote {out_path.name}")


if __name__ == "__main__":
    main()
