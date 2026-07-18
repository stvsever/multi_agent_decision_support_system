#!/usr/bin/env python3
"""
Step 01 - deterministic exploration of the raw feature structure.

Profiles every candidate predictor (type, missingness, distribution) plus the
target, and writes ``ontology/feature_manifest.json``. No LLM calls here.
"""

import json

import _bootstrap  # noqa: F401  (sets sys.path)
import pandas as pd

import config
from validation.common import manifest as manifest_mod


def main() -> None:
    df = pd.read_csv(config.PARTICIPANTS_TSV, sep="\t", na_values=["n/a", "N/A", ""])
    manifest = manifest_mod.build_manifest(
        df=df,
        dataset_name=config.DATASET_LABEL,
        target=config.TARGET,
        feature_specs=config.FEATURE_SPECS,
        excluded=config.EXCLUDED_COLUMNS,
    )
    config.ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.ONTOLOGY_DIR / "feature_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[01] Participants: {manifest['n_participants']}")
    print(f"[01] Target: {manifest['target']['label']} "
          f"(mean {manifest['target']['mean']}, sd {manifest['target']['std']}, "
          f"range {manifest['target']['minimum']}-{manifest['target']['maximum']})")
    print(f"[01] Predictors profiled: {manifest['n_predictors']}")
    for p in manifest["predictors"]:
        extra = (f"mean={p['mean']}" if p["stat_type"] == "numeric" else f"cats={p['categories']}")
        print(f"      {p['column']:22} {p['stat_type']:9} coverage={p['coverage_pct']:5}%  {extra}")
    print(f"[01] Wrote {out_path.relative_to(config.ROOT.parent)}")


if __name__ == "__main__":
    main()
