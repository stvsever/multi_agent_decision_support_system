#!/usr/bin/env python3
"""
Step 11 - extract functional connectome features (movie-watching fMRI).

For each subject in the reference cohort: download fMRIPrep BOLD + confounds,
parcellate with Schaefer-100/Yeo-7, denoise, and reduce to 28 network-level FC
features. Raw BOLD is deleted after each subject to bound disk use. Writes:

  brain/connectome/connectome_features.csv    participant_id + 28 FC features
  brain/connectome/connectome_specs.json      feature label/description/type/units
  brain/connectome/network_fc/<sub>.npy       7x7 network FC matrix (for notebooks)

The reference cohort is smaller than morphometry's because fMRI is heavy; it is
still above the minimum needed for a stable cohort z-score.
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import config
from validation.common import connectome as conn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-size", type=int, default=config.BRAIN_CONN_REF_SIZE)
    ap.add_argument("--keep-bold", action="store_true", help="do not delete raw BOLD after use")
    args = ap.parse_args()

    df = pd.read_csv(config.PARTICIPANTS_TSV, sep="\t", na_values=["n/a", "N/A", ""])
    subset = config.select_subset_ids(df)
    ref_ids = config.brain_reference_ids(df, args.ref_size, include=subset)
    print(f"[11] Connectome extraction for {len(ref_ids)} subjects "
          f"(incl. {len(subset)} run-subset). Loading atlas ...")

    atlas = conn.load_atlas(n_rois=100, yeo_networks=7, resolution_mm=2)
    networks = conn.atlas_networks(atlas)

    cache = config.BRAIN_CACHE_DIR / "connectome"
    fc_dir = config.CONNECTOME_DIR / "network_fc"
    fc_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, pid in enumerate(ref_ids, 1):
        try:
            paths = conn.download_func(pid, config.ACCESSION, cache)
            if not paths.get("bold") or not paths.get("confounds"):
                print(f"[11] ({i}/{len(ref_ids)}) {pid}: missing files, skipped")
                continue
            feats, net_mat, _parcel_fc = conn.extract_subject_fc(paths, atlas, networks)
            feats["participant_id"] = pid
            rows.append(feats)
            np.save(fc_dir / f"{pid}.npy", net_mat)
            print(f"[11] ({i}/{len(ref_ids)}) {pid}: {len(feats)-1} FC features")
        except Exception as exc:  # noqa: BLE001
            print(f"[11] ({i}/{len(ref_ids)}) {pid}: FAILED {type(exc).__name__}: {exc}")
        finally:
            if not args.keep_bold:
                for key in ("bold", "mask"):
                    p = (paths or {}).get(key) if "paths" in dir() else None
                    try:
                        if p and p.exists():
                            p.unlink()
                    except Exception:
                        pass

    table = pd.DataFrame(rows).set_index("participant_id").sort_index()
    specs = conn.feature_specs()
    cols = [c for c in specs if c in table.columns]
    table = table[cols]

    config.CONNECTOME_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.CONNECTOME_DIR / "connectome_features.csv")
    kept_specs = {c: specs[c] for c in cols}
    with open(config.CONNECTOME_DIR / "connectome_specs.json", "w") as f:
        json.dump(kept_specs, f, indent=2)

    print(f"[11] Extracted {len(cols)} connectome features for {len(table)} subjects.")
    print(f"[11] Wrote brain/connectome/connectome_features.csv and connectome_specs.json")


if __name__ == "__main__":
    main()
