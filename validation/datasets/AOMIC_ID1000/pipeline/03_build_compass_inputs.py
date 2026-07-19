#!/usr/bin/env python3
"""
Step 03 - project the dataset onto each tier and emit COMPASS input files.

For every complexity tier that has all its feature groups available, this:
  * projects the master ontology onto the tier's feature columns,
  * fits the reference model over the full merged cohort (tier columns only),
  * writes the four engine files per subset participant under
    ``compass_inputs/<tier_id>/<sub>/``.

Ground-truth targets are recorded once in ``results/subset.json``.

Run for all available tiers (default) or one tier with ``--tier <id>``.
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import config
from validation.common import compass_writer, deviation, tiers as tiermod


def _available(tier, groups) -> bool:
    return all(groups.get(g) for g in tier["groups"])


def _write_tier(tier, ontology, df, subset_ids, groups):
    cols = tiermod.tier_columns(tier, groups)
    allowed = set(cols)
    projected = tiermod.project_ontology(ontology, allowed)
    specs = {c: s for c, s in config.all_feature_specs().items() if c in allowed}

    mode = deviation.resolve_reference_mode(
        requested=config.REFERENCE_MODE, n_participants=len(df),
        has_external_norms=bool(config.EXTERNAL_NORMS),
    )
    ref = deviation.ReferenceModel(specs, mode=mode)
    ref.fit(df, external_norms=config.EXTERNAL_NORMS)

    tier_dir = config.INPUTS_DIR / tier["id"]
    n_feat_total = projected["n_features"]
    if n_feat_total != len(cols):
        raise SystemExit(
            f"[03] {tier['id']}: ontology projection has {n_feat_total} leaves but tier "
            f"expects {len(cols)} columns. Rebuild the master ontology (step 02)."
        )
    for pid in subset_ids:
        row = df[df["participant_id"] == pid].iloc[0]
        encoded = ref.encode_participant(row)
        payloads = compass_writer.build_participant_payloads(
            participant_id=pid, ontology=projected, encoded=encoded,
            target_note=config.TARGET_SCALE_NOTE, reference_mode=mode,
        )
        compass_writer.write_participant(tier_dir / pid, payloads)
    n_dom = len(projected["domains"])
    print(f"[03] {tier['id']:22} {n_feat_total:3} features, {n_dom} domains -> "
          f"compass_inputs/{tier['id']}/ ({len(subset_ids)} subjects)")
    return {"id": tier["id"], "label": tier["label"], "n_features": n_feat_total,
            "n_domains": n_dom, "groups": tier["groups"], "reference_mode": mode}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=None, help="build a single tier by id")
    args = ap.parse_args()

    with open(config.ONTOLOGY_DIR / "subclass_structure.json") as f:
        ontology = json.load(f)
    df = config.load_merged_frame()
    groups = config.feature_groups()
    subset_ids = config.select_subset_ids(df)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subset_records = []
    for pid in subset_ids:
        gt = float(pd.to_numeric(pd.Series([df[df["participant_id"] == pid][config.TARGET["column"]].iloc[0]]),
                                 errors="coerce").iloc[0])
        subset_records.append({"participant_id": pid, "ground_truth": round(gt, 2)})
    with open(config.RESULTS_DIR / "subset.json", "w") as f:
        json.dump({"dataset": config.DATASET_LABEL, "target": config.TARGET,
                   "participants": subset_records}, f, indent=2)

    selected = [t for t in config.TIERS if (args.tier is None or t["id"] == args.tier)]
    tier_meta = []
    for tier in selected:
        if not _available(tier, groups):
            missing = [g for g in tier["groups"] if not groups.get(g)]
            print(f"[03] {tier['id']:22} SKIPPED (missing groups: {missing})")
            continue
        tier_meta.append(_write_tier(tier, ontology, df, subset_ids, groups))

    with open(config.INPUTS_DIR / "tiers.json", "w") as f:
        json.dump({"subset": subset_ids, "tiers": tier_meta}, f, indent=2)
    print(f"[03] Built {len(tier_meta)} tiers for {len(subset_ids)} subjects.")


if __name__ == "__main__":
    main()
