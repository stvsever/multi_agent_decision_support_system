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
import shutil

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import config
from validation.common import compass_writer, deviation, tiers as tiermod


def _available(tier, groups) -> bool:
    return all(groups.get(g) for g in tier["groups"])


def _write_tier(tier, ontology, df, reference_df, subset_records, groups, target_note):
    cols = tiermod.tier_columns(tier, groups)
    allowed = set(cols)
    projected = tiermod.project_ontology(ontology, allowed)
    specs = {c: s for c, s in config.all_feature_specs().items() if c in allowed}

    reference_feature_n_min = int(reference_df[cols].notna().sum().min())
    mode = deviation.resolve_reference_mode(
        requested=config.REFERENCE_MODE, n_participants=reference_feature_n_min,
        has_external_norms=bool(config.EXTERNAL_NORMS),
    )
    if mode == "cohort" and reference_feature_n_min < deviation.MIN_COHORT_N:
        raise SystemExit(
            f"[03] {tier['id']}: only {reference_feature_n_min} disjoint reference "
            f"participants cover every feature; need at least {deviation.MIN_COHORT_N}"
        )
    ref = deviation.ReferenceModel(specs, mode=mode)
    ref.fit(reference_df, external_norms=config.EXTERNAL_NORMS)

    tier_dir = config.INPUTS_DIR / tier["id"]
    tier_dir.mkdir(parents=True, exist_ok=True)
    expected_dirs = {rec["participant_id"] for rec in subset_records}
    for old_dir in tier_dir.iterdir():
        if old_dir.is_dir() and old_dir.name not in expected_dirs:
            shutil.rmtree(old_dir)
    n_feat_total = projected["n_features"]
    if n_feat_total != len(cols):
        raise SystemExit(
            f"[03] {tier['id']}: ontology projection has {n_feat_total} leaves but tier "
            f"expects {len(cols)} columns. Rebuild the master ontology (step 02)."
        )
    for rec in subset_records:
        evaluation_id = rec["participant_id"]
        source_id = rec["source_participant_id"]
        row = df[df["participant_id"] == source_id].iloc[0]
        encoded = ref.encode_participant(row)
        payloads = compass_writer.build_participant_payloads(
            participant_id=evaluation_id, ontology=projected, encoded=encoded,
            target_note=target_note, reference_mode=mode,
        )
        serialized = json.dumps(payloads)
        if source_id in serialized:
            raise AssertionError(f"source participant id leaked into payload for {evaluation_id}")
        compass_writer.write_participant(tier_dir / evaluation_id, payloads)
    n_dom = len(projected["domains"])
    print(f"[03] {tier['id']:22} {n_feat_total:3} features, {n_dom} domains -> "
          f"compass_inputs/{tier['id']}/ ({len(subset_records)} subjects)")
    return {"id": tier["id"], "label": tier["label"], "n_features": n_feat_total,
            "n_domains": n_dom, "groups": tier["groups"], "reference_mode": mode,
            "reference_feature_n_min": reference_feature_n_min,
            "evaluation_reference_disjoint": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=None, help="build a single tier by id")
    args = ap.parse_args()

    with open(config.ONTOLOGY_DIR / "subclass_structure.json") as f:
        ontology = json.load(f)
    df = config.load_merged_frame()
    groups = config.feature_groups()
    source_subset_ids = config.select_subset_ids(df)
    reference_df = df[~df["participant_id"].isin(source_subset_ids)].copy()

    excluded_predictors = set(config.EXCLUDED_COLUMNS).intersection(config.all_feature_specs())
    if excluded_predictors:
        raise AssertionError(f"excluded/target columns present in predictor specs: {excluded_predictors}")

    reference_target = pd.to_numeric(reference_df[config.TARGET["column"]], errors="coerce").dropna()
    reference_mean = float(reference_target.mean())
    reference_sd = float(reference_target.std(ddof=0))
    target_note = config.target_scale_note(reference_mean, reference_sd)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subset_records = []
    for index, source_id in enumerate(source_subset_ids, 1):
        evaluation_id = f"eval-{index:04d}"
        gt = float(pd.to_numeric(pd.Series([df[df["participant_id"] == source_id][config.TARGET["column"]].iloc[0]]),
                                 errors="coerce").iloc[0])
        subset_records.append({
            "participant_id": evaluation_id,
            "source_participant_id": source_id,
            "ground_truth": round(gt, 2),
        })
    with open(config.RESULTS_DIR / "subset.json", "w") as f:
        json.dump({"dataset": config.DATASET_LABEL, "target": config.TARGET,
                   "selection": {
                       "method": "seeded random target-blind draw after predictor completeness filtering",
                       "seed": config.RANDOM_SEED,
                       "n_evaluation": len(subset_records),
                       "evaluation_reference_disjoint": True,
                       "model_facing_ids_blinded": True,
                   },
                   "target_calibration": {
                       "source": "disjoint non-evaluation reference split",
                       "n": int(reference_target.size),
                       "native_mean": round(reference_mean, 6),
                       "native_sd": round(reference_sd, 6),
                       "iq_equivalent_mean": 100.0,
                       "iq_equivalent_sd": 15.0,
                       "formula": "100 + 15 * (native_IST - native_mean) / native_sd",
                       "caveat": "Interpretive cohort standardization, not an official IST norm conversion",
                   },
                   "agent_target_scale_note": target_note,
                   "participants": subset_records}, f, indent=2)

    selected = [t for t in config.TIERS if (args.tier is None or t["id"] == args.tier)]
    tier_meta = []
    for tier in selected:
        if not _available(tier, groups):
            missing = [g for g in tier["groups"] if not groups.get(g)]
            print(f"[03] {tier['id']:22} SKIPPED (missing groups: {missing})")
            continue
        tier_meta.append(_write_tier(
            tier, ontology, df, reference_df, subset_records, groups, target_note
        ))

    with open(config.INPUTS_DIR / "tiers.json", "w") as f:
        json.dump({"subset": [r["participant_id"] for r in subset_records],
                   "selection_seed": config.RANDOM_SEED,
                   "evaluation_reference_disjoint": True,
                   "tiers": tier_meta}, f, indent=2)
    print(f"[03] Built {len(tier_meta)} tiers for {len(subset_records)} blinded subjects.")


if __name__ == "__main__":
    main()
