#!/usr/bin/env python3
"""
Step 03 - project the dataset onto the ontology and emit COMPASS input files.

Fits the reference model over the full cohort (mode auto-selected: cohort /
external / absolute), selects a deterministic subset for the live run, and writes
the four engine-format files per participant under ``compass_inputs/<sub>/``.
Ground-truth targets are recorded separately in ``results/subset.json`` and are
never placed in the participant inputs.
"""

import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import config
from validation.common import compass_writer, deviation


CORE_COMPLETE = ["NEO_N", "NEO_E", "NEO_O", "NEO_A", "NEO_C", "BAS_drive", "BIS", "STAI_T", "BMI"]


def _select_subset(df: pd.DataFrame) -> pd.DataFrame:
    target = config.TARGET["column"]
    ok = df[pd.to_numeric(df[target], errors="coerce").notna()].copy()
    ok = ok[pd.to_numeric(ok["BMI"], errors="coerce").fillna(0) > 0]
    for col in CORE_COMPLETE:
        ok = ok[pd.to_numeric(ok[col], errors="coerce").notna()]
    ok = ok.sort_values("participant_id")
    # Spread across the target distribution for a representative smoke test.
    tvals = pd.to_numeric(ok[target], errors="coerce")
    ok = ok.assign(_t=tvals).sort_values("_t")
    idx = np.linspace(0, len(ok) - 1, config.SUBSET_SIZE).round().astype(int)
    return ok.iloc[idx].drop(columns="_t").sort_values("participant_id")


def main() -> None:
    with open(config.ONTOLOGY_DIR / "subclass_structure.json") as f:
        ontology = json.load(f)
    with open(config.ONTOLOGY_DIR / "feature_manifest.json") as f:
        manifest = json.load(f)

    df = pd.read_csv(config.PARTICIPANTS_TSV, sep="\t", na_values=["n/a", "N/A", ""])

    mode = deviation.resolve_reference_mode(
        requested=config.REFERENCE_MODE,
        n_participants=len(df),
        has_external_norms=bool(config.EXTERNAL_NORMS),
    )
    print(f"[03] Reference mode: {config.REFERENCE_MODE} -> resolved '{mode}' (n={len(df)})")

    ref = deviation.ReferenceModel(config.FEATURE_SPECS, mode=mode)
    if mode == "absolute":
        # No cohort/external reference: attach LLM-estimated ranges so the engine
        # still sees High/Normal/Low context.
        from validation.common.llm import OntologyLLM
        print("[03] Absolute mode: estimating reference ranges via LLM ...")
        ranges = deviation.estimate_reference_ranges(
            OntologyLLM(model=config.ONTOLOGY_MODEL), manifest["predictors"]
        )
        ref.set_llm_ranges(ranges)
    ref.fit(df, external_norms=config.EXTERNAL_NORMS)

    subset = _select_subset(df)
    config.INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    subset_records = []
    for _, row in subset.iterrows():
        pid = str(row["participant_id"])
        encoded = ref.encode_participant(row)
        payloads = compass_writer.build_participant_payloads(
            participant_id=pid,
            ontology=ontology,
            encoded=encoded,
            target_note=config.TARGET_SCALE_NOTE,
            reference_mode=mode,
        )
        compass_writer.write_participant(config.INPUTS_DIR / pid, payloads)
        gt = float(pd.to_numeric(pd.Series([row[config.TARGET["column"]]]), errors="coerce").iloc[0])
        present = payloads["data_overview"]["domain_coverage"]
        n_present = sum(v["present_leaves"] for v in present.values())
        n_total = sum(v["total_leaves"] for v in present.values())
        subset_records.append({
            "participant_id": pid,
            "ground_truth": round(gt, 2),
            "present_features": n_present,
            "total_features": n_total,
        })
        print(f"[03] {pid}: target={gt:.1f}  features {n_present}/{n_total} present")

    subset_path = config.RESULTS_DIR / "subset.json"
    with open(subset_path, "w") as f:
        json.dump({
            "dataset": config.DATASET_LABEL,
            "target": config.TARGET,
            "reference_mode": mode,
            "subset_size": len(subset_records),
            "participants": subset_records,
        }, f, indent=2)
    print(f"[03] Wrote {len(subset_records)} participant folders and results/subset.json")


if __name__ == "__main__":
    main()
