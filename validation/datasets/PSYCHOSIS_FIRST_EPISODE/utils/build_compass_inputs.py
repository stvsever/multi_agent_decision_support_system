#!/usr/bin/env python3
"""Prepare the psychosis COMPASS artifacts WITHOUT running the engine (no spend).

Builds the improved abstract ontology once (deterministic, LLM-free), emits its OWL
/ viewer / benchmark / QA artifacts, writes the four COMPASS files for a balanced,
leakage-safe evaluation cohort across all five tiers, and exports a ground-truth
annotations table covering EVERY recording so downstream validation has the target
phenotype for all subjects.

    python -m utils.build_compass_inputs                # 5 psychosis + 5 control
    python -m utils.build_compass_inputs --n 10         # 10 + 10

Feature values for all recordings live in results/{eeg,non_eeg}_features.csv; only a
balanced evaluation slice needs the (large) per-subject input folders, which the
validation notebook can extend on demand.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import compass_task as K
from utils import config as C


def _eeg_names() -> list[str]:
    """EEG feature columns straight from the table (avoids the heavy extractor import)."""
    cols = pd.read_csv(C.RESULTS_ROOT / "eeg_features.csv", nrows=1).columns
    return [c for c in cols if c != "recording_id"]


def _dictionary_map() -> dict[str, dict[str, str]]:
    path = C.RESULTS_ROOT / "non_eeg_feature_dictionary.csv"
    if not path.exists():
        return {}
    d = pd.read_csv(path)
    out = {}
    for _, r in d.iterrows():
        out[r["column_name"]] = {
            "label": str(r.get("column_name", "")).split("__")[-1].replace("_", " ").title(),
            "description": str(r.get("description", "") or ""),
            "source_table": str(r.get("source_table", "") or ""),
        }
    return out


def select_cohort(frame, eeg_names, n_per_group, seed):
    complete = frame[frame[eeg_names[0]].notna()].copy()
    label = "target__psychosis__case_control_label"
    rng = np.random.default_rng(seed)
    controls = sorted(rng.permutation(
        complete[complete[label] == "Control"]["recording_id"].tolist()).tolist())[:n_per_group]
    psy = complete[(complete[label] == "Psychosis") &
                   pd.to_numeric(complete[K.BPRS_TOTAL], errors="coerce").notna()].copy()
    psy["bprs"] = pd.to_numeric(psy[K.BPRS_TOTAL], errors="coerce")
    psy = psy.sort_values("bprs")
    idx = np.linspace(0, len(psy) - 1, n_per_group).round().astype(int)
    psychosis = psy.iloc[idx]["recording_id"].tolist()
    return psychosis, controls


def write_annotations(frame: pd.DataFrame, out_path: Path) -> int:
    """Ground-truth phenotype for every recording (validation targets, never inputs)."""
    label = "target__psychosis__case_control_label"
    records = []
    for _, row in frame.iterrows():
        gt = {c: (None if pd.isna(v := pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0])
                  else round(float(v), 3)) for c in K.ALL_OUTPUTS}
        records.append({
            "recording_id": row["recording_id"],
            "diagnosis": row.get(label),
            "ground_truth": gt,
        })
    payload = {
        "dataset": K.DATASET_LABEL,
        "task": "diagnosis (binary) -> BPRS total -> SAPS/SANS symptom globals",
        "class_labels": [K.CONTROL_LABEL, K.CASE_LABEL],
        "outputs": K.ALL_OUTPUTS,
        "n_recordings": len(records),
        "annotations": records,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return len(records)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5, help="subjects per group in the eval cohort")
    args = ap.parse_args()

    eeg_names = _eeg_names()
    eeg = pd.read_csv(C.RESULTS_ROOT / "eeg_features.csv")
    non_eeg = pd.read_csv(C.RESULTS_ROOT / "non_eeg_features.csv")
    frame = K.build_merged_frame(eeg, non_eeg)
    non_eeg_dict = _dictionary_map()
    groups = K.resolve_predictor_groups(non_eeg, eeg_names)
    tiers = K.build_tiers(groups)
    predictor_cols = sorted({c for t in tiers for c in t["columns"]})

    psychosis, controls = select_cohort(frame, eeg_names, args.n,
                                        int(C.load_config()["random_seed"]))
    eval_ids = psychosis + controls
    reference_ids = set(frame["recording_id"]) - set(eval_ids)
    print(f"[prep] cohort: {len(psychosis)} psychosis + {len(controls)} control "
          f"(reference n={len(reference_ids)})")

    # Deterministic ontology (all predictors carry explicit abstract paths) -> no LLM.
    K.ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    ontology, specs = K.build_ontology(frame, predictor_cols, non_eeg_dict, eeg_names, llm=None)
    K.write_ontology_artifacts(ontology, frame, specs, reference_ids, predictor_cols, llm=None)
    depth = ontology.get("repair_stats", {}).get("max_depth", "?")
    print(f"[prep] ontology: {len(ontology['domains'])} domains, {ontology['n_features']} leaves, "
          f"max depth {depth}; artifacts -> {K.ONTOLOGY_DIR}")

    stats = K.reference_target_stats(frame, reference_ids)
    global_instruction = K.build_global_instruction(stats)

    for tier in tiers:
        _, out_root = K.write_tier_inputs(tier, eval_ids, reference_ids, ontology, specs,
                                          frame, global_instruction)
        print(f"[prep] tier {tier['id']}: wrote {len(eval_ids)} subjects -> {out_root}")

    n_ann = write_annotations(frame, K.COMPASS_DIR / "annotations.json")
    (K.COMPASS_DIR / "eval_cohort.json").write_text(json.dumps(
        {"psychosis": psychosis, "control": controls,
         "reference_stats": stats, "global_instruction": global_instruction}, indent=2))
    print(f"[prep] annotations for all {n_ann} recordings -> {K.COMPASS_DIR/'annotations.json'}")
    print(f"[prep] eval cohort + global instruction -> {K.COMPASS_DIR/'eval_cohort.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
