#!/usr/bin/env python3
"""Run the five-tier COMPASS ladder on 10 subjects (5 psychosis, 5 control).

Builds the master ontology once, writes the four input files per subject per tier,
then runs the hierarchical prediction for every (tier, subject) pair in parallel
with an OpenRouter spend guard. Results are cached to
results/compass/ladder/predictions.json so notebook 03 can load and visualize them
without re-spending.

    python -m utils.run_ladder                 # 10 subjects x 5 tiers
    python -m utils.run_ladder --max-usd 3     # tighter spend cap
    python -m utils.run_ladder --tier T1_demographic_socioeconomic --n 2   # probe
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import compass_task as K
from utils import config as C
from utils.features import build_schema


def _credit_usage() -> float | None:
    from validation.common.llm import _load_key
    try:
        key = _load_key()
    except Exception:
        return None
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return float(json.loads(resp.read().decode()).get("data", {}).get("total_usage", 0.0))
    except Exception:
        return None


def select_cohort(frame: pd.DataFrame, eeg_names: list[str], n_per_group: int, seed: int):
    """Deterministic 5 psychosis + 5 control, all feature-eligible and rated where
    applicable, psychosis spread across BPRS severity."""
    complete = frame[frame[eeg_names[0]].notna()].copy()
    label = "target__psychosis__case_control_label"
    rng = np.random.default_rng(seed)
    controls = complete[complete[label] == "Control"]["recording_id"].tolist()
    controls = sorted(rng.permutation(controls).tolist())[:n_per_group]
    psy = complete[(complete[label] == "Psychosis") &
                   pd.to_numeric(complete[K.BPRS_TOTAL], errors="coerce").notna()].copy()
    psy["bprs"] = pd.to_numeric(psy[K.BPRS_TOTAL], errors="coerce")
    psy = psy.sort_values("bprs")
    idx = np.linspace(0, len(psy) - 1, n_per_group).round().astype(int)
    psychosis = psy.iloc[idx]["recording_id"].tolist()
    return psychosis, controls


def _worker(job: dict) -> dict:
    import contextlib
    import io as _io
    spec = K.build_task_spec()
    started = time.time()
    buf = _io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            pred = K.run_engine_on(Path(job["participant_dir"]), spec, job["global_instruction"],
                                   model=job["model"], max_iter=job["max_iter"])
        return {"tier": job["tier"], "recording_id": job["recording_id"],
                "true_group": job["true_group"], "ground_truth": job["ground_truth"],
                "prediction": pred, "duration_s": round(time.time() - started, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"tier": job["tier"], "recording_id": job["recording_id"],
                "true_group": job["true_group"], "ground_truth": job["ground_truth"],
                "error": f"{type(exc).__name__}: {exc}", "log_tail": buf.getvalue()[-1200:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5, help="subjects per group (default 5+5)")
    ap.add_argument("--model", default=K.ONTOLOGY_MODEL)
    ap.add_argument("--max-iter", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--tier", default=None)
    ap.add_argument("--rebuild-ontology", action="store_true")
    args = ap.parse_args()

    _, eeg_names = build_schema()
    eeg = pd.read_csv(C.RESULTS_ROOT / "eeg_features.csv")
    non_eeg = pd.read_csv(C.RESULTS_ROOT / "non_eeg_features.csv")
    frame = K.build_merged_frame(eeg, non_eeg)
    non_eeg_dict = _dictionary_map()
    groups = K.resolve_predictor_groups(non_eeg, eeg_names)
    tiers = K.build_tiers(groups)
    tiers = [t for t in tiers if not args.tier or t["id"] == args.tier]
    predictor_cols = sorted(set(c for t in K.build_tiers(groups) for c in t["columns"]))

    psychosis, controls = select_cohort(frame, eeg_names, args.n, int(C.load_config()["random_seed"]))
    eval_ids = psychosis + controls
    reference_ids = set(frame["recording_id"]) - set(eval_ids)
    label = "target__psychosis__case_control_label"
    truth = frame.set_index("recording_id")
    print(f"[ladder] cohort: {len(psychosis)} psychosis + {len(controls)} control")

    # Ontology (built once, cached) plus its full linguistic representation.
    K.ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    onto_path = K.ONTOLOGY_DIR / "subclass_structure.json"
    owl_path = K.ONTOLOGY_DIR / "psychosis_fep.owl"
    if onto_path.exists() and not args.rebuild_ontology:
        ontology = json.loads(onto_path.read_text())
        specs = K.build_specs(predictor_cols, non_eeg_dict, set(eeg_names))
        print("[ladder] loaded cached ontology")
        if not owl_path.exists():
            try:
                from validation.common.llm import OntologyLLM
                llm = OntologyLLM(model=args.model)
            except Exception:
                llm = None
            K.write_ontology_artifacts(ontology, frame, specs, reference_ids, predictor_cols, llm=llm)
            print("[ladder] emitted ontology artifacts (OWL, viewer, benchmark, QA report)")
    else:
        from validation.common.llm import OntologyLLM
        llm = OntologyLLM(model=args.model)
        t = time.time()
        ontology, specs = K.build_ontology(frame, predictor_cols, non_eeg_dict, eeg_names, llm=llm)
        report = K.write_ontology_artifacts(ontology, frame, specs, reference_ids, predictor_cols, llm=llm)
        print(f"[ladder] ontology built in {time.time()-t:.1f}s; "
              f"artifacts (OWL, viewer, benchmark, QA) -> {K.ONTOLOGY_DIR}")

    stats = K.reference_target_stats(frame, reference_ids)
    global_instruction = K.build_global_instruction(stats)

    # Write inputs for every tier, then build the job list.
    jobs = []
    for tier in tiers:
        _, out_root = K.write_tier_inputs(tier, eval_ids, reference_ids, ontology, specs,
                                          frame, global_instruction)
        for rid in eval_ids:
            gt = {c: (float(v) if pd.notna(v := pd.to_numeric(pd.Series([truth.loc[rid, c]]),
                     errors="coerce").iloc[0]) else None) for c in K.ALL_OUTPUTS}
            jobs.append({"tier": tier["id"], "recording_id": rid,
                         "true_group": truth.loc[rid, label],
                         "ground_truth": gt, "participant_dir": str(out_root / rid),
                         "global_instruction": global_instruction,
                         "model": args.model, "max_iter": args.max_iter})
    print(f"[ladder] {len(jobs)} runs ({len(tiers)} tiers x {len(eval_ids)} subjects)")

    usage0 = _credit_usage()
    if usage0 is not None:
        print(f"[ladder] OpenRouter usage so far ${usage0:.4f}; cap +${args.max_usd:.2f}")

    K.LADDER_DIR.mkdir(parents=True, exist_ok=True)
    rows, stop = [], False
    pool = ProcessPoolExecutor(max_workers=max(1, args.workers))
    it = iter(jobs)
    pending = {}

    def submit():
        try:
            nxt = next(it)
        except StopIteration:
            return False
        pending[pool.submit(_worker, nxt)] = nxt
        return True

    for _ in range(min(args.workers, len(jobs))):
        submit()
    done = 0
    try:
        while pending and not stop:
            finished, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for fut in finished:
                pending.pop(fut)
                done += 1
                row = fut.result()
                rows.append(row)
                tag = row.get("error") or (row["prediction"].get("diagnosis_label"))
                print(f"[ladder] {done}/{len(jobs)} {row['tier']} {row['recording_id']} "
                      f"true={row['true_group']} -> {tag}", flush=True)
                usage = _credit_usage()
                if usage is not None and usage0 is not None and usage - usage0 >= args.max_usd:
                    print(f"[ladder] spend guard hit (+${usage-usage0:.2f}); stopping")
                    stop = True
                    break
                submit()
    finally:
        for fut in pending:
            fut.cancel()
        pool.shutdown(wait=True, cancel_futures=True)

    usage1 = _credit_usage()
    spend = (usage1 - usage0) if (usage0 is not None and usage1 is not None) else None
    payload = {
        "dataset": K.DATASET_LABEL, "model": args.model,
        "task": "diagnosis (binary) -> BPRS total -> SAPS/SANS globals",
        "outputs": K.ALL_OUTPUTS, "class_labels": [K.CONTROL_LABEL, K.CASE_LABEL],
        "tiers": [t["id"] for t in tiers], "cohort": {"psychosis": psychosis, "control": controls},
        "reference_stats": stats, "global_instruction": global_instruction,
        "usd_spent": round(spend, 4) if spend is not None else None,
        "n_runs": len(rows), "predictions": rows,
    }
    out = K.LADDER_DIR / "predictions.json"
    out.write_text(json.dumps(payload, indent=2))
    ok = sum(1 for r in rows if not r.get("error"))
    print(f"[ladder] done: {ok}/{len(rows)} ok | spend "
          f"{'$'+format(spend,'.4f') if spend is not None else 'unknown'} -> {out}")
    return 0


def _dictionary_map() -> dict[str, dict[str, str]]:
    path = C.RESULTS_ROOT / "non_eeg_feature_dictionary.csv"
    if not path.exists():
        return {}
    d = pd.read_csv(path)
    out = {}
    for _, r in d.iterrows():
        out[r["column_name"]] = {"label": str(r.get("column_name", "")).split("__")[-1].replace("_", " ").title(),
                                 "description": str(r.get("description", "") or ""),
                                 "source_table": str(r.get("source_table", "") or "")}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
