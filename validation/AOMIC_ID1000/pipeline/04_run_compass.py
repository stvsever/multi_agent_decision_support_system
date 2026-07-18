#!/usr/bin/env python3
"""
Step 04 - run the COMPASS engine on the subset (univariate IQ regression).

Drives the real engine pipeline in-process via ``run_compass_pipeline`` so we use
the exact production actor-critic flow. All roles use a small OpenRouter model
(google/gemini-3.1-flash-lite). Tool steps inside each plan run in parallel
(ThreadPoolExecutor) exactly as in production; participants are run sequentially
to keep the shared settings singleton and logs clean.

Outputs:
  results/compass_runs/participant_<id>/...   full per-participant engine outputs
  results/predictions.json                    predicted vs ground-truth summary
"""

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import config
from src.full_stack.backend.config.settings import get_settings, LLMBackend
from src.full_stack.backend.data.models.prediction_task import build_task_spec_from_flat_args
from main import run_compass_pipeline


def _configure_engine(model: str) -> None:
    settings = get_settings()
    settings.models.backend = LLMBackend.OPENROUTER
    settings.models.public_model_name = model
    for role in ("orchestrator", "critic", "predictor", "integrator", "communicator", "tool"):
        setattr(settings.models, f"{role}_model", model)
    # Route all engine outputs into the validation results folder.
    settings.paths.output_dir = config.RESULTS_DIR / "compass_runs"
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)


def _extract_prediction(result: dict) -> float | None:
    pred = (result.get("internal_context") or {}).get("prediction")
    root = getattr(pred, "root_prediction", None)
    reg = getattr(root, "regression", None) if root is not None else None
    values = getattr(reg, "values", None) if reg is not None else None
    if isinstance(values, dict) and values:
        col = config.TARGET["column"]
        if col in values:
            return float(values[col])
        return float(next(iter(values.values())))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.ONTOLOGY_MODEL)
    ap.add_argument("--max-iter", type=int, default=config.MAX_ITERATIONS)
    ap.add_argument("--limit", type=int, default=None, help="cap number of participants")
    args = ap.parse_args()

    _configure_engine(args.model)

    with open(config.RESULTS_DIR / "subset.json") as f:
        subset = json.load(f)
    participants = subset["participants"]
    if args.limit:
        participants = participants[: args.limit]

    task_spec = build_task_spec_from_flat_args(
        prediction_type="regression_univariate",
        target_label=config.TARGET["label"],
        regression_outputs=[config.TARGET["column"]],
    )

    rows = []
    print(f"[04] Running COMPASS on {len(participants)} participants "
          f"(model={args.model}, max_iter={args.max_iter})")
    for i, rec in enumerate(participants, 1):
        pid = rec["participant_id"]
        pdir = config.INPUTS_DIR / pid
        print(f"\n[04] ({i}/{len(participants)}) {pid}  ground_truth={rec['ground_truth']}")
        t0 = time.time()
        try:
            result = run_compass_pipeline(
                participant_dir=pdir,
                target_condition=config.TARGET["label"],
                control_condition="",
                prediction_task_spec=task_spec,
                agent_instructions={"global": config.TARGET_SCALE_NOTE},
                max_iterations=args.max_iter,
                verbose=False,
                interactive_ui=False,
            )
            predicted = _extract_prediction(result)
            rows.append({
                "participant_id": pid,
                "ground_truth": rec["ground_truth"],
                "predicted": round(predicted, 2) if predicted is not None else None,
                "abs_error": round(abs(predicted - rec["ground_truth"]), 2) if predicted is not None else None,
                "verdict": result.get("verdict"),
                "iterations": result.get("iterations"),
                "duration_seconds": round(time.time() - t0, 1),
                "output_dir": result.get("output_dir"),
                "error": None,
            })
            print(f"[04] {pid}: predicted={predicted}  gt={rec['ground_truth']}  "
                  f"verdict={result.get('verdict')}  {rows[-1]['duration_seconds']}s")
        except Exception as exc:  # noqa: BLE001 - record and continue over the subset
            rows.append({
                "participant_id": pid, "ground_truth": rec["ground_truth"],
                "predicted": None, "abs_error": None, "error": f"{type(exc).__name__}: {exc}",
                "duration_seconds": round(time.time() - t0, 1),
            })
            print(f"[04] {pid}: FAILED - {type(exc).__name__}: {exc}")

        # Persist incrementally so partial progress survives interruptions.
        with open(config.RESULTS_DIR / "predictions.json", "w") as f:
            json.dump({
                "dataset": config.DATASET_LABEL,
                "model": args.model,
                "max_iterations": args.max_iter,
                "reference_mode": subset.get("reference_mode"),
                "target": config.TARGET,
                "predictions": rows,
            }, f, indent=2)

    done = [r for r in rows if r.get("predicted") is not None]
    print(f"\n[04] Completed {len(done)}/{len(rows)} with a numeric prediction. "
          f"Wrote results/predictions.json")


if __name__ == "__main__":
    main()
