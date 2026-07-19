#!/usr/bin/env python3
"""
Step 04 - run the COMPASS engine on the subset for one or more tiers.

Drives the real engine pipeline (`run_compass_pipeline`) per participant via
OpenRouter with a small model. Tool steps within each plan run in parallel; tiers
and participants run sequentially to keep the shared settings singleton clean.

Outputs per tier:
  results/<tier_id>/compass_runs/participant_<id>/...   full engine outputs
  results/<tier_id>/predictions.json                    predicted vs ground truth
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401

import config
from src.full_stack.backend.config.settings import get_settings, LLMBackend
from src.full_stack.backend.data.models.prediction_task import build_task_spec_from_flat_args
from main import run_compass_pipeline


def _configure_engine(model: str, out_dir) -> None:
    settings = get_settings()
    settings.models.backend = LLMBackend.OPENROUTER
    settings.models.public_model_name = model
    for role in ("orchestrator", "critic", "predictor", "integrator", "communicator", "tool"):
        setattr(settings.models, f"{role}_model", model)
    settings.paths.output_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)


def _extract_prediction(result: dict):
    pred = (result.get("internal_context") or {}).get("prediction")
    root = getattr(pred, "root_prediction", None)
    reg = getattr(root, "regression", None) if root is not None else None
    values = getattr(reg, "values", None) if reg is not None else None
    if isinstance(values, dict) and values:
        col = config.TARGET["column"]
        return float(values.get(col, next(iter(values.values()))))
    return None


def _run_tier(tier_id: str, model: str, max_iter: int, subset, task_spec):
    inputs_dir = config.INPUTS_DIR / tier_id
    out_root = config.RESULTS_DIR / tier_id
    _configure_engine(model, out_root / "compass_runs")
    rows = []
    print(f"\n[04] === Tier {tier_id}: {len(subset)} participants ===")
    for i, rec in enumerate(subset, 1):
        pid = rec["participant_id"]
        t0 = time.time()
        try:
            result = run_compass_pipeline(
                participant_dir=inputs_dir / pid,
                target_condition=config.TARGET["label"], control_condition="",
                prediction_task_spec=task_spec,
                agent_instructions={"global": config.TARGET_SCALE_NOTE},
                max_iterations=max_iter, verbose=False, interactive_ui=False,
            )
            pv = _extract_prediction(result)
            rows.append({"participant_id": pid, "ground_truth": rec["ground_truth"],
                         "predicted": round(pv, 2) if pv is not None else None,
                         "abs_error": round(abs(pv - rec["ground_truth"]), 2) if pv is not None else None,
                         "verdict": result.get("verdict"), "duration_seconds": round(time.time() - t0, 1)})
            print(f"[04] {tier_id} {pid}: pred={pv} gt={rec['ground_truth']} "
                  f"({rows[-1]['duration_seconds']}s)")
        except Exception as exc:  # noqa: BLE001
            rows.append({"participant_id": pid, "ground_truth": rec["ground_truth"],
                         "predicted": None, "abs_error": None,
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"[04] {tier_id} {pid}: FAILED {type(exc).__name__}: {exc}")
        with open(out_root / "predictions.json", "w") as f:
            json.dump({"dataset": config.DATASET_LABEL, "tier": tier_id, "model": model,
                       "max_iterations": max_iter, "target": config.TARGET, "predictions": rows}, f, indent=2)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=None, help="run one tier id (default: all built tiers)")
    ap.add_argument("--model", default=config.ONTOLOGY_MODEL)
    ap.add_argument("--max-iter", type=int, default=config.MAX_ITERATIONS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(config.RESULTS_DIR / "subset.json") as f:
        subset = json.load(f)["participants"]
    if args.limit:
        subset = subset[: args.limit]
    with open(config.INPUTS_DIR / "tiers.json") as f:
        built = [t["id"] for t in json.load(f)["tiers"]]
    tier_ids = [args.tier] if args.tier else built

    task_spec = build_task_spec_from_flat_args(
        prediction_type="regression_univariate", target_label=config.TARGET["label"],
        regression_outputs=[config.TARGET["column"]],
    )
    for tid in tier_ids:
        _run_tier(tid, args.model, args.max_iter, subset, task_spec)
    print(f"\n[04] Completed {len(tier_ids)} tier(s).")


if __name__ == "__main__":
    main()
