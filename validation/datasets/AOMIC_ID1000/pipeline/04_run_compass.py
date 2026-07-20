#!/usr/bin/env python3
"""Step 04: run all participant and tier jobs with bounded process isolation.

Every (tier, participant) job runs in a separate worker process context so the
engine's settings singleton cannot leak paths between concurrent jobs. Workers
never write the shared predictions file. The parent process atomically checkpoints
each tier after every completion, which makes the run resumable without retaining
thousands of verbose engine artifacts.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import _bootstrap  # noqa: F401

import config
from src.full_stack.backend.config.settings import LLMBackend, get_settings
from src.full_stack.backend.data.models.prediction_task import build_task_spec_from_flat_args
from main import run_compass_pipeline


def _configure_engine(model: str, task_work_dir: Path) -> None:
    settings = get_settings()
    settings.models.backend = LLMBackend.OPENROUTER
    settings.models.public_model_name = model
    for role in ("orchestrator", "critic", "predictor", "integrator", "communicator", "tool"):
        setattr(settings.models, f"{role}_model", model)
    settings.paths.output_dir = task_work_dir / "outputs"
    settings.paths.logs_dir = task_work_dir / "logs"
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)


def _extract_prediction(result: dict):
    pred = (result.get("internal_context") or {}).get("prediction")
    root = getattr(pred, "root_prediction", None)
    reg = getattr(root, "regression", None) if root is not None else None
    values = getattr(reg, "values", None) if reg is not None else None
    if isinstance(values, dict) and values:
        col = config.TARGET["column"]
        return float(values.get(col, next(iter(values.values()))))
    return None


def _run_job(job: dict) -> dict:
    """Run one isolated job and return a compact, serializable result row."""
    tier_id = job["tier"]
    evaluation_id = job["participant_id"]
    task_work_dir = Path(job["results_dir"]) / tier_id / "_compass_work" / evaluation_id
    _configure_engine(job["model"], task_work_dir)
    task_spec = build_task_spec_from_flat_args(
        prediction_type="regression_univariate",
        target_label=config.TARGET["label"],
        regression_outputs=[config.TARGET["column"]],
    )
    errors = []
    started = time.time()
    output_buffer = io.StringIO()

    try:
        for attempt in range(1, job["retries"] + 2):
            try:
                with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                    result = run_compass_pipeline(
                        participant_dir=Path(job["inputs_dir"]) / tier_id / evaluation_id,
                        target_condition=config.TARGET["label"],
                        control_condition="",
                        prediction_task_spec=task_spec,
                        agent_instructions={"global": job["target_scale_note"]},
                        max_iterations=job["max_iterations"],
                        verbose=False,
                        interactive_ui=False,
                    )
                predicted = _extract_prediction(result)
                if predicted is None:
                    raise ValueError(f"Missing regression output '{config.TARGET['column']}'")
                return {
                    "participant_id": evaluation_id,
                    "ground_truth": job["ground_truth"],
                    "predicted": round(predicted, 4),
                    "abs_error": round(abs(predicted - job["ground_truth"]), 4),
                    "verdict": result.get("verdict"),
                    "attempts": attempt,
                    "duration_seconds": round(time.time() - started, 1),
                }
            except Exception as exc:  # noqa: BLE001
                error_text = f"attempt {attempt}: {type(exc).__name__}: {exc}"
                errors.append(error_text)
                if any(marker in error_text.lower() for marker in (
                    "key limit exceeded", "insufficient credits"
                )):
                    break
                if attempt <= job["retries"]:
                    time.sleep(job["retry_base_seconds"] * (2 ** (attempt - 1)))
        return {
            "participant_id": evaluation_id,
            "ground_truth": job["ground_truth"],
            "predicted": None,
            "abs_error": None,
            "attempts": job["retries"] + 1,
            "duration_seconds": round(time.time() - started, 1),
            "error": errors[-1] if errors else "unknown error",
            "error_history": errors,
            "log_tail": output_buffer.getvalue()[-2000:],
            "provider_budget_exhausted": any(
                any(marker in error.lower() for marker in (
                    "key limit exceeded", "insufficient credits"
                ))
                for error in errors
            ),
        }
    finally:
        if not job["keep_run_artifacts"]:
            shutil.rmtree(task_work_dir, ignore_errors=True)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def _prediction_payload(tier_id: str, rows: list, args, requested_n: int) -> dict:
    return {
        "dataset": config.DATASET_LABEL,
        "tier": tier_id,
        "model": args.model,
        "max_iterations": args.max_iter,
        "target": config.TARGET,
        "execution": {
            "requested_n": requested_n,
            "workers": args.workers,
            "retries": args.retries,
            "process_isolated": True,
            "parent_atomic_checkpoints": True,
            "model_facing_ids_blinded": True,
        },
        "predictions": rows,
    }


def _load_resumable_rows(path: Path, args, valid_ids: set[str]) -> dict[str, dict]:
    if not args.resume or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    if payload.get("model") != args.model or payload.get("max_iterations") != args.max_iter:
        return {}
    return {
        row["participant_id"]: row
        for row in payload.get("predictions", [])
        if row.get("participant_id") in valid_ids and row.get("predicted") is not None
    }


def _openrouter_key_status():
    """Read non-secret key budget metadata without making a model request."""
    key = get_settings().openrouter_api_key
    if not key:
        return None
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8")).get("data", {})
    except Exception:
        return None


def _openrouter_credit_status():
    """Read account credit totals without making a model request."""
    key = get_settings().openrouter_api_key
    if not key:
        return None
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8")).get("data", {})
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=None, help="run one tier id (default: all built tiers)")
    parser.add_argument("--model", default=config.ONTOLOGY_MODEL)
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITERATIONS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12,
                        help="bounded process workers across all tier-participant jobs")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--keep-run-artifacts", action="store_true")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    key_status = _openrouter_key_status()
    if key_status is not None and key_status.get("limit_remaining") is not None:
        remaining = float(key_status["limit_remaining"])
        if remaining <= 0:
            raise SystemExit(
                "[04] OpenRouter key spending limit is exhausted. Increase or reset "
                "the configured key limit, then rerun; successful checkpoints resume."
            )
        print(f"[04] OpenRouter key budget remaining: ${remaining:.2f}")
    credit_status = _openrouter_credit_status()
    if credit_status is not None:
        credit_balance = float(credit_status.get("total_credits", 0)) - float(
            credit_status.get("total_usage", 0)
        )
        if credit_balance <= 0:
            raise SystemExit(
                "[04] OpenRouter account credits are exhausted. Add credits, then "
                "rerun; successful checkpoints resume."
            )
        print(f"[04] OpenRouter account credit balance: ${credit_balance:.2f}")

    subset_payload = json.loads((config.RESULTS_DIR / "subset.json").read_text())
    subset = subset_payload["participants"]
    if args.limit:
        subset = subset[:args.limit]
    target_scale_note = subset_payload["agent_target_scale_note"]
    built = json.loads((config.INPUTS_DIR / "tiers.json").read_text())["tiers"]
    built_ids = [tier["id"] for tier in built]
    tier_ids = [args.tier] if args.tier else built_ids
    unknown = sorted(set(tier_ids) - set(built_ids))
    if unknown:
        raise SystemExit(f"Unknown or unbuilt tiers: {unknown}")

    order = {rec["participant_id"]: index for index, rec in enumerate(subset)}
    valid_ids = set(order)
    rows_by_tier: dict[str, dict[str, dict]] = {}
    jobs = []
    for tier_id in tier_ids:
        tier_root = config.RESULTS_DIR / tier_id
        legacy = tier_root / "compass_runs"
        if legacy.exists():
            shutil.rmtree(legacy)
        pred_path = tier_root / "predictions.json"
        rows_by_tier[tier_id] = _load_resumable_rows(pred_path, args, valid_ids)
        for rec in subset:
            if rec["participant_id"] in rows_by_tier[tier_id]:
                continue
            jobs.append({
                "tier": tier_id,
                "participant_id": rec["participant_id"],
                "ground_truth": rec["ground_truth"],
                "inputs_dir": str(config.INPUTS_DIR),
                "results_dir": str(config.RESULTS_DIR),
                "model": args.model,
                "max_iterations": args.max_iter,
                "target_scale_note": target_scale_note,
                "retries": max(0, args.retries),
                "retry_base_seconds": max(0.0, args.retry_base_seconds),
                "keep_run_artifacts": args.keep_run_artifacts,
            })

    def checkpoint(tier_id: str) -> None:
        rows = sorted(rows_by_tier[tier_id].values(), key=lambda row: order[row["participant_id"]])
        _atomic_json(
            config.RESULTS_DIR / tier_id / "predictions.json",
            _prediction_payload(tier_id, rows, args, requested_n=len(subset)),
        )

    for tier_id in tier_ids:
        checkpoint(tier_id)

    total = len(jobs)
    resumed = len(tier_ids) * len(subset) - total
    print(f"[04] Scheduled {total} jobs across {len(tier_ids)} tiers and "
          f"{len(subset)} participants; resumed={resumed}; workers={args.workers}")
    quota_exhausted = False
    if jobs:
        pool = ProcessPoolExecutor(max_workers=max(1, args.workers))
        job_iter = iter(jobs)
        pending = {}

        def submit_one():
            try:
                next_job = next(job_iter)
            except StopIteration:
                return False
            pending[pool.submit(_run_job, next_job)] = next_job
            return True

        for _ in range(min(max(1, args.workers), len(jobs))):
            submit_one()
        completed = 0
        try:
            while pending and not quota_exhausted:
                done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    completed += 1
                    try:
                        row = future.result()
                    except Exception as exc:  # noqa: BLE001
                        row = {
                            "participant_id": job["participant_id"],
                            "ground_truth": job["ground_truth"],
                            "predicted": None,
                            "abs_error": None,
                            "attempts": 0,
                            "error": f"worker failure: {type(exc).__name__}: {exc}",
                        }
                    tier_id = job["tier"]
                    rows_by_tier[tier_id][row["participant_id"]] = row
                    checkpoint(tier_id)
                    status = f"pred={row['predicted']}" if row.get("predicted") is not None else "FAILED"
                    print(f"[04] {completed}/{total} {tier_id} {row['participant_id']} {status} "
                          f"attempts={row.get('attempts')}", flush=True)
                    if row.get("provider_budget_exhausted"):
                        quota_exhausted = True
                        break
                    submit_one()
        except KeyboardInterrupt:
            print("[04] Interrupted; atomically saved completed predictions.")
            raise
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)

        if quota_exhausted:
            raise SystemExit(
                "[04] OpenRouter key limit reached during the run. Successful "
                "checkpoints are preserved; increase/reset the limit and rerun."
            )

    valid = 0
    failed = 0
    for tier_id in tier_ids:
        checkpoint(tier_id)
        tier_rows = rows_by_tier[tier_id].values()
        tier_valid = sum(row.get("predicted") is not None for row in tier_rows)
        tier_failed = len(subset) - tier_valid
        valid += tier_valid
        failed += tier_failed
        print(f"[04] {tier_id}: valid={tier_valid}/{len(subset)}, failed={tier_failed}")
    print(f"[04] Completed: valid={valid}, failed={failed}")


if __name__ == "__main__":
    main()
