#!/usr/bin/env python3
"""
Step 06 - hierarchical IST demo on a small, diverse cohort across all tiers.

This runs the engine in its **hierarchical mixed-task** mode: the root predicts the
IST total intelligence (univariate regression) and a child node predicts the three
IST subscales (fluid, memory, crystallised) as a multivariate regression under it.
The three subscales are prediction OUTPUTS only; they are never predictors (see
``config.EXCLUDED_COLUMNS``), so there is no target leakage.

It reuses the already-built per-tier ``compass_inputs`` (only the task spec and the
result extraction change, not the participant inputs), injects a rich global
instruction so the model knows exactly what the IST is and how every score is
scaled, and guards the OpenRouter spend so a demo stays cheap.

  python 06_run_hierarchical_demo.py --n 10 --max-usd 5
  python 06_run_hierarchical_demo.py --n 1 --tier T6_connectome   # cost/structure probe
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import config
from src.full_stack.backend.config.settings import LLMBackend, get_settings
from src.full_stack.backend.data.models.prediction_task import (
    PredictionMode,
    PredictionTaskNode,
    PredictionTaskSpec,
)
from main import run_compass_pipeline

OUT_DIR = config.RESULTS_DIR / "hierarchical_10subject"
ALL_OUTPUTS = [config.TARGET["column"]] + [s["column"] for s in config.SUBSCALES]


# --------------------------------------------------------------------------- #
# Task spec, context, cohort
# --------------------------------------------------------------------------- #

def build_task_spec() -> PredictionTaskSpec:
    """Root = total intelligence (univariate); child = 3 subscales (multivariate)."""
    return PredictionTaskSpec(
        task_id="ist_hierarchical",
        root=PredictionTaskNode(
            node_id="total_intelligence",
            display_name=config.TARGET["label"],
            mode=PredictionMode.UNIVARIATE_REGRESSION,
            regression_outputs=[config.TARGET["column"]],
            unit_by_output={config.TARGET["column"]: config.TARGET.get("units", "points")},
            children=[
                PredictionTaskNode(
                    node_id="ist_subscales",
                    display_name="IST subscales: fluid, memory, crystallised",
                    mode=PredictionMode.MULTIVARIATE_REGRESSION,
                    regression_outputs=[s["column"] for s in config.SUBSCALES],
                    unit_by_output={s["column"]: "IST points" for s in config.SUBSCALES},
                )
            ],
        ),
    )


def reference_stats(df: pd.DataFrame, evaluation_source_ids: set) -> dict:
    """Native mean/sd for the total and each subscale over the disjoint reference split."""
    ref = df[~df["participant_id"].astype(str).isin(evaluation_source_ids)]
    stats = {}
    for col in ALL_OUTPUTS:
        s = pd.to_numeric(ref[col], errors="coerce").dropna()
        stats[col] = {"mean": round(float(s.mean()), 2), "sd": round(float(s.std(ddof=0)), 2), "n": int(s.size)}
    return stats


def build_global_instruction(stats: dict) -> str:
    """Rich, interpretable context so the LLM knows the scales and their meaning."""
    lines = [config.IST_CONTEXT, "", "Predict the following related scores on their NATIVE IST scales:"]
    tgt = config.TARGET
    st = stats[tgt["column"]]
    lines.append(f"- {tgt['column']} ({tgt['label']}): the overall composite. "
                 f"Reference cohort mean={st['mean']}, sd={st['sd']} native IST points.")
    for sub in config.SUBSCALES:
        st = stats[sub["column"]]
        lines.append(f"- {sub['column']} ({sub['label']}): {sub['description']} "
                     f"Reference cohort mean={st['mean']}, sd={st['sd']} native IST points.")
    lines += [
        "",
        "The three subscales are components of the total and are typically positively "
        "correlated with it. No IST total or subscale value for this participant is "
        "provided as input; infer every score only from the non-cognitive multimodal "
        "evidence in this record. Return one numeric value per output on its native scale.",
    ]
    return "\n".join(lines)


def select_diverse(records: list, n: int) -> list:
    """Pick n participants spread evenly across the total-intelligence range."""
    ordered = sorted(records, key=lambda r: r["ground_truth"])
    if n >= len(ordered):
        return ordered
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    seen, picked = set(), []
    for i in idx:
        while int(i) in seen and int(i) < len(ordered) - 1:
            i += 1
        seen.add(int(i))
        picked.append(ordered[int(i)])
    return picked


# --------------------------------------------------------------------------- #
# Budget + execution
# --------------------------------------------------------------------------- #

def _credit_usage() -> float | None:
    """Total OpenRouter usage in USD so far (None if unavailable)."""
    key = get_settings().openrouter_api_key
    if not key:
        return None
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode()).get("data", {})
            return float(data.get("total_usage", 0.0))
    except Exception:
        return None


def _configure_engine(model: str, work_dir: Path) -> None:
    s = get_settings()
    s.models.backend = LLMBackend.OPENROUTER
    s.models.public_model_name = model
    for role in ("orchestrator", "critic", "predictor", "integrator", "communicator", "tool"):
        setattr(s.models, f"{role}_model", model)
    s.paths.output_dir = work_dir / "outputs"
    s.paths.logs_dir = work_dir / "logs"
    s.paths.output_dir.mkdir(parents=True, exist_ok=True)
    s.paths.logs_dir.mkdir(parents=True, exist_ok=True)


def _extract_all(result: dict) -> dict:
    """Collect every node's regression outputs from the hierarchical prediction."""
    pred = (result.get("internal_context") or {}).get("prediction")
    root = getattr(pred, "root_prediction", None)
    out = {}
    if root is not None:
        for node in root.walk():
            reg = getattr(node, "regression", None)
            for k, v in (getattr(reg, "values", None) or {}).items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
    return out


def _run_job(job: dict) -> dict:
    tier_id, eid = job["tier"], job["participant_id"]
    work = OUT_DIR / "_work" / tier_id / eid
    _configure_engine(job["model"], work)
    spec = build_task_spec()
    started = time.time()
    buf = io.StringIO()
    try:
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                result = run_compass_pipeline(
                    participant_dir=Path(job["inputs_dir"]) / tier_id / eid,
                    target_condition=config.TARGET["label"],
                    control_condition="",
                    prediction_task_spec=spec,
                    agent_instructions={"global": job["global_instruction"]},
                    max_iterations=job["max_iterations"],
                    verbose=False,
                    interactive_ui=False,
                )
            preds = _extract_all(result)
            row = {"participant_id": eid, "tier": tier_id,
                   "predicted": {k: round(preds.get(k), 3) if preds.get(k) is not None else None for k in ALL_OUTPUTS},
                   "ground_truth": job["ground_truth"], "verdict": result.get("verdict"),
                   "duration_seconds": round(time.time() - started, 1)}
            row["missing_outputs"] = [k for k in ALL_OUTPUTS if row["predicted"].get(k) is None]
            return row
        except Exception as exc:  # noqa: BLE001
            return {"participant_id": eid, "tier": tier_id, "predicted": None,
                    "ground_truth": job["ground_truth"],
                    "error": f"{type(exc).__name__}: {exc}", "log_tail": buf.getvalue()[-1500:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of diverse subjects")
    ap.add_argument("--tier", default=None, help="single tier id (default: all built tiers)")
    ap.add_argument("--model", default=config.ONTOLOGY_MODEL)
    ap.add_argument("--max-iter", type=int, default=config.MAX_ITERATIONS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-usd", type=float, default=5.0, help="stop scheduling if usage grows past this")
    args = ap.parse_args()

    subset = json.loads((config.RESULTS_DIR / "subset.json").read_text())
    records = subset["participants"]
    picked = select_diverse(records, args.n)
    picked_ids = {r["participant_id"] for r in picked}

    df = config.load_merged_frame()
    eval_source_ids = {str(r["source_participant_id"]) for r in records}
    stats = reference_stats(df, eval_source_ids)
    global_instruction = build_global_instruction(stats)

    # Subscale ground truth (evaluation-only bookkeeping; never seen by the engine).
    by_source = df.set_index(df["participant_id"].astype(str))
    for r in picked:
        src = str(r["source_participant_id"])
        r["ground_truth_all"] = {config.TARGET["column"]: r["ground_truth"]}
        for sub in config.SUBSCALES:
            val = pd.to_numeric(pd.Series([by_source.loc[src, sub["column"]]]), errors="coerce").iloc[0]
            r["ground_truth_all"][sub["column"]] = None if pd.isna(val) else round(float(val), 2)

    built = json.loads((config.INPUTS_DIR / "tiers.json").read_text())["tiers"]
    tier_ids = [args.tier] if args.tier else [t["id"] for t in built]

    jobs = [{
        "tier": t, "participant_id": r["participant_id"], "ground_truth": r["ground_truth_all"],
        "inputs_dir": str(config.INPUTS_DIR), "model": args.model, "max_iterations": args.max_iter,
        "global_instruction": global_instruction,
    } for t in tier_ids for r in picked]

    usage0 = _credit_usage()
    print(f"[06] Hierarchical IST demo: {len(picked)} diverse subjects x {len(tier_ids)} tiers "
          f"= {len(jobs)} runs | model={args.model}")
    print(f"[06] Cohort total-IST spread: {sorted(r['ground_truth'] for r in picked)}")
    if usage0 is not None:
        print(f"[06] OpenRouter usage so far: ${usage0:.4f}; will stop scheduling past +${args.max_usd:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list = []
    stop = False
    pool = ProcessPoolExecutor(max_workers=max(1, args.workers))
    job_iter = iter(jobs)
    pending = {}

    def submit_one() -> bool:
        try:
            nxt = next(job_iter)
        except StopIteration:
            return False
        pending[pool.submit(_run_job, nxt)] = nxt
        return True

    for _ in range(min(args.workers, len(jobs))):
        submit_one()
    done_count = 0
    try:
        while pending and not stop:
            done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                pending.pop(fut)
                done_count += 1
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    row = {"error": f"worker: {type(exc).__name__}: {exc}"}
                rows.append(row)
                tag = "ERR" if row.get("error") or row.get("predicted") is None else \
                    "pred=" + ",".join(f"{k.split('_')[-1][:4]}={row['predicted'].get(k)}" for k in ALL_OUTPUTS)
                print(f"[06] {done_count}/{len(jobs)} {row.get('tier')} {row.get('participant_id')} {tag}", flush=True)
                usage = _credit_usage()
                if usage is not None and usage0 is not None and (usage - usage0) >= args.max_usd:
                    print(f"[06] Spend guard hit: +${usage - usage0:.2f} >= ${args.max_usd:.2f}. Stopping.")
                    stop = True
                    break
                submit_one()
    finally:
        for fut in pending:
            fut.cancel()
        pool.shutdown(wait=True, cancel_futures=True)

    usage1 = _credit_usage()
    spend = (usage1 - usage0) if (usage0 is not None and usage1 is not None) else None
    payload = {
        "dataset": config.DATASET_LABEL,
        "task": "hierarchical_mixed: total intelligence (univariate) -> 3 IST subscales (multivariate)",
        "model": args.model,
        "outputs": ALL_OUTPUTS,
        "reference_stats": stats,
        "ist_context": config.IST_CONTEXT,
        "global_instruction": global_instruction,
        "cohort": [{"participant_id": r["participant_id"], "ground_truth_all": r["ground_truth_all"]} for r in picked],
        "n_runs": len(rows),
        "usd_spent": round(spend, 4) if spend is not None else None,
        "predictions": rows,
    }
    (OUT_DIR / "predictions.json").write_text(json.dumps(payload, indent=2))
    ok = sum(1 for r in rows if r.get("predicted") and not r.get("missing_outputs"))
    print(f"[06] Done: {ok}/{len(rows)} complete runs. "
          f"Spend this run: {'$'+format(spend, '.4f') if spend is not None else 'unknown'}. "
          f"-> {OUT_DIR/'predictions.json'}")


if __name__ == "__main__":
    main()
