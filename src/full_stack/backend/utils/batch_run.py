"""COMPASS batch runner: run main.py for a fixed participant list, one after the other.

Every main.py option that matters for a cohort has a batch option of the same
name (Predictor and companion models, routing, decision-model settings, context
windows, token budgets, local backend); an unset option leaves main.py's
default. Any other main.py flag given here is passed through unchanged, for
example ``--iterations 2`` or ``--global_instruction "..."``.

    python src/full_stack/backend/utils/batch_run.py --backend openrouter \\
        --public_model deepseek/deepseek-v4-flash-0731 --predictor_model typesafe/jev-1.13 --check_config
    python src/full_stack/backend/utils/batch_run.py --backend openrouter \\
        --public_model deepseek/deepseek-v4-flash-0731 --predictor_model typesafe/jev-1.13 --preflight_check

``--check_config`` checks every participant offline (models, context windows,
Predictor input budget, API keys, and the route each record takes) and runs
nothing; ``--preflight_check`` does the same first and starts the cohort only
when no check reports a problem.
"""

import argparse
import json
import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Config
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# DATA_ROOT: configurable via environment variable for HPC vs local
# On HPC: set DATA_ROOT env var in the Slurm script (points to HPC_data)
# Locally: defaults to the bundled pseudo-data inputs.
DATA_ROOT = Path(os.getenv(
    "DATA_ROOT",
    str(BACKEND_ROOT / "data" / "pseudo_data" / "inputs")
))

MAIN_SCRIPT = PROJECT_ROOT / "main.py"
VALIDATION_DIR = BACKEND_ROOT / "utils" / "validation" / "with_annotated_dataset"
VALIDATION_METRICS_SCRIPT = VALIDATION_DIR / "run_validation_metrics.py"
VALIDATION_DETAILED_SCRIPT = VALIDATION_DIR / "detailed_analysis.py"
VALIDATION_EVALUATE_SCRIPT = VALIDATION_DIR / "run_evaluation.py"
VALIDATION_TEMPLATE_EXAMPLES_DIR = VALIDATION_DIR / "annotation_templates" / "examples"
RESULTS_DIR = PROJECT_ROOT / "results"

VALIDATION_TEMPLATE_BY_MODE = {
    "binary": "binary_targets_example.json",
    "multiclass": "multiclass_annotations_example.json",
    "regression_univariate": "regression_univariate_annotations_example.json",
    "regression_multivariate": "regression_multivariate_annotations_example.json",
    "hierarchical": "hierarchical_annotations_example.json",
}

# ─── Participant Cohort ───────────────────────────────────────────────────
# Example participant cohort (anonymized placeholders).
# Replace these IDs with your real participant IDs / EIDs for your dataset.
#
# Format:
#   id:         participant identifier (folder: participant_ID{id})
#   expected:   ground-truth label (CASE or CONTROL)
#   target_str: phenotype label passed to main.py --target_label
PARTICIPANTS = [
    {"id": "01", "expected": "CASE",    "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
    {"id": "02", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "03", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "04", "expected": "CASE",    "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
    {"id": "05", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
]

# ─── main.py flags forwarded by this runner ───────────────────────────────
# Each batch option has the name of the main.py flag it feeds and is forwarded
# only when set, so main.py keeps its own default otherwise.
TASK_VALUE_FLAGS = ("control_label", "class_labels", "task_spec_file", "task_spec_json")
# Hosted backends (openrouter, openai): the model every LLM role runs on.
HOSTED_VALUE_FLAGS = ("public_model", "public_max_context_tokens")
# --backend local only; main.py ignores them on a hosted backend.
LOCAL_VALUE_FLAGS = (
    "model",
    "max_tokens",
    "local_engine",
    "local_dtype",
    "local_quant",
    "local_kv_cache_dtype",
    "local_tensor_parallel",
    "local_pipeline_parallel",
    "local_gpu_mem_util",
    "local_max_model_len",
    "local_attn",
)
LOCAL_SWITCH_FLAGS = ("local_enforce_eager", "local_trust_remote_code")
# Predictor choice, routing, decision-model settings, context windows and budgets (every backend).
RUN_VALUE_FLAGS = (
    "predictor_model",
    "companion_model",
    "orchestration",
    "orchestration_threshold",
    "decision_provider",
    "decision_choice_orders",
    "decision_score_levels",
    "decision_compiler_model",
    "decision_stability_threshold",
    "decision_regression_stability_threshold",
    "decision_sufficiency_threshold",
    "context_window",
    "reasoning_effort",
    "max_agent_input",
    "max_agent_output",
    "max_tool_input",
    "max_tool_output",
)
# BooleanOptionalAction flags: --flag or --no-flag.
RUN_BOOLEAN_FLAGS = ("decision_refine",)
# Repeatable flags (action="append").
RUN_REPEATED_FLAGS = ("role_context_window",)

# main.py exits with 2 when argparse or its own validation refuses the command
# line; every participant would then fail the same way.
USAGE_ERROR_EXIT_CODE = 2


def _is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def build_main_command(
    participant_path: Any,
    target_str: str,
    batch_args: Mapping[str, Any],
    *,
    check_config: bool = False,
) -> List[str]:
    """The main.py command line for one participant (pure: no process, no file access)."""
    prediction_type = str(batch_args.get("prediction_type") or "binary")
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        str(participant_path),
        "--prediction_type", prediction_type,
        "--target_label", target_str,
    ]
    if check_config:
        # Offline check only: JSON on stdout, notes and loader output on stderr.
        cmd.extend(["--check_config", "--check_format", "json"])
    else:
        cmd.extend(["--detailed_log", "--quiet"])

    for name in TASK_VALUE_FLAGS:
        if _is_set(batch_args.get(name)):
            cmd.extend([f"--{name}", str(batch_args[name])])
    if _is_set(batch_args.get("regression_output")):
        cmd.extend(["--regression_output", str(batch_args["regression_output"])])
    elif _is_set(batch_args.get("regression_outputs")):
        cmd.extend(["--regression_outputs", str(batch_args["regression_outputs"])])

    backend = str(batch_args.get("backend") or "").strip()
    if backend:
        cmd.extend(["--backend", backend])
    if backend == "local":
        for name in LOCAL_VALUE_FLAGS:
            if _is_set(batch_args.get(name)):
                cmd.extend([f"--{name}", str(batch_args[name])])
        for name in LOCAL_SWITCH_FLAGS:
            if batch_args.get(name):
                cmd.append(f"--{name}")
    else:
        for name in HOSTED_VALUE_FLAGS:
            if _is_set(batch_args.get(name)):
                cmd.extend([f"--{name}", str(batch_args[name])])

    for name in RUN_VALUE_FLAGS:
        if _is_set(batch_args.get(name)):
            cmd.extend([f"--{name}", str(batch_args[name])])
    for name in RUN_BOOLEAN_FLAGS:
        value = batch_args.get(name)
        if value is not None:
            cmd.append(f"--{name}" if value else f"--no-{name}")
    for name in RUN_REPEATED_FLAGS:
        values = batch_args.get(name) or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            if _is_set(value):
                cmd.extend([f"--{name}", str(value)])

    cmd.extend(str(arg) for arg in (batch_args.get("extra_main_args") or []))
    return cmd


def resolve_participant_path(pid: str) -> str:
    """Participant folder under DATA_ROOT: participant_ID<id>, else ID<id>, else <id>."""
    path = os.path.join(DATA_ROOT, f"participant_ID{pid}")
    if not os.path.exists(path):
        for alt in (os.path.join(DATA_ROOT, f"ID{pid}"), os.path.join(DATA_ROOT, pid)):
            if os.path.exists(alt):
                return alt
    return path


def _subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    return env


def run_participant(pid_info):
    pid = pid_info["id"]
    target_str = pid_info["target_str"]
    path = resolve_participant_path(pid)

    # Output file for this process (to avoid PIPE deadlock)
    results_dir = Path(RESULTS_DIR) / f"participant_{pid}"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file_path = results_dir / f"batch_out_{pid}.txt"
    out_file = open(out_file_path, "w")

    cmd = build_main_command(path, target_str, BATCH_ARGS)

    print(f"Launching {pid} ({pid_info['expected']})...")
    print(f"  > Path:   {path}")
    print(f"  > Target: {target_str[:80]}...")
    print(f"  > Cmd:    {' '.join(cmd[:6])}...")
    # Use Popen with file stdout AND stderr merged, and custom env
    proc = subprocess.Popen(cmd, stdout=out_file, stderr=subprocess.STDOUT, text=True, env=_subprocess_env())
    return proc, pid, out_file, out_file_path


def binary_outcome_from_report(report: Mapping[str, Any]) -> str:
    """CASE or CONTROL from a COMPASS report JSON (its root binary node), else UNKNOWN.

    main.py puts the binary labels in the task spec as [target, control], so the
    first class label is the CASE side whatever the target is called.
    """
    prediction = report.get("prediction") or {}
    root = prediction.get("root_prediction") or {}
    label = (root.get("classification") or {}).get("predicted_label") or prediction.get("classification")
    if not label:
        return "UNKNOWN"
    label = str(label).strip()
    if label.upper() in {"CASE", "CONTROL"}:
        return label.upper()
    spec_root = (prediction.get("prediction_task_spec") or {}).get("root") or {}
    labels = [str(x).strip().lower() for x in (spec_root.get("class_labels") or [])]
    if len(labels) == 2 and label.lower() in labels:
        return "CASE" if label.lower() == labels[0] else "CONTROL"
    if label.lower() == str(prediction.get("control_condition") or "").strip().lower():
        return "CONTROL"
    if label.lower() == str(prediction.get("target_condition") or "").strip().lower():
        return "CASE"
    return "UNKNOWN"


REPORT_SAVED_PREFIX = "[Report] Saved to:"


def read_binary_outcome(
    pid: str,
    participant_path: Any,
    since: Optional[float] = None,
    log_path: Optional[Path] = None,
) -> str:
    """The binary outcome of this batch's run for one participant (UNKNOWN when no fresh report).

    main.py writes ``<results>/participant_<participant id>/report_<id>.json``,
    where the id is the record's own or its folder name, and prints the path in
    the run's log. Reports older than ``since`` (an earlier run) are ignored.
    """
    reports: List[Path] = []
    if log_path is not None and Path(log_path).exists():
        for line in Path(log_path).read_text(errors="replace").splitlines():
            if REPORT_SAVED_PREFIX in line:
                saved = Path(line.split(REPORT_SAVED_PREFIX, 1)[1].strip())
                if saved.suffix == ".json" and saved.exists():
                    reports.append(saved)
    folders = {Path(RESULTS_DIR) / f"participant_{pid}", Path(RESULTS_DIR) / f"participant_{Path(str(participant_path)).name}"}
    for folder in folders:
        if folder.is_dir():
            reports.extend(folder.glob("report_*.json"))
    fresh = [r for r in reports if since is None or r.stat().st_mtime >= since]
    for path in sorted(fresh, key=lambda r: r.stat().st_mtime, reverse=True):
        try:
            outcome = binary_outcome_from_report(json.loads(path.read_text()))
        except (OSError, ValueError, AttributeError):
            continue
        if outcome != "UNKNOWN":
            return outcome
    # Markdown reports of older engine versions.
    legacy = Path(RESULTS_DIR) / f"participant_{pid}" / f"report_{pid}.md"
    if legacy.exists() and (since is None or legacy.stat().st_mtime >= since):
        content = legacy.read_text(errors="replace")
        if "**Classification**: CASE" in content:
            return "CASE"
        if "**Classification**: CONTROL" in content:
            return "CONTROL"
    return "UNKNOWN"


def _tail(text: str, lines: int = 12) -> str:
    return "\n".join(str(text or "").rstrip().splitlines()[-lines:])


def _error_lines(text: str, fallback_lines: int = 6) -> List[str]:
    """The "error:" lines of a failed main.py (argparse prints its whole usage first), else the tail."""
    errors = [line.strip() for line in str(text or "").splitlines() if "error:" in line.lower()]
    return errors or _tail(text, fallback_lines).splitlines()


def check_participant_config(pid_info: Mapping[str, Any]) -> Dict[str, Any]:
    """Run ``main.py --check_config`` for one participant (offline: no model call, no spend)."""
    pid = str(pid_info["id"])
    path = resolve_participant_path(pid)
    results_dir = Path(RESULTS_DIR) / f"participant_{pid}"
    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_main_command(path, str(pid_info["target_str"]), BATCH_ARGS, check_config=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_subprocess_env())
    report: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(proc.stdout)
        if isinstance(parsed, dict):
            report = parsed
    except ValueError:
        report = None
    if report is not None:
        output_path = results_dir / f"config_check_{pid}.json"
        output_path.write_text(json.dumps(report, indent=2, default=str))
    else:
        output_path = results_dir / f"config_check_{pid}.txt"
        output_path.write_text(f"$ {' '.join(cmd)}\n\n[stdout]\n{proc.stdout}\n[stderr]\n{proc.stderr}")
    return {
        "id": pid,
        "path": path,
        "exit_code": int(proc.returncode),
        "report": report,
        "stderr": proc.stderr,
        "stdout": proc.stdout,
        "output_path": str(output_path),
    }


def _describe_check(check: Mapping[str, Any]) -> List[str]:
    report = check.get("report")
    if report is None:
        lines = [f"  ✗ {check['id']}: check failed (exit code {check['exit_code']}), see {check['output_path']}"]
        output = f"{check.get('stdout') or ''}\n{check.get('stderr') or ''}"
        lines.extend(f"      {line}" for line in _error_lines(output))
        return lines
    problems = list(report.get("problems") or [])
    record = report.get("record") or {}
    if record:
        unit = "tokens" if str(record.get("unit", "")).startswith("tokens") else str(record.get("unit", "tokens"))
        detail = (
            f"{str(record.get('route', '?')).upper()}: {int(record.get('input_tokens') or 0):,} {unit} "
            f"against a budget of {int(record.get('budget_tokens') or 0):,}"
        )
    else:
        detail = "record not measured"
    marker = "✗" if problems or check["exit_code"] != 0 else "✓"
    lines = [f"  {marker} {check['id']}: {detail}"]
    lines.extend(f"      problem: {problem}" for problem in problems)
    lines.extend(f"      note: {note}" for note in (report.get("notes") or []))
    return lines


def run_config_checks(participants: Optional[Sequence[Mapping[str, Any]]] = None) -> int:
    """Check every participant offline; 0 when no check reports a problem, else 1."""
    participants = list(PARTICIPANTS if participants is None else participants)
    print()
    print("=" * 60)
    print(" CONFIGURATION CHECK (offline: no model call, no spend)")
    print("=" * 60)
    checks: List[Dict[str, Any]] = []
    for pid_info in participants:
        check = check_participant_config(pid_info)
        checks.append(check)
        for line in _describe_check(check):
            print(line)
        if check["exit_code"] == USAGE_ERROR_EXIT_CODE and check["report"] is None:
            print("  main.py refused the command line; the other participants would fail the same way.")
            break
    first = next((c["report"] for c in checks if c.get("report")), None)
    if first:
        roles = first.get("roles") or {}
        predictor = roles.get("predictor") or {}
        companion = roles.get("orchestrator") or {}
        print()
        print(f"  Backend:      {first.get('backend')}    endpoint: {first.get('endpoint')}")
        print(f"  Predictor:    {predictor.get('model')} ({first.get('predictor_kind')}), window {predictor.get('context_window')}")
        print(f"  Other roles:  {companion.get('model')}, window {companion.get('context_window')}")
        orchestration = first.get("orchestration") or {}
        routing_line = str(orchestration.get("mode"))
        if routing_line == "auto":
            routing_line += f" (threshold {orchestration.get('threshold_tokens') or 'the Predictor input budget'})"
        print(f"  Routing:      {routing_line}")
        keys = first.get("credentials") or {}
        if keys:
            print("  API keys:     " + ", ".join(f"{k} {'present' if v else 'MISSING'}" for k, v in keys.items()))
    routes: Dict[str, int] = {}
    for check in checks:
        route = ((check.get("report") or {}).get("record") or {}).get("route")
        if route:
            routes[str(route)] = routes.get(str(route), 0) + 1
    failed = [c for c in checks if c["exit_code"] != 0 or c.get("report") is None]
    print()
    if routes:
        print("  Routes:       " + ", ".join(f"{count} {route}" for route, count in sorted(routes.items())))
    print(f"  Checked:      {len(checks)} of {len(participants)} participant(s), {len(failed)} with problems")
    print("=" * 60)
    return 1 if failed or len(checks) < len(participants) else 0

BATCH_ARGS = {}


def _resolve_validation_results_dir(explicit_path: str = "") -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        return None

    candidates = [
        RESULTS_DIR / "participant_runs",
        RESULTS_DIR,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _resolve_metrics_script() -> Optional[Path]:
    if VALIDATION_METRICS_SCRIPT.exists():
        return VALIDATION_METRICS_SCRIPT
    return None


def _validation_template_hint(prediction_type: str) -> Path:
    name = VALIDATION_TEMPLATE_BY_MODE.get(str(prediction_type or "binary").strip().lower(), "binary_targets_example.json")
    return VALIDATION_TEMPLATE_EXAMPLES_DIR / name


def _run_validation_command(cmd: List[str], label: str) -> int:
    print(f"\n  [{label}] {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  ✗ {label} failed (exit={proc.returncode})")
    else:
        print(f"  ✓ {label} complete")
    return int(proc.returncode)


def run_posthoc_validation(args) -> None:
    if not args.run_validation:
        return

    print()
    print("=" * 60)
    print(" POST-HOC ANNOTATED VALIDATION")
    print("=" * 60)

    engine = str(getattr(args, "validation_engine", "both") or "both").strip().lower()
    run_legacy = engine in {"legacy", "both"}
    run_evaluate = engine in {"evaluate", "both"}

    metrics_script = _resolve_metrics_script()
    if run_legacy and metrics_script is None:
        print("  ⚠ Legacy validation skipped: no validation metrics script found.")
        run_legacy = False
    if run_legacy and not VALIDATION_DETAILED_SCRIPT.exists():
        print("  ⚠ Legacy validation skipped: detailed validation script not found.")
        run_legacy = False
    if run_evaluate and not VALIDATION_EVALUATE_SCRIPT.exists():
        print("  ⚠ Evaluation skipped: run_evaluation.py not found.")
        run_evaluate = False
    if not (run_legacy or run_evaluate):
        return

    results_dir = _resolve_validation_results_dir(args.validation_results_dir)
    if results_dir is None or not results_dir.exists():
        print("  ⚠ Skipped: validation results directory not found.")
        return

    prediction_type = str(args.prediction_type or "binary").strip().lower()
    needs_binary_targets = prediction_type == "binary"
    targets_file = str(args.validation_targets_file or "").strip()
    annotations_json = str(args.validation_annotations_json or "").strip()

    if needs_binary_targets and not targets_file:
        print("  ⚠ Skipped: binary validation requires --validation_targets_file.")
        print(f"  ↳ Template example: {_validation_template_hint(prediction_type)}")
        return
    if (not needs_binary_targets) and not annotations_json:
        print("  ⚠ Skipped: non-binary validation requires --validation_annotations_json.")
        print(f"  ↳ Template example: {_validation_template_hint(prediction_type)}")
        return
    if needs_binary_targets and not Path(targets_file).exists():
        print(f"  ⚠ Skipped: validation targets file not found: {targets_file}")
        print(f"  ↳ Template example: {_validation_template_hint(prediction_type)}")
        return
    if needs_binary_targets and Path(targets_file).suffix.lower() != ".json":
        print(f"  ⚠ Skipped: binary validation requires JSON --validation_targets_file, got: {targets_file}")
        print("  ↳ Legacy txt targets are no longer supported.")
        print(f"  ↳ Template example: {_validation_template_hint(prediction_type)}")
        return
    if (not needs_binary_targets) and not Path(annotations_json).exists():
        print(f"  ⚠ Skipped: validation annotations file not found: {annotations_json}")
        print(f"  ↳ Template example: {_validation_template_hint(prediction_type)}")
        return

    output_root = Path(args.validation_output_dir) if str(args.validation_output_dir or "").strip() else (RESULTS_DIR / "analysis")
    if prediction_type == "binary":
        metrics_out = output_root / "binary_confusion_matrix"
        detailed_out = output_root / "details"
    else:
        metrics_out = output_root / f"{prediction_type}_metrics"
        detailed_out = output_root / f"{prediction_type}_details"
    evaluation_out = output_root / "evaluation"

    common_args = [
        "--results_dir", str(results_dir),
        "--prediction_type", prediction_type,
    ]
    if needs_binary_targets:
        common_args.extend(["--targets_file", targets_file])
    else:
        common_args.extend(["--annotations_json", annotations_json])
    if str(args.validation_disorder_groups or "").strip():
        common_args.extend(["--disorder_groups", str(args.validation_disorder_groups).strip()])

    print(f"  Results dir:      {results_dir}")
    print(f"  Template example: {_validation_template_hint(prediction_type)}")
    return_codes: List[int] = []

    if run_legacy:
        metrics_out.mkdir(parents=True, exist_ok=True)
        detailed_out.mkdir(parents=True, exist_ok=True)
        print(f"  Metrics output:   {metrics_out}")
        print(f"  Detailed output:  {detailed_out}")
        metrics_cmd = [sys.executable, str(metrics_script), *common_args, "--output_dir", str(metrics_out)]
        detailed_cmd = [sys.executable, str(VALIDATION_DETAILED_SCRIPT), *common_args, "--output_dir", str(detailed_out)]
        return_codes.append(_run_validation_command(metrics_cmd, "validation-metrics"))
        return_codes.append(_run_validation_command(detailed_cmd, "validation-detailed"))

    if run_evaluate:
        # Tidy evaluation library: every task kind, missing predictions counted,
        # bootstrap 95% CIs, standard figures (see with_annotated_dataset/README.md).
        evaluation_out.mkdir(parents=True, exist_ok=True)
        print(f"  Evaluation output: {evaluation_out}")
        evaluate_cmd = [
            sys.executable, str(VALIDATION_EVALUATE_SCRIPT), "evaluate",
            "--predictions", str(results_dir),
            "--annotations", targets_file if needs_binary_targets else annotations_json,
            "--out", str(evaluation_out),
        ]
        task_spec_file = str(getattr(args, "task_spec_file", "") or "").strip()
        if task_spec_file and Path(task_spec_file).exists():
            evaluate_cmd.extend(["--task-spec", task_spec_file])
        group_by = str(getattr(args, "validation_group_by", "") or "").strip()
        if group_by:
            evaluate_cmd.extend(["--group-by", group_by])
        return_codes.append(_run_validation_command(evaluate_cmd, "validation-evaluate"))

    if all(rc == 0 for rc in return_codes):
        print("  ✓ Post-hoc annotated validation complete.")
    else:
        print("  ⚠ Post-hoc annotated validation completed with errors.")


def build_parser() -> argparse.ArgumentParser:
    """The batch runner's command line. Unknown flags are passed to main.py unchanged."""
    parser = argparse.ArgumentParser(
        description=(
            "COMPASS Batch Runner: sequential participant processing. Options named like a main.py flag are "
            "forwarded to main.py when set; any other main.py flag is passed through unchanged."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # A prefix of a main.py flag must not be taken for a batch option.
        allow_abbrev=False,
        epilog="""
Examples:
  python src/full_stack/backend/utils/batch_run.py --backend openrouter --check_config
  python src/full_stack/backend/utils/batch_run.py --backend openrouter --public_model deepseek/deepseek-v4-flash-0731 \\
      --predictor_model typesafe/jev-1.13 --decision_provider typesafe --preflight_check
  python src/full_stack/backend/utils/batch_run.py --backend local --model Qwen/Qwen3-14B-AWQ --local_max_model_len 32768 \\
      --predictor_model typesafe/jev-1.13 --orchestration auto
        """,
    )
    parser.add_argument("--backend", choices=["openrouter", "openai", "local"], default="local")
    parser.add_argument(
        "--prediction_type",
        choices=["binary", "multiclass", "regression_univariate", "regression_multivariate", "hierarchical"],
        default="binary",
        help="Prediction task type passed to main.py",
    )
    parser.add_argument("--control_label", type=str, default=None)
    parser.add_argument("--class_labels", type=str, default="")
    parser.add_argument("--regression_outputs", type=str, default="")
    parser.add_argument("--regression_output", type=str, default="")
    parser.add_argument("--task_spec_file", type=str, default="")
    parser.add_argument("--task_spec_json", type=str, default="")

    hosted = parser.add_argument_group("hosted backends (openrouter, openai)")
    hosted.add_argument(
        "--public_model",
        type=str,
        default=None,
        help="Model every LLM role runs on (main.py default: deepseek/deepseek-v4-flash-0731). "
             "Without it, --model is used here on a hosted backend.",
    )
    hosted.add_argument("--public_max_context_tokens", type=int, default=None,
                        help="Context window for a hosted model the catalog and built-in table do not know.")

    routing = parser.add_argument_group("Predictor, routing and context windows (see python main.py --help)")
    routing.add_argument("--predictor_model", type=str, default=None,
                         help="Model for the Predictor only, for example typesafe/jev-1.13.")
    routing.add_argument("--companion_model", type=str, default=None,
                         help="Conventional LLM for every role except the Predictor (hosted backends).")
    routing.add_argument("--orchestration", choices=["auto", "always", "never"], default=None)
    routing.add_argument("--orchestration_threshold", type=int, default=None)
    routing.add_argument("--context_window", type=int, default=None,
                         help="Force this context window for every LLM role.")
    routing.add_argument("--role_context_window", action="append", default=None, metavar="ROLE=TOKENS",
                         help="Force one role's context window, for example predictor=128000,tool=32000 (repeatable).")
    routing.add_argument("--reasoning_effort", choices=["provider_default", "off", "low", "medium", "high"], default=None)

    decision = parser.add_argument_group("structured decision-model Predictor")
    decision.add_argument("--decision_provider", choices=["openrouter", "typesafe"], default=None)
    decision.add_argument("--decision_choice_orders", type=int, default=None)
    decision.add_argument("--decision_score_levels", type=int, default=None)
    decision.add_argument("--decision_refine", action=argparse.BooleanOptionalAction, default=None)
    decision.add_argument("--decision_compiler_model", type=str, default=None)
    decision.add_argument("--decision_stability_threshold", type=float, default=None)
    decision.add_argument("--decision_regression_stability_threshold", type=float, default=None)
    decision.add_argument("--decision_sufficiency_threshold", type=float, default=None)

    checks = parser.add_argument_group("offline configuration check")
    checks.add_argument(
        "--check_config",
        action="store_true",
        help="Run main.py --check_config for every participant (no model call, no spend), print the route "
             "each record takes, and exit (code 1 when a check reports a problem). Nothing is run.",
    )
    checks.add_argument(
        "--preflight_check",
        action="store_true",
        help="Run the same offline check first and start the cohort only when no check reports a problem.",
    )

    budgets = parser.add_argument_group("token budgets")
    budgets.add_argument("--max_agent_input", type=int, default=None)
    budgets.add_argument("--max_agent_output", type=int, default=None)
    budgets.add_argument("--max_tool_input", type=int, default=None)
    budgets.add_argument("--max_tool_output", type=int, default=None)

    local = parser.add_argument_group("local backend (--backend local)")
    local.add_argument("--model", type=str, default=None)
    local.add_argument("--max_tokens", type=int, default=32768)
    local.add_argument("--local_engine", type=str, default="auto")
    local.add_argument("--local_dtype", type=str, default="auto")
    local.add_argument("--local_quant", type=str, default=None)
    local.add_argument("--local_kv_cache_dtype", type=str, default=None)
    local.add_argument("--local_tensor_parallel", type=int, default=1)
    local.add_argument("--local_pipeline_parallel", type=int, default=1)
    local.add_argument("--local_gpu_mem_util", type=float, default=0.9)
    local.add_argument("--local_max_model_len", type=int, default=0)
    local.add_argument("--local_enforce_eager", action="store_true")
    local.add_argument("--local_trust_remote_code", action="store_true")
    local.add_argument("--local_attn", type=str, default="auto")

    validation = parser.add_argument_group("post-hoc annotated validation")
    validation.add_argument(
        "--run_validation",
        action="store_true",
        help="Run post-hoc annotated validation after batch completion.",
    )
    validation.add_argument(
        "--validation_results_dir",
        type=str,
        default="",
        help="Directory containing participant output folders for validation (default: auto-detect).",
    )
    validation.add_argument(
        "--validation_output_dir",
        type=str,
        default="",
        help="Output root for validation artifacts (default: ../results/analysis).",
    )
    validation.add_argument(
        "--validation_targets_file",
        type=str,
        default="",
        help="Binary ground-truth targets JSON file (required when --prediction_type=binary and --run_validation).",
    )
    validation.add_argument(
        "--validation_annotations_json",
        type=str,
        default="",
        help="Generalized annotations JSON (required for non-binary validation).",
    )
    validation.add_argument(
        "--validation_disorder_groups",
        type=str,
        default="",
        help="Optional comma-separated groups/disorders for subgroup validation artifacts.",
    )
    validation.add_argument(
        "--validation_engine",
        choices=["legacy", "evaluate", "both"],
        default="both",
        help="Validation to run: legacy mode-specific scripts, the tidy evaluation library, or both.",
    )
    validation.add_argument(
        "--validation_group_by",
        type=str,
        default="",
        help="Comma-separated grouping columns for the evaluation library (for example disorder,route).",
    )
    return parser


def batch_args_from_args(args: argparse.Namespace, extra_main_args: Sequence[str] = ()) -> Dict[str, Any]:
    """The options run_participant forwards to main.py, from the parsed batch command line."""
    regression_output = str(args.regression_output or "").strip()
    regression_outputs = str(args.regression_outputs or "").strip()
    if regression_output and regression_outputs:
        parsed_multi = [x.strip() for x in regression_outputs.split(",") if x.strip()]
        if parsed_multi != [regression_output]:
            raise ValueError("--regression_output conflicts with --regression_outputs. Use one form or provide matching values.")

    batch_args: Dict[str, Any] = {
        "backend": args.backend,
        "prediction_type": args.prediction_type,
        "regression_outputs": regression_outputs,
        "regression_output": regression_output,
        "extra_main_args": list(extra_main_args),
    }
    for name in (
        *TASK_VALUE_FLAGS,
        *HOSTED_VALUE_FLAGS,
        *LOCAL_VALUE_FLAGS,
        *LOCAL_SWITCH_FLAGS,
        *RUN_VALUE_FLAGS,
        *RUN_BOOLEAN_FLAGS,
        *RUN_REPEATED_FLAGS,
    ):
        batch_args[name] = getattr(args, name, None)
    # main.py reads --model only on --backend local. On a hosted backend this
    # runner used to forward --model too, where main.py silently ignored it;
    # keep that spelling working by treating it as the hosted model.
    if args.backend != "local" and _is_set(args.model) and not _is_set(args.public_model):
        batch_args["public_model"] = args.model
    return batch_args


def main(argv: Optional[Sequence[str]] = None):
    parser = build_parser()
    args, extra_main_args = parser.parse_known_args(argv)
    try:
        batch_args = batch_args_from_args(args, extra_main_args)
    except ValueError as exc:
        parser.error(str(exc))
    BATCH_ARGS.clear()
    BATCH_ARGS.update(batch_args)

    n = len(PARTICIPANTS)
    n_cases = sum(1 for p in PARTICIPANTS if p["expected"] == "CASE")
    n_controls = n - n_cases
    model_line = BATCH_ARGS.get("model") if args.backend == "local" else BATCH_ARGS.get("public_model")
    role_windows = ",".join(BATCH_ARGS.get("role_context_window") or []) or None

    print("=" * 60)
    print(" COMPASS Batch Runner")
    print("=" * 60)
    print(f"  Participants: {n} ({n_cases} CASE, {n_controls} CONTROL)")
    print(f"  Data root:    {DATA_ROOT}")
    print(f"  Backend:      {args.backend}")
    print(f"  Prediction:   {args.prediction_type}")
    print(f"  Model:        {model_line or '<main.py default>'}")
    print(f"  Predictor:    {args.predictor_model or '<same model>'}")
    if args.companion_model:
        print(f"  Companion:    {args.companion_model}")
    print(
        f"  Routing:      {args.orchestration or 'auto (main.py default)'}"
        + (f", threshold {args.orchestration_threshold}" if args.orchestration_threshold is not None else "")
    )
    if args.context_window is not None or role_windows:
        print(f"  Windows:      context_window={args.context_window}, role_context_window={role_windows}")
    if args.backend == "local":
        print(f"  Context:      {args.max_tokens}")
    print(
        f"  Budgets:      agent(in={args.max_agent_input}, out={args.max_agent_output}) | "
        f"tool(in={args.max_tool_input}, out={args.max_tool_output})"
    )
    if args.backend == "local":
        print(
            f"  Local cfg:    engine={args.local_engine}, dtype={args.local_dtype}, "
            f"quant={args.local_quant}, gpu_mem={args.local_gpu_mem_util}, "
            f"max_model_len={args.local_max_model_len}"
        )
    elif _is_set(args.model) and not _is_set(args.public_model):
        print(f"  Note:         --model {args.model} is used as --public_model on the {args.backend} backend.")
    if extra_main_args:
        print(f"  Passed on:    {' '.join(extra_main_args)}")
    print(f"  Validation:   {'ON' if args.run_validation else 'OFF'}")
    if args.run_validation:
        print(f"  Val results:  {args.validation_results_dir or '<auto>'}")
        print(f"  Val output:   {args.validation_output_dir or str(RESULTS_DIR / 'analysis')}")
        if args.prediction_type == "binary":
            print(f"  Val targets:  {args.validation_targets_file or '<required>'}")
        else:
            print(f"  Val annjson:  {args.validation_annotations_json or '<required>'}")
    print(f"  Processing:   SEQUENTIAL (1 GPU)")
    print("=" * 60)

    if args.check_config:
        sys.exit(run_config_checks())
    if args.preflight_check and run_config_checks() != 0:
        print("  Preflight check failed: no participant was run. Fix the problems above, or drop --preflight_check.")
        sys.exit(1)
    print()

    results = {}
    timings = {}
    starts = {}
    logs = {}
    batch_start = time.time()

    # Launch sequentially
    for i, p in enumerate(PARTICIPANTS, 1):
        pid = p["id"]
        print(f"\n{'─' * 60}")
        print(f" [{i}/{n}] Participant {pid} ({p['expected']})")
        print(f"{'─' * 60}")

        path = resolve_participant_path(pid)
        if not os.path.exists(path):
            print(f"  ✗ Participant folder not found: {path} (skipped)")
            results[pid] = "MISSING"
            continue

        t0 = time.time()
        # File times have coarse resolution on some file systems.
        starts[pid] = t0 - 2.0
        proc, pid, out_file, out_path = run_participant(p)
        logs[pid] = out_path

        # Wait for this one to finish immediately
        proc.wait()
        out_file.close()

        elapsed = time.time() - t0
        timings[pid] = elapsed
        td = timedelta(seconds=int(elapsed))

        if proc.returncode != 0:
            print(f"  ✗ ERROR for {pid} (exit code {proc.returncode}), {td}")
            results[pid] = "ERROR"
            if proc.returncode == USAGE_ERROR_EXIT_CODE:
                try:
                    for line in _error_lines(Path(out_path).read_text(errors="replace"), 8):
                        print(f"      {line}")
                except OSError:
                    pass
                print("  main.py refused the command line; the remaining participants are skipped.")
                for rest in PARTICIPANTS[i:]:
                    results[rest["id"]] = "SKIPPED"
                break
        else:
            print(f"  ✓ Finished {pid}, {td}")
            results[pid] = "DONE"

    batch_elapsed = time.time() - batch_start
    batch_td = timedelta(seconds=int(batch_elapsed))

    # ─── Timing Summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(" TIMING SUMMARY")
    print("=" * 60)
    for p in PARTICIPANTS:
        pid = p["id"]
        t = timings.get(pid, 0)
        td = timedelta(seconds=int(t))
        status = results.get(pid, "UNKNOWN")
        print(f"  {pid} ({p['expected']:>7}): {td}  [{status}]")
    print(f"\n  Total batch wall time: {batch_td}")
    if timings:
        avg = sum(timings.values()) / len(timings)
        print(f"  Avg per participant:   {timedelta(seconds=int(avg))}")

    # ─── Classification Summary (binary mode only) ──────────────────────
    if args.prediction_type != "binary":
        print()
        print("=" * 60)
        print(" SUMMARY")
        print("=" * 60)
        print("  Binary confusion summary is skipped for non-binary prediction modes.")
        print("  Use annotated-dataset validation scripts for multiclass/regression/hierarchical metrics.")
        print(f"  Total wall time: {batch_td}")
        run_posthoc_validation(args)
        return

    print()
    print("=" * 60)
    print(" CLASSIFICATION SUMMARY")
    print("=" * 60)

    correct = 0
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for p in PARTICIPANTS:
        pid = p["id"]
        expected = p["expected"]

        # Outcome from the report this batch's run wrote (root binary node)
        actual = "UNKNOWN"
        if results.get(pid) == "DONE":
            actual = read_binary_outcome(
                pid, resolve_participant_path(pid), since=starts.get(pid), log_path=logs.get(pid)
            )

        # Score
        is_correct = (actual == expected)
        if is_correct: correct += 1

        # Confusion Matrix
        if expected == "CASE" and actual == "CASE": confusion["TP"] += 1
        elif expected == "CONTROL" and actual == "CONTROL": confusion["TN"] += 1
        elif expected == "CONTROL" and actual == "CASE": confusion["FP"] += 1
        elif expected == "CASE" and actual == "CONTROL": confusion["FN"] += 1

        t = timings.get(pid, 0)
        td = timedelta(seconds=int(t))
        marker = "✓" if is_correct else "✗"
        print(f"  {marker} {pid}: Expected {expected:>7} → Actual {actual:>7}  ({td})")

    print(f"\n  CONFUSION MATRIX:")
    print(f"    TP: {confusion['TP']}  FN: {confusion['FN']}")
    print(f"    FP: {confusion['FP']}  TN: {confusion['TN']}")
    print(f"\n  Accuracy: {correct}/{len(PARTICIPANTS)}")
    print(f"  Total wall time: {batch_td}")
    run_posthoc_validation(args)

if __name__ == "__main__":
    main()
