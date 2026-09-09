"""
Run worker: executes one COMPASS pipeline in its own process.

Isolation is the point. The engine configures itself through a process-global
settings singleton and a global LLM client, so a worker per run is what makes
concurrent batch execution, per-run model overrides, and hard cancellation safe.

The worker speaks newline-delimited JSON on stdout:

    {"t": "ready"}
    {"t": "event",  "event": {...}, "state": {...}}
    {"t": "usage",  "by_model": {...}, "totals": {...}}
    {"t": "log",    "level": "info", "message": "..."}
    {"t": "result", "result": {...}}
    {"t": "error",  "error": "...", "traceback": "..."}
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_stdout_lock = threading.Lock()

#: The protocol owns the real stdout. The engine's own prints are redirected to
#: stderr during `claim_stdout`, so nothing can interleave inside a JSON line.
_protocol = sys.stdout


def claim_stdout() -> None:
    """Reserve file descriptor 1 for the protocol and send prints to stderr."""
    global _protocol
    duplicated = os.dup(sys.stdout.fileno())
    _protocol = os.fdopen(duplicated, "w", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = sys.stderr


def emit(payload: Dict[str, Any]) -> None:
    line = json.dumps(payload, default=str, ensure_ascii=False)
    with _stdout_lock:
        _protocol.write(line + "\n")
        _protocol.flush()


def _usage_snapshot() -> Dict[str, Any]:
    """Exact per-model token counts straight from the engine's LLM client."""
    from src.full_stack.backend.utils.llm_client import _llm_client_instance

    client = _llm_client_instance
    if client is None:
        return {"by_model": {}, "totals": {"prompt": 0, "completion": 0, "calls": 0}}
    tracker = getattr(client, "token_tracker", None)
    if tracker is None:
        return {"by_model": {}, "totals": {"prompt": 0, "completion": 0, "calls": 0}}

    by_model: Dict[str, Dict[str, int]] = {}
    for call in list(getattr(tracker, "calls", [])):
        model = str(call.get("model") or "unknown")
        bucket = by_model.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
        bucket["prompt"] += int(call.get("prompt") or 0)
        bucket["completion"] += int(call.get("completion") or 0)
        bucket["calls"] += 1
    return {
        "by_model": by_model,
        "totals": {
            "prompt": int(getattr(tracker, "total_prompt_tokens", 0)),
            "completion": int(getattr(tracker, "total_completion_tokens", 0)),
            "calls": len(getattr(tracker, "calls", [])),
        },
    }


def _start_usage_poller(stop: threading.Event, interval: float = 1.5) -> threading.Thread:
    parent = os.getppid()

    def loop() -> None:
        last = ""
        while not stop.is_set():
            # If the supervisor is gone nobody is reading, and the run would
            # otherwise keep calling the provider and billing for it.
            if os.getppid() != parent:
                os._exit(143)
            try:
                snapshot = _usage_snapshot()
                encoded = json.dumps(snapshot, sort_keys=True)
                if encoded != last and snapshot["totals"]["calls"]:
                    last = encoded
                    emit({"t": "usage", **snapshot})
            except Exception:
                pass
            stop.wait(interval)

    thread = threading.Thread(target=loop, name="usage-poller", daemon=True)
    thread.start()
    return thread


def main() -> int:
    if len(sys.argv) < 2:
        emit({"t": "error", "error": "worker requires a job file path"})
        return 2

    claim_stdout()
    job = json.loads(Path(sys.argv[1]).read_text())

    from src.full_stack.backend.api.config_store import DashboardConfig
    from src.full_stack.backend.api.engine_bridge import (
        apply_config_to_settings,
        build_task_spec,
        normalise_instructions,
    )
    from src.full_stack.backend.api.schemas import TaskSpecInput
    from src.full_stack.backend.runtime.event_bus import attach_sink, get_ui
    from src.full_stack.backend.utils.llm_client import reset_llm_client

    config = DashboardConfig.model_validate(job["config"])
    task = TaskSpecInput.model_validate(job["task"])
    participant_dir = Path(job["participant_dir"]).expanduser().resolve()

    attach_sink(lambda payload: emit({"t": "event", **payload}))
    ui = get_ui(enabled=True)
    ui.enabled = True

    settings = apply_config_to_settings(config)
    if job.get("output_dir"):
        chosen = Path(str(job["output_dir"])).expanduser()
        chosen.mkdir(parents=True, exist_ok=True)
        settings.paths.output_dir = chosen
    reset_llm_client()

    task_spec = build_task_spec(task)
    target_condition, control_condition = task_spec.legacy_target_control()
    instructions = normalise_instructions(config.instructions)

    emit(
        {
            "t": "ready",
            "participant_dir": str(participant_dir),
            "participant_id": participant_dir.name,
            "task": task_spec.model_dump(mode="json"),
            "pid": os.getpid(),
        }
    )

    stop = threading.Event()
    _start_usage_poller(stop)

    try:
        import main as compass_main

        if job.get("audit"):
            result = compass_main.run_dataflow_audit(
                participant_dir=participant_dir,
                target_condition=target_condition,
                control_condition=control_condition,
                prediction_task_spec=task_spec,
                verbose=config.engine.verbose,
            )
        else:
            result = compass_main.run_compass_pipeline(
                participant_dir=participant_dir,
                target_condition=target_condition,
                control_condition=control_condition,
                prediction_task_spec=task_spec,
                agent_instructions=instructions,
                max_iterations=config.engine.max_iterations,
                verbose=config.engine.verbose,
                interactive_ui=True,
                generate_deep_phenotype=bool(job.get("generate_deep_phenotype", True)),
                generate_xai_report=False,
                deep_report_focus_modalities=str(job.get("focus_modalities") or ""),
                deep_report_general_instruction=str(job.get("general_instruction") or ""),
            )
    except BaseException as exc:  # surfaced verbatim; the engine fails loudly by design
        stop.set()
        emit({"t": "usage", **_usage_snapshot()})
        emit({"t": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return 1

    stop.set()
    time.sleep(0.1)
    emit({"t": "usage", **_usage_snapshot()})

    payload = dict(result or {})
    payload.pop("internal_context", None)  # live pydantic objects, not transportable
    emit({"t": "result", "result": payload})
    return 0


if __name__ == "__main__":
    sys.exit(main())
