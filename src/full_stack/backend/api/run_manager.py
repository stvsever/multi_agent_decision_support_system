"""
Run supervision.

Each run is a worker subprocess whose newline-delimited JSON output is folded
into a live record and broadcast to every connected browser. Records persist
under the user's config directory so history survives a restart.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config.settings import PROJECT_ROOT
from ..runtime.event_bus import STAGE_NAMES
from .config_store import load_config, worker_environment
from .cost import (
    actual_cost_from_usage,
    estimate_run,
    expected_plan_steps,
    participant_domain_count,
    participant_input_tokens,
)
from .engine_bridge import build_task_spec, describe_task, merge_overrides
from .paths import runs_dir
from .schemas import DashboardConfig, RunRequest
from .flow import plan_to_graph

MAX_LIVE_EVENTS = 4000
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunRecord:
    """Everything the interface knows about one run."""

    def __init__(
        self,
        run_id: str,
        *,
        participant_dir: Path,
        config: DashboardConfig,
        request: RunRequest,
        task_summary: Dict[str, Any],
        estimate: Dict[str, Any],
        batch_id: Optional[str] = None,
        audit: bool = False,
    ) -> None:
        self.id = run_id
        self.participant_dir = participant_dir
        self.participant_id = participant_dir.name
        self.label = request.label or participant_dir.name
        self.config = config
        self.request = request
        self.task_summary = task_summary
        self.estimate = estimate
        self.batch_id = batch_id
        self.audit = audit

        self.status = "queued"
        self.created_at = _now()
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.error: Optional[str] = None
        self.traceback: Optional[str] = None
        self.exit_code: Optional[int] = None

        self.state: Dict[str, Any] = {
            "status": "Queued",
            "current_stage": -1,
            "stages": list(STAGE_NAMES),
            "steps": [],
            "history": [],
            "iteration": 1,
            "total_tokens": 0,
            "progress": 0,
            "max_steps": 1,
            "completed": False,
        }
        self.events: List[Dict[str, Any]] = []
        self.usage: Dict[str, Any] = {"by_model": {}, "totals": {"prompt": 0, "completion": 0, "calls": 0}}
        self.cost: Dict[str, Any] = {"usd": None, "lines": [], "total_tokens": 0, "fully_priced": False}
        self.result: Optional[Dict[str, Any]] = None
        self.graph: Optional[Dict[str, Any]] = None
        self.logs: List[str] = []

        self._process: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self.cancel_requested = False
        self._lock = threading.RLock()
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self.dir = runs_dir() / run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- serialisation --------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "participant_id": self.participant_id,
            "participant_dir": str(self.participant_dir),
            "status": self.status,
            "audit": self.audit,
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "task": self.task_summary,
            "estimate": {
                "usd": self.estimate.get("usd"),
                "total_tokens": self.estimate.get("total_tokens"),
            },
            "cost": {"usd": self.cost.get("usd"), "total_tokens": self.cost.get("total_tokens")},
            "stage": self.state.get("current_stage"),
            "progress": self.state.get("progress"),
            "max_steps": self.state.get("max_steps"),
            "verdict": (self.result or {}).get("verdict"),
            "prediction": (self.result or {}).get("prediction"),
        }

    def detail(self, since_event: int = 0) -> Dict[str, Any]:
        with self._lock:
            return {
                **self.summary(),
                "state": self.state,
                "events": [e for e in self.events if int(e.get("id") or 0) > since_event],
                "usage": self.usage,
                "cost": self.cost,
                "estimate": self.estimate,
                "result": self.result,
                "graph": self.graph,
                "logs": self.logs[-200:],
                "traceback": self.traceback,
                "effective_config": self.config.model_dump(mode="json"),
            }

    def persist(self) -> None:
        with self._lock:
            payload = json.dumps(
                {
                    **self.summary(),
                    "state": self.state,
                    "usage": self.usage,
                    "cost": self.cost,
                    "result": self.result,
                    "graph": self.graph,
                    "estimate": self.estimate,
                    "exit_code": self.exit_code,
                },
                indent=2,
                default=str,
            )
        # Write then rename: a reader never sees a half-written record.
        target = self.dir / "meta.json"
        try:
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(payload)
            tmp.replace(target)
        except OSError:
            pass

    # -- broadcasting ---------------------------------------------------------

    def subscribe(self, sink: Callable[[Dict[str, Any]], None]) -> Callable[[Dict[str, Any]], None]:
        with self._lock:
            self._subscribers.append(sink)
        return sink

    def unsubscribe(self, sink: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            if sink in self._subscribers:
                self._subscribers.remove(sink)

    def _broadcast(self, message: Dict[str, Any]) -> None:
        with self._lock:
            sinks = list(self._subscribers)
        for sink in sinks:
            try:
                sink(message)
            except Exception:
                # A disconnected reader must never take the run down with it.
                pass

    # -- lifecycle ------------------------------------------------------------

    def _set_status(self, status: str, **extra: Any) -> bool:
        """
        Move to `status`, returning whether the move happened.

        A terminal state is final: the pump thread and an HTTP cancel can reach
        this concurrently, and whichever lands first decides the outcome.
        """
        with self._lock:
            if self.status in TERMINAL_STATES and status != self.status:
                return False
            self.status = status
            for key, value in extra.items():
                setattr(self, key, value)
        self._broadcast({"type": "status", "run": self.summary()})
        self.persist()
        return True

    def _append_event(self, event: Dict[str, Any], state: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)
            if len(self.events) > MAX_LIVE_EVENTS:
                del self.events[: len(self.events) - MAX_LIVE_EVENTS]
            self.state = state
            if event.get("type") == "PLAN":
                plan = (event.get("data") or {}).get("plan") or {}
                self.graph = plan_to_graph(plan)
        payload = {"type": "event", "event": event, "state": state}
        if event.get("type") == "PLAN":
            payload["graph"] = self.graph
        self._broadcast(payload)

    def _apply_usage(self, usage: Dict[str, Any]) -> None:
        # Priced from the cache only: this runs on the thread draining the
        # worker's stdout, and a network round trip here would stall the run.
        cost = actual_cost_from_usage(usage.get("by_model") or {}, cached_only=True)
        with self._lock:
            self.usage = usage
            self.cost = cost
        self._broadcast({"type": "usage", "usage": usage, "cost": cost})

    def start(self) -> None:
        job = {
            "run_id": self.id,
            "participant_dir": str(self.participant_dir),
            "config": self.config.model_dump(mode="json"),
            "task": self.request.task.model_dump(mode="json"),
            "generate_deep_phenotype": self.request.generate_deep_phenotype,
            "audit": self.audit,
            "output_dir": self.config.workspace.output_dir or "",
        }
        job_file = self.dir / "job.json"
        job_file.write_text(json.dumps(job, indent=2, default=str))

        env = worker_environment(self.config)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)

        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", "src.full_stack.backend.api.worker", str(job_file)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._set_status("running", started_at=_now())
        threading.Thread(target=self._pump_stdout, name=f"run-{self.id}-out", daemon=True).start()
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr, name=f"run-{self.id}-err", daemon=True
        )
        self._stderr_thread.start()

    def _pump_stdout(self) -> None:
        assert self._process is not None
        stream = self._process.stdout
        raw_log = (self.dir / "events.ndjson").open("a", encoding="utf-8")
        try:
            for line in stream or []:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._record_log(line)
                    continue
                if not isinstance(message, dict):
                    self._record_log(line)
                    continue
                raw_log.write(line + "\n")
                raw_log.flush()
                try:
                    self._handle(message)
                except Exception as exc:  # a bad message must not stop the drain
                    self._record_log(f"[dashboard] could not apply a worker message: {exc}")
        finally:
            raw_log.close()
            self._finalise()

    def _pump_stderr(self) -> None:
        assert self._process is not None
        for line in self._process.stderr or []:
            text = line.rstrip()
            if text:
                self._record_log(text)

    def _record_log(self, text: str) -> None:
        with self._lock:
            self.logs.append(text)
            if len(self.logs) > 2000:
                del self.logs[: len(self.logs) - 2000]
        self._broadcast({"type": "log", "line": text})

    def _handle(self, message: Dict[str, Any]) -> None:
        kind = message.get("t")
        if kind == "event":
            self._append_event(message.get("event") or {}, message.get("state") or {})
        elif kind == "usage":
            self._apply_usage({"by_model": message.get("by_model") or {}, "totals": message.get("totals") or {}})
        elif kind == "result":
            with self._lock:
                self.result = message.get("result") or {}
            self._broadcast({"type": "result", "result": self.result})
        elif kind == "error":
            with self._lock:
                self.error = str(message.get("error") or "Run failed.")
                self.traceback = message.get("traceback")
        elif kind == "log":
            self._record_log(str(message.get("message") or ""))
        elif kind == "ready":
            self._broadcast({"type": "ready", "data": message})

    def _finalise(self) -> None:
        process = self._process
        code = process.wait() if process else -1

        # Let the stderr drain finish: a worker that died during import writes
        # its traceback there and nowhere else.
        thread = self._stderr_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)

        with self._lock:
            self.exit_code = code
            cancelling = self.status == "cancelling"
            has_error = bool(self.error)
            if not cancelling and code != 0 and not has_error:
                tail = "\n".join(self.logs[-12:]).strip()
                self.error = f"Worker exited with code {code}." + (f"\n{tail}" if tail else "")

        if cancelling:
            status = "cancelled"
        elif code == 0 and not has_error:
            status = "succeeded"
        else:
            status = "failed"
        self._set_status(status, finished_at=_now())
        self._broadcast({"type": "done", "run": self.summary()})

    def cancel(self) -> bool:
        with self._lock:
            if self.status in TERMINAL_STATES:
                return True
            self.cancel_requested = True
            process = self._process

        if process is None or process.poll() is not None:
            if self._set_status("cancelled", finished_at=_now()):
                self._broadcast({"type": "done", "run": self.summary()})
            return True

        self._set_status("cancelling")
        try:
            process.send_signal(signal.SIGTERM)
        except OSError:
            return False
        # Daemon, so a Ctrl-C during an active run does not stall the exit.
        timer = threading.Timer(6.0, self._force_kill)
        timer.daemon = True
        timer.start()
        return True

    def _force_kill(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass


class ArchivedRun:
    """
    A finished run reloaded from disk after a restart.

    It exposes the same read surface as a live record so history, reports, and
    cost comparisons survive restarting the service, but it owns no process and
    cannot be cancelled or streamed.
    """

    def __init__(self, run_id: str, payload: Dict[str, Any], directory: Path) -> None:
        self.id = run_id
        self.dir = directory
        self.archived = True
        self._payload = payload
        self.status = str(payload.get("status") or "succeeded")
        self.created_at = str(payload.get("created_at") or "")
        self.batch_id = payload.get("batch_id")
        self.participant_id = str(payload.get("participant_id") or "")
        self.participant_dir = Path(str(payload.get("participant_dir") or "."))
        self.label = str(payload.get("label") or self.participant_id)
        self.error = payload.get("error")
        self.result = payload.get("result")
        self.cost = payload.get("cost") or {"usd": None, "total_tokens": 0, "lines": []}
        self.usage = payload.get("usage") or {"by_model": {}, "totals": {}}
        self.state = payload.get("state") or {}
        self.graph = payload.get("graph")
        self.estimate = payload.get("estimate") or {}
        self.config = load_config()

    def summary(self) -> Dict[str, Any]:
        keys = (
            "id", "label", "participant_id", "participant_dir", "status", "audit", "batch_id",
            "created_at", "started_at", "finished_at", "error", "task", "estimate", "cost",
            "stage", "progress", "max_steps", "verdict", "prediction",
        )
        return {**{k: self._payload.get(k) for k in keys}, "archived": True}

    def detail(self, since_event: int = 0) -> Dict[str, Any]:
        events = []
        ndjson = self.dir / "events.ndjson"
        if ndjson.exists():
            for line in ndjson.read_text(errors="replace").splitlines():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if message.get("t") == "event":
                    event = message.get("event") or {}
                    if int(event.get("id") or 0) > since_event:
                        events.append(event)
        return {
            **self.summary(),
            "state": self.state,
            "events": events,
            "usage": self.usage,
            "cost": self.cost,
            "estimate": self.estimate,
            "result": self.result,
            "graph": self.graph,
            "logs": [],
            "traceback": None,
            "effective_config": self.config.model_dump(mode="json"),
        }

    def cancel(self) -> bool:
        return True

    def subscribe(self, sink: Callable[[Dict[str, Any]], None]) -> Callable[[Dict[str, Any]], None]:
        # Nothing more will happen to a finished run: say so and be done.
        try:
            sink({"type": "done", "run": self.summary()})
        except Exception:
            pass
        return sink

    def unsubscribe(self, sink: Callable[[Dict[str, Any]], None]) -> None:
        return None


class RunManager:
    """Registry, scheduler, and history for runs and batches."""

    def __init__(self, rehydrate: bool = True) -> None:
        self._runs: Dict[str, Any] = {}
        self._batches: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        if rehydrate:
            self._load_history()

    def _load_history(self, limit: int = 200) -> None:
        """Reload finished runs so history is not lost on restart."""
        try:
            directories = sorted(runs_dir().iterdir(), key=lambda p: p.name, reverse=True)
        except OSError:
            return
        for directory in directories[:limit]:
            meta = directory / "meta.json"
            if not directory.is_dir() or not meta.exists():
                continue
            try:
                payload = json.loads(meta.read_text() or "{}")
            except json.JSONDecodeError:
                continue
            run_id = str(payload.get("id") or directory.name)
            if run_id in self._runs:
                continue
            # A run recorded as in-flight cannot still be running: its worker
            # died with the previous process.
            if payload.get("status") not in TERMINAL_STATES:
                payload["status"] = "cancelled"
                payload["error"] = payload.get("error") or "The service restarted while this run was in flight."
            self._runs[run_id] = ArchivedRun(run_id, payload, directory)

    # -- creation -------------------------------------------------------------

    def create_run(
        self,
        request: RunRequest,
        *,
        audit: bool = False,
        batch_id: Optional[str] = None,
        autostart: bool = True,
    ) -> RunRecord:
        participant_dir = Path(request.participant_dir).expanduser().resolve()
        if not participant_dir.is_dir():
            raise FileNotFoundError(f"No participant directory at {participant_dir}")

        config = merge_overrides(load_config(refresh=True), request.overrides)
        task_spec = build_task_spec(request.task)
        estimate = estimate_run(
            config=config,
            input_tokens=participant_input_tokens(participant_dir),
            plan_steps=expected_plan_steps(participant_domain_count(participant_dir)),
            include_deep_report=request.generate_deep_phenotype and not audit,
        )

        block_above = config.cost.block_above_usd
        projected = estimate.get("usd")
        if not audit and block_above and projected is not None and projected > block_above:
            raise ValueError(
                f"Projected cost ${projected:.4f} exceeds the configured hard stop of "
                f"${block_above:.2f}. Raise the limit in Settings to proceed."
            )

        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        record = RunRecord(
            run_id,
            participant_dir=participant_dir,
            config=config,
            request=request,
            task_summary=describe_task(task_spec),
            estimate=estimate,
            batch_id=batch_id or request.batch_id,
            audit=audit,
        )
        with self._lock:
            self._runs[run_id] = record
        record.persist()
        if autostart:
            record.start()
        return record

    def create_batch(
        self,
        *,
        participant_dirs: List[str],
        request_factory,
        concurrency: int,
        continue_on_error: bool,
        label: str = "",
    ) -> Dict[str, Any]:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        records: List[RunRecord] = []
        try:
            for participant_dir in participant_dirs:
                records.append(
                    self.create_run(request_factory(participant_dir), batch_id=batch_id, autostart=False)
                )
        except Exception:
            # Roll back, or the caller gets an error while queued runs linger in
            # the registry belonging to a batch that never existed.
            for record in records:
                self.delete_run(record.id)
            raise
        batch = {
            "id": batch_id,
            "label": label or f"{len(records)} participants",
            "created_at": _now(),
            "run_ids": [r.id for r in records],
            "concurrency": max(1, int(concurrency)),
            "continue_on_error": bool(continue_on_error),
            "status": "running",
            "cancelled": False,
        }
        with self._lock:
            self._batches[batch_id] = batch
        threading.Thread(
            target=self._drive_batch, args=(batch_id,), name=f"batch-{batch_id}", daemon=True
        ).start()
        return batch

    def _drive_batch(self, batch_id: str) -> None:
        batch = self._batches[batch_id]
        pending = list(batch["run_ids"])
        active: List[RunRecord] = []
        failed = False

        while pending or active:
            if batch.get("cancelled"):
                for record in active:
                    record.cancel()
                for run_id in pending:
                    run = self._runs.get(run_id)
                    if run and run.status == "queued":
                        run._set_status("cancelled", finished_at=_now())
                pending.clear()
                break

            while pending and len(active) < batch["concurrency"]:
                if failed and not batch["continue_on_error"]:
                    break
                record = self._runs.get(pending.pop(0))
                # A run can be cancelled or deleted while it waits its turn.
                if record is None or record.status in TERMINAL_STATES or record.cancel_requested:
                    continue
                record.start()
                active.append(record)

            time.sleep(0.4)
            for record in list(active):
                if record.status in TERMINAL_STATES:
                    active.remove(record)
                    if record.status == "failed":
                        failed = True

            if failed and not batch["continue_on_error"] and not active:
                for run_id in pending:
                    run = self._runs.get(run_id)
                    if run and run.status == "queued":
                        run._set_status("cancelled", finished_at=_now())
                pending.clear()

        batch["status"] = "cancelled" if batch.get("cancelled") else ("failed" if failed else "completed")
        batch["finished_at"] = _now()

    # -- lookup ---------------------------------------------------------------

    def get(self, run_id: str) -> Optional[Any]:
        return self._runs.get(run_id)

    def list_runs(self, limit: int = 100, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._runs.values())
        if batch_id:
            records = [r for r in records if r.batch_id == batch_id]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return [r.summary() for r in records[:limit]]

    def batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        batch = self._batches.get(batch_id)
        if not batch:
            return None
        with self._lock:
            records = [self._runs.get(i) for i in batch["run_ids"]]
        runs = [record.summary() for record in records if record is not None]
        done = [r for r in runs if r["status"] in TERMINAL_STATES]
        spend = sum(float((r.get("cost") or {}).get("usd") or 0) for r in runs)
        return {
            **batch,
            "runs": runs,
            "completed": len(done),
            "total": len(runs),
            "succeeded": sum(1 for r in runs if r["status"] == "succeeded"),
            "failed": sum(1 for r in runs if r["status"] == "failed"),
            "spend_usd": round(spend, 6),
        }

    def list_batches(self) -> List[Dict[str, Any]]:
        with self._lock:
            ids = list(self._batches.keys())
        out = [self.batch_status(i) for i in ids]
        return sorted([b for b in out if b], key=lambda b: b["created_at"], reverse=True)

    def cancel_batch(self, batch_id: str) -> bool:
        batch = self._batches.get(batch_id)
        if not batch:
            return False
        batch["cancelled"] = True
        return True

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            record = self._runs.pop(run_id, None)
        if record is None:
            return False
        if record.status not in TERMINAL_STATES:
            record.cancel()
        shutil.rmtree(record.dir, ignore_errors=True)
        return True

    def shutdown(self) -> None:
        with self._lock:
            records = list(self._runs.values())
        for record in records:
            if record.status not in TERMINAL_STATES and not getattr(record, "archived", False):
                record.cancel()


_manager: Optional[RunManager] = None


def get_run_manager() -> RunManager:
    global _manager
    if _manager is None:
        _manager = RunManager()
    return _manager
