"""
COMPASS runtime event bus.

The engine emits progress through a single process-local emitter obtained with
``get_ui()``. The emitter keeps a reducer-backed :class:`EventStore` snapshot and
forwards every event to any attached sink, which is how the dashboard service
streams a live run to the browser.

This module has no web-framework dependency on purpose: the engine imports it,
the API service consumes it.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Stage indices used by every agent when it reports progress.
STAGE_NAMES: List[str] = [
    "Initialization",
    "Orchestration",
    "Execution",
    "Integration",
    "Prediction",
    "Evaluation",
    "Communication",
]


def json_safe(obj: Any) -> Any:
    """Coerce engine objects (pydantic models, enums, paths) into plain JSON."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return json_safe(fn())
            except Exception:
                continue
    if hasattr(obj, "__dict__"):
        try:
            return json_safe(vars(obj))
        except Exception:
            return str(obj)
    return str(obj)


class EventStore:
    """Thread-safe storage for pipeline events with optimized state tracking."""
    def __init__(self):
        self._lock = threading.Lock()
        self._sinks: List[Callable[[Dict[str, Any]], None]] = []
        self.reset()  # init msg

    def add_sink(self, sink: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback that receives every event as plain JSON."""
        with self._lock:
            if sink not in self._sinks:
                self._sinks.append(sink)

    def remove_sink(self, sink: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def reset(self):
        self.events = []
        self.state = {
            "participant_id": "Unknown",
            "participant_dir": None,
            "target": "None",
            "control": None,
            "prediction_spec": None,
            "status": "Ready to Launch",
            "start_time": None,
            "total_tokens": 0,
            "progress": 0,
            "max_steps": 1, 
            "steps": [],
            "history": [], # Archive of steps from previous iterations
            "prediction": None,
            "critic": None,
            "critic_summary": None,
            "completed": False,
            "completion": None,
            "latest_update_id": 0,
            "current_stage": -1, # -1: Setup, 0:Init, 1:Plan, 2:Exec, 3:Predict, 4:Evaluate
            "stages": list(STAGE_NAMES),
            "iteration": 1,
            "deep_report_status": "idle",
            "deep_report_available": False,
            "deep_report_error": None,
            "deep_report_last_generated_at": None,
        }

    def add_event(self, event_type, data):
        data = json_safe(data)
        with self._lock:
            now = datetime.now()
            timestamp = now.strftime("%H:%M:%S")
            # Enriched event data
            event = {
                "id": self.state["latest_update_id"] + 1,
                "time": timestamp,
                "ts": now.isoformat(timespec="milliseconds"),
                "type": event_type,
                "data": data,
            }
            self.events.append(event)
            self.state["latest_update_id"] = event["id"]
            
            # --- State Reducer Logic ---
            if event_type == "STATUS":
                self.state["status"] = data["message"]
                if "stage" in data:
                    self.state["current_stage"] = data["stage"]
                if "iteration" in data:
                     self.state["iteration"] = data["iteration"]

            elif event_type == "INIT":
                self.state["participant_id"] = data["participant_id"]
                self.state["target"] = data["target"]
                self.state["control"] = data.get("control")
                self.state["prediction_spec"] = data.get("prediction_spec")
                self.state["max_iterations"] = data.get("max_iterations", 1)
                self.state["config"] = data.get("config", {}) # Store token config
                self.state["start_time"] = timestamp
                self.state["status"] = "Initializing Engine..."
                self.state["current_stage"] = 0
                self.state["steps"] = [] 
                self.state["history"] = []
                self.state["plans"] = {} # Store plans by iteration
                self.state["prediction"] = None
                self.state["critic"] = None
                self.state["critic_summary"] = None
                self.state["total_tokens"] = 0
                self.state["progress"] = 0
                self.state["completed"] = False
                self.state["completion"] = None
                self.state["deep_report_status"] = "idle"
                self.state["deep_report_available"] = False
                self.state["deep_report_error"] = None
                self.state["deep_report_last_generated_at"] = None
                
            elif event_type == "PLAN":
                current_iter = self.state.get("iteration", 1)
                
                # Store full plan
                if "plans" not in self.state: self.state["plans"] = {}
                self.state["plans"][str(current_iter)] = data.get("plan", {})
                
                # Legacy support for progress bar
                self.state["max_steps"] = data.get("steps", 10)
                self.state["status"] = f"Orchestrating Plan (Iteration {current_iter})..."
                self.state["current_stage"] = 1
                
                # Archive previous steps steps to history
                if self.state["steps"]:
                    # Mark them as historical if needed, or distinct
                    self.state.setdefault("history", []).extend(self.state["steps"])
                
                self.state["steps"] = [] # Clear for new plan steps
                # Clear previous results to prevent stale modal
                self.state["prediction"] = None
                self.state["critic_summary"] = None
                self.state["critic"] = None
                self.state["progress"] = 0
                
            elif event_type == "STEP_START":
                existing = next((s for s in self.state["steps"] if s["id"] == data["id"]), None)
                if not existing:
                    self.state["steps"].append({
                        "id": data["id"],
                        "tool": data["tool"],
                        "desc": data["desc"],
                        "status": "running",
                        "tokens": 0,
                        "startTime": time.time(),
                        "duration": 0,
                        "iteration": self.state.get("iteration", 1)
                    })
                else:
                    # Allow repeated STEP_START for the same step id to refresh live descriptions
                    # (used for dynamic fusion/chunking progress without creating extra timeline rows).
                    existing["tool"] = data.get("tool", existing.get("tool"))
                    existing["desc"] = data.get("desc", existing.get("desc"))
                    existing["status"] = "running"
                if "stage" in data and data["stage"] is not None:
                    self.state["current_stage"] = data["stage"]
                else:
                    # Fallback inference for backward compatibility.
                    step_id = int(data.get("id") or 0)
                    if step_id >= 930:
                        self.state["current_stage"] = 6
                    elif step_id >= 920:
                        self.state["current_stage"] = 5
                    elif step_id >= 910:
                        self.state["current_stage"] = 4
                    elif step_id >= 900:
                        self.state["current_stage"] = 3
                    else:
                        self.state["current_stage"] = 2
                self.state["status"] = f"Running Step {data['id']}: {data['tool']}"
                
            elif event_type == "STEP_COMPLETE":
                for s in self.state["steps"]:
                    if s["id"] == data["id"]:
                        s["status"] = "complete"
                        s["tokens"] = data["tokens"]
                        s["preview"] = data.get("preview", "")
                        if "startTime" in s:
                            s["duration"] = round(time.time() - s["startTime"], 2)
                self.state["total_tokens"] += data["tokens"]
                self.state["progress"] += 1
                self.state["status"] = "Step Complete"
                
            elif event_type == "STEP_FAIL":
                existing = next((s for s in self.state["steps"] if s["id"] == data["id"]), None)
                if existing:
                    existing["status"] = "failed"
                    existing["error"] = data["error"]
                else:
                    # A step whose dependencies never finished is failed without
                    # ever having started, so there is no row to update. Without
                    # one the failure never reached the screen, and the run read
                    # as though those steps had not been planned at all.
                    self.state["steps"].append({
                        "id": data["id"],
                        "tool": data.get("tool", ""),
                        "desc": data.get("desc", ""),
                        "status": "failed",
                        "error": data["error"],
                        "tokens": 0,
                        "startTime": time.time(),
                        "duration": 0,
                        "iteration": self.state.get("iteration", 1),
                    })
                self.state["status"] = "Step Failed"
            
            elif event_type == "REPAIR":
                 for s in self.state["steps"]:
                    if s["id"] == data["id"]:
                        s["status"] = "repairing"
                        s["msg"] = data["strategy"]
                 self.state["status"] = "Attempting Auto-Repair"

            elif event_type == "FUSION":
                self.state["fusion_data"] = data
                self.state["status"] = "Fusion Complete"
                self.state["current_stage"] = 3
                # Step is now handled via explicit STEP_START/COMPLETE in executor

            elif event_type == "PREDICTION":
                self.state["prediction"] = data
                self.state["status"] = "Prediction Generated"
                self.state["current_stage"] = 4 
                # Add or update virtual step for Predictor
                pred_id = 910 + self.state.get("iteration", 1)
                existing = next((s for s in self.state["steps"] if s["id"] == pred_id), None)
                prob_val = data.get("prob")
                prob_text = f"{float(prob_val):.1%}" if isinstance(prob_val, (int, float)) else "N/A"
                if existing:
                    existing["status"] = "complete"
                    existing["tokens"] = 0
                    existing["desc"] = f"Generated prediction: {data.get('result', 'Unknown')} ({prob_text})"
                    if "startTime" in existing:
                        existing["duration"] = round(time.time() - existing["startTime"], 2)
                else:
                    self.state["steps"].append({
                        "id": pred_id,
                        "tool": "Predictor Agent",
                        "desc": f"Generated prediction: {data.get('result', 'Unknown')} ({prob_text})",
                        "status": "complete",
                        "tokens": 0,
                        "startTime": time.time(),
                        "duration": 0.5,
                        "iteration": self.state.get("iteration", 1)
                    })

            elif event_type == "CRITIC":
                self.state["status"] = f"Critic Verdict: {data['verdict']}"
                self.state["critic_summary"] = data.get("summary", "") 
                self.state["critic"] = data
                self.state["current_stage"] = 5
                
                # Determine status based on verdict
                verdict = data.get('verdict', 'UNKNOWN')
                is_pass = verdict == 'SATISFACTORY'
                step_status = "complete" if is_pass else "failed"
                
                # Add virtual step for Critic
                self.state["steps"].append({
                    "id": 920 + self.state["iteration"],
                    "tool": "Critic Agent",
                    "desc": f"Verdict: {verdict}",
                    "preview": data.get("summary", "No details provided."),
                    "status": step_status, 
                    "tokens": 0,
                    "startTime": time.time(),
                    "duration": 0.5,
                    "iteration": self.state.get("iteration", 1)
                })

            elif event_type == "COMPLETE":
                # Ensure we don't duplicate logic, just set status
                self.state["status"] = "Pipeline Completed"
                # What finished, not the size of the plan. Snapping progress to
                # max_steps is what made a run that lost three steps report
                # "20 of 20 steps".
                finished = sum(1 for s in self.state["steps"] if s.get("status") == "complete")
                self.state["progress"] = max(self.state.get("progress", 0), finished)
                # Always snap to final stage in case new stages are added (e.g., Communication)
                self.state["current_stage"] = max(0, len(self.state.get("stages", [])) - 1)
                self.state["completed"] = True
                self.state["completion"] = data
            elif event_type == "DEEP_REPORT":
                status = str(data.get("status") or "").lower().strip()
                if status:
                    self.state["deep_report_status"] = status
                if "available" in data:
                    self.state["deep_report_available"] = bool(data.get("available"))
                if "error" in data:
                    self.state["deep_report_error"] = data.get("error")
                if status == "completed":
                    self.state["deep_report_last_generated_at"] = datetime.now().isoformat()

            sinks = list(self._sinks)
            snapshot_state = json_safe(self.state)

        for sink in sinks:
            try:
                sink({"event": event, "state": snapshot_state})
            except Exception:
                # A broken sink must never take down a running pipeline.
                pass

    def get_snapshot(self, since_id=0):
        with self._lock:
            new_events = [e for e in self.events if e["id"] > int(since_id)]
            return {
                "state": json_safe(self.state),
                "events": json_safe(new_events),
            }


# --- Process-local singletons -------------------------------------------------

_event_store = EventStore()
_ui_instance: Optional["RunEventEmitter"] = None


def get_event_store() -> EventStore:
    """The process-local event store the engine writes into."""
    return _event_store


def attach_sink(sink: Callable[[Dict[str, Any]], None]) -> Callable[[Dict[str, Any]], None]:
    """Stream every subsequent engine event to ``sink``."""
    _event_store.add_sink(sink)
    return sink


def detach_sink(sink: Callable[[Dict[str, Any]], None]) -> None:
    _event_store.remove_sink(sink)


class RunEventEmitter:
    """
    The progress interface the engine holds.

    Every agent calls these methods; the emitter turns them into reducer events.
    ``enabled`` gates the emission so a plain CLI run stays silent.
    """

    def __init__(self) -> None:
        self.enabled = False

    # -- lifecycle ------------------------------------------------------------

    def set_status(self, message, stage=None, iteration=None):
        data: Dict[str, Any] = {"message": message}
        if stage is not None:
            data["stage"] = stage
        if iteration is not None:
            data["iteration"] = iteration
        _event_store.add_event("STATUS", data)

    def on_pipeline_start(
        self,
        participant_id,
        target,
        control=None,
        prediction_spec=None,
        participant_dir=None,
        max_iterations=3,
        token_config=None,
    ):
        _event_store.add_event(
            "INIT",
            {
                "participant_id": participant_id,
                "target": target,
                "control": control,
                "prediction_spec": prediction_spec,
                "max_iterations": max_iterations,
                "config": token_config or {},
            },
        )
        if participant_dir:
            _event_store.state["participant_dir"] = str(participant_dir)

    def on_plan_created(self, plan):
        if hasattr(plan, "total_steps"):
            steps = plan.total_steps
            domains = plan.priority_domains
            plan_dict = json_safe(plan)
        else:
            steps = plan.get("total_steps", 0)
            domains = plan.get("priority_domains", [])
            plan_dict = plan

        _event_store.add_event("PLAN", {"steps": steps, "domains": domains, "plan": plan_dict})

    # -- plan execution -------------------------------------------------------

    def on_step_start(self, step_id, tool_name, description, parallel_with=None, stage=None):
        payload: Dict[str, Any] = {"id": step_id, "tool": tool_name, "desc": description}
        if parallel_with:
            payload["parallel_with"] = list(parallel_with)
        if stage is not None:
            payload["stage"] = stage
        _event_store.add_event("STEP_START", payload)

    def on_step_complete(self, step_id, tokens, duration_ms, preview=""):
        _event_store.add_event(
            "STEP_COMPLETE",
            {"id": step_id, "tokens": tokens, "duration_ms": duration_ms, "preview": preview},
        )

    def on_step_failed(self, step_id, error, tool_name=None, description=None):
        payload: Dict[str, Any] = {"id": step_id, "error": str(error)}
        if tool_name:
            payload["tool"] = tool_name
        if description:
            payload["desc"] = description
        _event_store.add_event("STEP_FAIL", payload)

    def on_auto_repair(self, step_id, strategy):
        _event_store.add_event("REPAIR", {"id": step_id, "strategy": strategy})

    def on_fusion_complete(self, fusion_data):
        _event_store.add_event("FUSION", fusion_data)

    # -- results --------------------------------------------------------------

    def on_prediction(self, classification, probability, confidence, prediction_payload=None):
        result_text = str(classification or "").strip() or "PREDICTION_READY"
        _event_store.add_event(
            "PREDICTION",
            {
                "result": result_text,
                "label": result_text,
                "prob": probability,
                "confidence": confidence,
                "payload": prediction_payload,
            },
        )
        if isinstance(probability, (int, float)):
            self.set_status(f"Predictor Assessment: {result_text} ({float(probability):.1%})", stage=4)
        else:
            self.set_status(f"Predictor Assessment: {result_text}", stage=4)

    def on_critic_verdict(
        self,
        verdict,
        confidence,
        checklist_passed,
        checklist_total,
        summary="",
        checklist=None,
        weaknesses=None,
        improvement_suggestions=None,
        domains_missed=None,
        composite_score=None,
        score_breakdown=None,
        iteration=None,
        fallback_used=False,
        fallback_reason=None,
        fallback_recommendation=None,
    ):
        _event_store.add_event(
            "CRITIC",
            {
                "verdict": verdict,
                "confidence": confidence,
                "passed": checklist_passed,
                "total": checklist_total,
                "summary": summary,
                "checklist": checklist or {},
                "weaknesses": weaknesses or [],
                "improvement_suggestions": improvement_suggestions or [],
                "domains_missed": domains_missed or [],
                "composite_score": composite_score,
                "score_breakdown": score_breakdown or {},
                "iteration": iteration,
                "fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason or "",
                "fallback_recommendation": fallback_recommendation or "",
            },
        )
        self.set_status(f"Critic Evaluation: {verdict}", stage=5)

    def on_pipeline_complete(
        self, result, probability, iterations, total_duration_secs, total_tokens, prediction_payload=None
    ):
        _event_store.add_event(
            "COMPLETE",
            {
                "result": result,
                "probability": probability,
                "iterations": iterations,
                "duration": total_duration_secs,
                "tokens": total_tokens,
                "prediction_payload": prediction_payload,
            },
        )

    # -- reporting ------------------------------------------------------------

    def on_deep_report(self, status, available=None, error=None, path=None):
        payload: Dict[str, Any] = {"status": status}
        if available is not None:
            payload["available"] = bool(available)
        if error is not None:
            payload["error"] = str(error)
        if path is not None:
            payload["path"] = str(path)
        _event_store.add_event("DEEP_REPORT", payload)

    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Escape hatch for events the service layer adds around a run."""
        _event_store.add_event(event_type, data)


def get_ui(enabled: bool = False) -> RunEventEmitter:
    """
    The process-local emitter.

    Emission stays off until something asks for it. Engine internals call this
    with no argument to read `enabled`, so defaulting to True here would turn a
    plain CLI run into an event producer.
    """
    global _ui_instance
    if _ui_instance is None:
        _ui_instance = RunEventEmitter()
    if enabled:
        _ui_instance.enabled = True
    return _ui_instance


def reset_ui() -> None:
    global _ui_instance
    _ui_instance = None
    _event_store.reset()
