"""Run lifecycle, live streaming, and cost."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ..config_store import load_config
from ..cost import estimate_run, expected_plan_steps, participant_domain_count, participant_input_tokens
from ..datasets import inspect_participant
from ..engine_bridge import merge_overrides
from ..run_manager import get_run_manager
from ..safe_paths import to_path
from ..schemas import AuditRequest, CostEstimateRequest, RunRequest

router = APIRouter(prefix="/runs", tags=["runs"])

HEARTBEAT_SECONDS = 15.0
TERMINAL = {"succeeded", "failed", "cancelled"}


def _readable(exc: ValidationError) -> str:
    """A validation failure the caller can act on, not a stack trace."""
    parts = []
    for error in exc.errors():
        location = ".".join(str(p) for p in error.get("loc", ()) if p != "__root__")
        parts.append(f"{location}: {error.get('msg', 'invalid value')}" if location else error.get("msg", "invalid"))
    return "; ".join(parts) or "Invalid configuration."


@router.post("/estimate")
def estimate(request: CostEstimateRequest) -> Dict[str, Any]:
    """Projected tokens and spend before anything is sent to a provider."""
    try:
        config = merge_overrides(load_config(refresh=True), request.overrides)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_readable(exc)) from exc
    per_participant: List[Dict[str, Any]] = []
    total_usd = 0.0
    total_tokens = 0
    priced = True

    for raw in request.participant_dirs or []:
        directory = to_path(raw) or Path(str(raw))
        tokens = participant_input_tokens(directory) if directory.is_dir() else 0
        line = estimate_run(
            config=config,
            input_tokens=tokens,
            plan_steps=expected_plan_steps(participant_domain_count(directory)),
            include_deep_report=request.generate_deep_phenotype,
        )
        per_participant.append({"participant_dir": str(directory), "id": directory.name, **line})
        total_tokens += line["total_tokens"]
        if line["usd"] is None:
            priced = False
        else:
            total_usd += line["usd"]

    warn = config.cost.warn_above_usd
    block = config.cost.block_above_usd
    return {
        "participants": per_participant,
        "totals": {
            "count": len(per_participant),
            "total_tokens": total_tokens,
            "usd": round(total_usd, 6) if priced else None,
            "usd_low": round(total_usd * 0.65, 6) if priced else None,
            "usd_high": round(total_usd * 1.35, 6) if priced else None,
            "fully_priced": priced,
        },
        "guards": {
            "warn_above_usd": warn,
            "block_above_usd": block,
            "warns": bool(priced and warn and total_usd > warn),
            "blocks": bool(priced and block and total_usd > block),
        },
        "effective_models": {
            role: (str(getattr(config.models.role_models, role, "") or "").strip() or config.models.default_model)
            for role in ("orchestrator", "integrator", "predictor", "critic", "communicator", "tool")
        },
    }


@router.post("")
def create_run(request: RunRequest) -> Dict[str, Any]:
    try:
        record = get_run_manager().create_run(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_readable(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record.summary()


@router.post("/audit")
def create_audit(request: AuditRequest) -> Dict[str, Any]:
    """Offline structural check: loads the data and builds the payload, no provider calls."""
    run_request = RunRequest(
        participant_dir=request.participant_dir,
        task=request.task,
        generate_deep_phenotype=False,
        label=f"Audit {Path(request.participant_dir).name}",
    )
    try:
        record = get_run_manager().create_run(run_request, audit=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_readable(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record.summary()


@router.get("")
def list_runs(limit: int = Query(100, ge=1, le=500), batch_id: Optional[str] = None) -> Dict[str, Any]:
    return {"runs": get_run_manager().list_runs(limit=limit, batch_id=batch_id)}


@router.get("/{run_id}")
def get_run(run_id: str, since_event: int = Query(0, ge=0)) -> Dict[str, Any]:
    record = get_run_manager().get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No run {run_id}")
    return record.detail(since_event=since_event)


@router.delete("/{run_id}")
def delete_run(run_id: str) -> Dict[str, Any]:
    if not get_run_manager().delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"No run {run_id}")
    return {"deleted": run_id}


@router.post("/{run_id}/cancel")
def cancel_run(run_id: str) -> Dict[str, Any]:
    record = get_run_manager().get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No run {run_id}")
    record.cancel()
    return record.summary()


@router.get("/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    """Server-sent events for one run: state, events, usage, and completion."""
    record = get_run_manager().get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No run {run_id}")

    async def publish() -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        # An asyncio queue fed through call_soon_threadsafe: a blocking get in
        # the default executor would pin one pool thread per open stream, and
        # the pool is small enough that a batch of streams would starve.
        inbox: asyncio.Queue = asyncio.Queue(maxsize=2048)

        def forward(message: Dict[str, Any]) -> None:
            try:
                loop.call_soon_threadsafe(inbox.put_nowait, message)
            except (RuntimeError, asyncio.QueueFull):
                # The loop is closing, or a stalled client filled its queue.
                pass

        channel = record.subscribe(forward)
        try:
            # Building the snapshot can copy thousands of events, so keep it off
            # the event loop thread.
            snapshot = await asyncio.to_thread(lambda: {"type": "snapshot", **record.detail()})
            yield f"data: {json.dumps(snapshot, default=str)}\n\n".encode()

            while True:
                try:
                    message = await asyncio.wait_for(inbox.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield b": heartbeat\n\n"
                    if record.status in TERMINAL:
                        break
                    continue
                yield f"data: {json.dumps(message, default=str)}\n\n".encode()
                if message.get("type") == "done":
                    break
        except asyncio.CancelledError:
            raise
        finally:
            record.unsubscribe(channel)

    return StreamingResponse(
        publish(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
