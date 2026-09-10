"""Batch execution over a cohort."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ..config_store import load_config
from ..engine_bridge import ConfigurationProblem
from ..run_manager import get_run_manager
from ..schemas import BatchRequest, RunRequest

router = APIRouter(prefix="/batches", tags=["batch"])


@router.post("")
def create_batch(request: BatchRequest) -> Dict[str, Any]:
    if not request.participant_dirs:
        raise HTTPException(status_code=400, detail="Select at least one participant.")
    config = load_config(refresh=True)
    concurrency = request.concurrency or config.batch.concurrency
    continue_on_error = (
        config.batch.continue_on_error if request.continue_on_error is None else request.continue_on_error
    )

    def factory(participant_dir: str) -> RunRequest:
        return RunRequest(
            participant_dir=participant_dir,
            task=request.task,
            overrides=request.overrides,
            generate_deep_phenotype=request.generate_deep_phenotype,
            label=request.label,
        )

    try:
        batch = get_run_manager().create_batch(
            participant_dirs=request.participant_dirs,
            request_factory=factory,
            concurrency=concurrency,
            continue_on_error=continue_on_error,
            label=request.label,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConfigurationProblem as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return get_run_manager().batch_status(batch["id"]) or batch


@router.get("")
def list_batches() -> Dict[str, Any]:
    return {"batches": get_run_manager().list_batches()}


@router.get("/{batch_id}")
def batch_status(batch_id: str) -> Dict[str, Any]:
    status = get_run_manager().batch_status(batch_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"No batch {batch_id}")
    return status


@router.post("/{batch_id}/cancel")
def cancel_batch(batch_id: str) -> Dict[str, Any]:
    if not get_run_manager().cancel_batch(batch_id):
        raise HTTPException(status_code=404, detail=f"No batch {batch_id}")
    return get_run_manager().batch_status(batch_id) or {"id": batch_id, "status": "cancelled"}
