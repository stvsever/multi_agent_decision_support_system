"""Hosting recipes for the open-weights backend, and what this machine can run."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException
from pydantic import ValidationError

from ..config_store import get_credential, load_config
from ..deploy import plan_deployment, probe_machine
from ..hf_catalog import model_detail
from ..schemas import LocalBackendConfig, readable_validation_error

router = APIRouter(prefix="/deploy", tags=["deploy"])


def _plan(local: LocalBackendConfig, lookup: bool) -> Dict[str, Any]:
    detail: Dict[str, Any] = {}
    if lookup and local.model_name.strip():
        # A repository lookup is the only way to size the model honestly, but it
        # is a network call, so it runs on a short timeout and a failure
        # downgrades the estimate rather than the whole plan.
        try:
            detail = model_detail(local.model_name, timeout=5.0)
        except Exception:
            detail = {}
    machine = probe_machine()
    plan = plan_deployment(
        local,
        model_detail=detail or None,
        hf_token=bool(get_credential("huggingface")),
        probe=machine,
    )
    return {**plan, "machine": machine}


@router.get("/plan")
def saved_plan(lookup: bool = True) -> Dict[str, Any]:
    """The recipe for the configuration currently saved."""
    return _plan(load_config(refresh=True).local, lookup)


@router.post("/plan")
def preview_plan(local: Dict[str, Any] = Body(...), lookup: bool = True) -> Dict[str, Any]:
    """
    The recipe for a configuration the user is still editing.

    The body is taken raw and validated here rather than in the signature, so a
    rejected setting comes back as the one sentence the constraint was written
    in instead of pydantic's own error list.
    """
    try:
        config = LocalBackendConfig.model_validate(local)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=readable_validation_error(exc)) from exc
    return _plan(config, lookup)


@router.get("/probe")
def probe() -> Dict[str, Any]:
    """What this machine actually has: accelerators, runtimes, and a scheduler."""
    return probe_machine()
