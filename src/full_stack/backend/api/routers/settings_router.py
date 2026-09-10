"""Configuration and credentials."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from ..catalog import CatalogError, account_status, fetch_catalog
from ..config_store import (
    credential_status,
    load_config,
    load_notices,
    patch_config,
    reset_config,
    set_credential,
)
from ..engine_bridge import effective_settings_snapshot, run_blockers
from ..schemas import ConfigPatch, CredentialUpdate, DashboardConfig, readable_validation_error

router = APIRouter(prefix="/settings", tags=["settings"])

CREDENTIAL_PROVIDERS = ("openrouter", "huggingface")

#: Shared with the deployment planner so a rejected local backend reads the same
#: on both screens.
_readable = readable_validation_error


@router.get("")
def read_settings() -> Dict[str, Any]:
    config = load_config(refresh=True)
    return {
        "config": config.model_dump(mode="json"),
        "effective": effective_settings_snapshot(config),
        "credentials": {
            provider: credential_status(provider).model_dump()
            for provider in CREDENTIAL_PROVIDERS
        },
        # Anything the loader had to repair, and anything that would stop a run
        # right now, so the interface can say so before the user starts one.
        "notices": load_notices(),
        "blockers": run_blockers(config),
    }


@router.patch("")
def update_settings(patch: ConfigPatch) -> Dict[str, Any]:
    # Sections are free-form dicts at the request boundary, so the real
    # constraints only fire when the merged config is validated.
    try:
        config = patch_config(patch)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_readable(exc)) from exc
    return {
        "config": config.model_dump(mode="json"),
        "effective": effective_settings_snapshot(config),
        "blockers": run_blockers(config),
    }


@router.put("")
def replace_settings(config: DashboardConfig) -> Dict[str, Any]:
    from ..config_store import save_config

    saved = save_config(config)
    return {
        "config": saved.model_dump(mode="json"),
        "effective": effective_settings_snapshot(saved),
        "blockers": run_blockers(saved),
    }


@router.post("/reset")
def reset() -> Dict[str, Any]:
    config = reset_config()
    return {
        "config": config.model_dump(mode="json"),
        "effective": effective_settings_snapshot(config),
        "blockers": run_blockers(config),
    }


@router.put("/credentials")
def put_credential(update: CredentialUpdate) -> Dict[str, Any]:
    status = set_credential(update.provider, update.api_key)
    verification = account_status() if update.provider == "openrouter" else {"configured": status.configured}
    return {"credential": status.model_dump(), "verification": verification}


@router.get("/credentials/verify")
def verify_credential() -> Dict[str, Any]:
    """Check the key against the provider without spending anything."""
    status = account_status()
    catalog_ready = True
    catalog_error = ""
    try:
        catalog = fetch_catalog()
        catalog_ready = bool(catalog.get("models"))
    except CatalogError as exc:
        catalog_ready = False
        catalog_error = str(exc)
    # Readiness is a key that works and a catalog that answers. Remaining credit
    # is reported for the cost guardrails and never decides this.
    ready = bool(status.get("valid")) and catalog_ready
    return {**status, "ready": ready, "catalog_ready": catalog_ready, "catalog_error": catalog_error}
