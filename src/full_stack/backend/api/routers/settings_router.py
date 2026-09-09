"""Configuration and credentials."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from ..catalog import CatalogError, account_status, fetch_catalog
from ..config_store import (
    credential_status,
    load_config,
    patch_config,
    reset_config,
    set_credential,
)
from ..engine_bridge import effective_settings_snapshot
from ..schemas import ConfigPatch, CredentialUpdate, DashboardConfig

router = APIRouter(prefix="/settings", tags=["settings"])


def _readable(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(p) for p in error.get("loc", ()) if p != "__root__")
        parts.append(f"{location}: {error.get('msg', 'invalid value')}" if location else error.get("msg", "invalid"))
    return "; ".join(parts) or "Invalid configuration."


@router.get("")
def read_settings() -> Dict[str, Any]:
    config = load_config(refresh=True)
    return {
        "config": config.model_dump(mode="json"),
        "effective": effective_settings_snapshot(config),
        "credentials": {
            provider: credential_status(provider).model_dump()
            for provider in ("openrouter", "openai")
        },
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
    }


@router.put("")
def replace_settings(config: DashboardConfig) -> Dict[str, Any]:
    from ..config_store import save_config

    saved = save_config(config)
    return {"config": saved.model_dump(mode="json"), "effective": effective_settings_snapshot(saved)}


@router.post("/reset")
def reset() -> Dict[str, Any]:
    config = reset_config()
    return {"config": config.model_dump(mode="json"), "effective": effective_settings_snapshot(config)}


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
    return {**status, "catalog_ready": catalog_ready, "catalog_error": catalog_error}
