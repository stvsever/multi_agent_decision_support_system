"""Model catalog and pricing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..catalog import CatalogError, account_status, fetch_catalog, get_model

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/models")
def models(
    refresh: bool = Query(False, description="Bypass the cache"),
    search: str = Query("", description="Match on id, name, or provider"),
    provider: str = Query("", description="Restrict to one provider slug"),
    embedding: Optional[bool] = Query(None, description="Only embedding or only chat models"),
    free_only: bool = Query(False),
    min_context: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=2000),
) -> Dict[str, Any]:
    try:
        catalog = fetch_catalog(force=refresh)
    except CatalogError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    rows: List[Dict[str, Any]] = catalog.get("models") or []
    needle = search.strip().lower()
    if needle:
        rows = [r for r in rows if needle in r["id"].lower() or needle in (r["name"] or "").lower()]
    if provider:
        rows = [r for r in rows if r["provider"] == provider]
    if embedding is not None:
        rows = [r for r in rows if bool(r["is_embedding"]) is embedding]
    if free_only:
        rows = [r for r in rows if r["is_free"]]
    if min_context:
        rows = [r for r in rows if (r["context_length"] or 0) >= min_context]

    providers = sorted({r["provider"] for r in (catalog.get("models") or [])})
    return {
        "models": rows[:limit],
        "total": len(rows),
        "providers": providers,
        "fetched_at": catalog.get("fetched_at"),
        "stale": catalog.get("stale", False),
        "error": catalog.get("error"),
    }


@router.get("/models/{model_id:path}")
def model_detail(model_id: str) -> Dict[str, Any]:
    row = get_model(model_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the catalog.")
    return row


@router.get("/account")
def account() -> Dict[str, Any]:
    return account_status()
