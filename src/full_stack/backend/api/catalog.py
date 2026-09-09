"""
Live OpenRouter model catalog, pricing, and account status.

Pricing is what makes the cost estimate real rather than a guess, so the catalog
is fetched from the provider, normalised to USD per million tokens, and cached
on disk with a short TTL so the interface stays responsive offline.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

from .config_store import get_credential, load_config
from .paths import catalog_cache_file

CATALOG_TTL_SECONDS = 60 * 30
_lock = threading.RLock()
_memory: Dict[str, Any] = {}

_EMBEDDING_HINTS = ("embed", "embedding", "text-embedding", "bge-", "gte-", "e5-")


class CatalogError(RuntimeError):
    """Raised when the provider catalog cannot be reached."""


def _headers(api_key: str = "") -> Dict[str, str]:
    config = load_config()
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if config.connection.openrouter_site_url:
        headers["HTTP-Referer"] = config.connection.openrouter_site_url
    if config.connection.openrouter_app_name:
        headers["X-Title"] = config.connection.openrouter_app_name
    return headers


def _base_url() -> str:
    return load_config().connection.openrouter_base_url.rstrip("/")


def _to_per_million(raw: Any) -> Optional[float]:
    """OpenRouter quotes USD per token as a string; humans read per million."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return round(value * 1_000_000, 6)


def _is_embedding(model_id: str, modality: str, supported: List[str]) -> bool:
    lowered = model_id.lower()
    if any(hint in lowered for hint in _EMBEDDING_HINTS):
        return True
    if "vector" in (modality or "").lower():
        return True
    return "embeddings" in {str(s).lower() for s in supported or []}


def _normalise(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    model_id = item.get("id")
    if not model_id:
        return None
    pricing = item.get("pricing") or {}
    architecture = item.get("architecture") or {}
    top_provider = item.get("top_provider") or {}
    modality = str(architecture.get("modality") or "")
    supported = item.get("supported_parameters") or []
    prompt_price = _to_per_million(pricing.get("prompt"))
    completion_price = _to_per_million(pricing.get("completion"))
    context = int(item.get("context_length") or top_provider.get("context_length") or 0) or None
    provider = str(model_id).split("/", 1)[0] if "/" in str(model_id) else "other"
    return {
        "id": model_id,
        "name": item.get("name") or model_id,
        "provider": provider,
        "description": (item.get("description") or "")[:400],
        "context_length": context,
        "max_completion_tokens": top_provider.get("max_completion_tokens"),
        "prompt_usd_per_mtok": prompt_price,
        "completion_usd_per_mtok": completion_price,
        "is_free": (prompt_price == 0 and completion_price == 0),
        "modality": modality,
        "input_modalities": architecture.get("input_modalities") or [],
        "supports_structured_output": "structured_outputs" in {str(s) for s in supported},
        "supports_tools": "tools" in {str(s) for s in supported},
        "is_embedding": _is_embedding(str(model_id), modality, supported),
        "created": item.get("created"),
    }


def _read_disk_cache() -> Optional[Dict[str, Any]]:
    path = catalog_cache_file()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return None


def _write_disk_cache(payload: Dict[str, Any]) -> None:
    try:
        catalog_cache_file().write_text(json.dumps(payload))
    except OSError:
        pass


def fetch_catalog(force: bool = False) -> Dict[str, Any]:
    """
    Return ``{"models": [...], "fetched_at": float, "stale": bool}``.

    A network failure degrades to the last good cache rather than an empty list,
    because a stale price is far more useful than no price at all.
    """
    now = time.time()
    with _lock:
        cached = _memory.get("catalog") or _read_disk_cache()
        if cached and not force and (now - float(cached.get("fetched_at") or 0)) < CATALOG_TTL_SECONDS:
            _memory["catalog"] = cached
            return {**cached, "stale": False}

    api_key = get_credential("openrouter")
    url = f"{_base_url()}/models"
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.get(url, headers=_headers(api_key))
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # network, TLS, auth, malformed body
        if cached:
            return {**cached, "stale": True, "error": str(exc)}
        raise CatalogError(f"Could not reach the model catalog: {exc}") from exc

    models = [row for row in (_normalise(item) for item in payload.get("data") or []) if row]
    models.sort(key=lambda row: (row["provider"], row["id"]))
    result = {"models": models, "fetched_at": now}
    with _lock:
        _memory["catalog"] = result
    _write_disk_cache(result)
    return {**result, "stale": False}


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    try:
        catalog = fetch_catalog()
    except CatalogError:
        return None
    for row in catalog.get("models") or []:
        if row["id"] == model_id:
            return row
    return None


def cached_catalog() -> Optional[Dict[str, Any]]:
    """Whatever is already in memory or on disk, without touching the network."""
    with _lock:
        cached = _memory.get("catalog")
    return cached or _read_disk_cache()


def pricing_index(cached_only: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Model id to catalog row.

    ``cached_only`` is for callers that must not block, such as the thread
    draining a running worker's output.
    """
    if cached_only:
        catalog = cached_catalog() or {}
    else:
        try:
            catalog = fetch_catalog()
        except CatalogError:
            return {}
    return {row["id"]: row for row in catalog.get("models") or []}


def account_status() -> Dict[str, Any]:
    """Key validity plus remaining credit, so budget warnings can be concrete."""
    api_key = get_credential("openrouter")
    if not api_key:
        return {"configured": False, "valid": False, "reason": "No OpenRouter key configured."}
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(f"{_base_url()}/key", headers=_headers(api_key))
            if response.status_code in (401, 403):
                return {"configured": True, "valid": False, "reason": "The provider rejected this key."}
            response.raise_for_status()
            data = (response.json() or {}).get("data") or {}
    except Exception as exc:
        return {"configured": True, "valid": False, "reason": f"Could not verify the key: {exc}"}

    limit = data.get("limit")
    usage = data.get("usage")
    remaining = data.get("limit_remaining")
    if remaining is None and limit is not None and usage is not None:
        try:
            remaining = float(limit) - float(usage)
        except (TypeError, ValueError):
            remaining = None
    return {
        "configured": True,
        "valid": True,
        "label": data.get("label") or "",
        "usage_usd": usage,
        "limit_usd": limit,
        "remaining_usd": remaining,
        "is_free_tier": bool(data.get("is_free_tier")),
        "rate_limit": data.get("rate_limit") or {},
    }
