"""
HuggingFace model lookup for the local inference backend.

When the backend is Local the model is a HuggingFace repository id rather than a
provider route, so the dashboard needs a different catalog: search by name, and
resolve a repository's real context window from its published config.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

HF_API = "https://huggingface.co/api/models"
HF_RAW = "https://huggingface.co/{model_id}/raw/main/{filename}"

#: Keys that carry a context window, most specific first.
_CONTEXT_KEYS = ("model_max_length", "max_position_embeddings", "max_sequence_length", "n_positions")


def _get_json(url: str, timeout: float = 15.0) -> Any:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        return response.json()


def _sane_context(value: Any) -> Optional[int]:
    """HF configs sometimes carry sentinel values such as 1e30 for "unbounded"."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0 or number > 10_000_000:
        return None
    return number


def search_models(query: str = "", task: str = "", limit: int = 60) -> Dict[str, Any]:
    params = [f"limit={max(1, min(limit, 200))}", "sort=downloads", "direction=-1"]
    if task.strip().lower() == "embedding":
        params.append("pipeline_tag=feature-extraction")
    if query.strip():
        params.append(f"search={quote(query.strip())}")
    url = f"{HF_API}?{'&'.join(params)}"

    try:
        payload = _get_json(url)
    except Exception as exc:
        return {"models": [], "error": f"Could not reach HuggingFace: {exc}"}

    rows: List[Dict[str, Any]] = []
    for item in payload if isinstance(payload, list) else []:
        model_id = item.get("modelId") or item.get("id")
        if not model_id:
            continue
        rows.append(
            {
                "id": model_id,
                "downloads": item.get("downloads") or 0,
                "likes": item.get("likes") or 0,
                "pipeline_tag": item.get("pipeline_tag") or "",
                "tags": [t for t in (item.get("tags") or []) if isinstance(t, str)][:12],
                "gated": bool(item.get("gated")),
                "is_embedding": (item.get("pipeline_tag") or "") == "feature-extraction",
            }
        )
    return {"models": rows}


def model_detail(model_id: str) -> Dict[str, Any]:
    """
    Resolve a repository's context window.

    ``tokenizer_config.json`` wins over ``config.json``: a tokenizer's
    ``model_max_length`` reflects the length the model was actually trained to
    serve, which is often shorter than the architectural maximum.
    """
    detail: Dict[str, Any] = {"id": model_id, "context_length": None, "is_embedding": False}

    try:
        info = _get_json(f"{HF_API}/{model_id}")
        detail["pipeline_tag"] = info.get("pipeline_tag") or ""
        detail["downloads"] = info.get("downloads") or 0
        detail["likes"] = info.get("likes") or 0
        detail["gated"] = bool(info.get("gated"))
        detail["is_embedding"] = detail["pipeline_tag"] == "feature-extraction"
    except Exception as exc:
        detail["error"] = f"Could not read the repository: {exc}"

    architectural: Optional[int] = None
    try:
        config = _get_json(HF_RAW.format(model_id=model_id, filename="config.json"))
        if isinstance(config, dict):
            for key in ("max_position_embeddings", "max_sequence_length", "n_positions"):
                architectural = _sane_context(config.get(key))
                if architectural:
                    break
            detail["architecture"] = config.get("architectures") or []
            detail["model_type"] = config.get("model_type") or ""
    except Exception:
        architectural = None

    served: Optional[int] = None
    try:
        tokenizer = _get_json(HF_RAW.format(model_id=model_id, filename="tokenizer_config.json"))
        if isinstance(tokenizer, dict):
            served = _sane_context(tokenizer.get("model_max_length"))
    except Exception:
        served = None

    detail["context_length"] = served or architectural
    detail["architectural_context_length"] = architectural
    return detail
