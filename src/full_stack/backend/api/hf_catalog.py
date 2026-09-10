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

#: Bytes one stored weight occupies, by the dtype a repository declares.
_BYTES_PER_WEIGHT = {
    "float32": 4.0, "f32": 4.0, "fp32": 4.0,
    "bfloat16": 2.0, "bf16": 2.0, "float16": 2.0, "f16": 2.0, "fp16": 2.0, "half": 2.0,
    "int8": 1.0, "i8": 1.0, "fp8": 1.0, "float8": 1.0, "f8_e4m3": 1.0, "f8_e5m2": 1.0,
    "int4": 0.5, "uint4": 0.5, "u4": 0.5, "i4": 0.5,
}


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


def _license_of(info: Dict[str, Any]) -> str:
    card = info.get("cardData") or {}
    declared = card.get("license")
    if isinstance(declared, list):
        declared = declared[0] if declared else ""
    if declared:
        return str(declared)
    for tag in info.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return ""


def _parameters_from_info(info: Dict[str, Any]) -> Optional[int]:
    """The published weight count, which most repositories carry already."""
    safetensors = info.get("safetensors") or {}
    total = safetensors.get("total")
    try:
        value = int(total)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _parameters_from_index(model_id: str, dtype: str, timeout: float = 10.0) -> Optional[int]:
    """Weight count implied by the size of the sharded checkpoint."""
    try:
        index = _get_json(HF_RAW.format(model_id=model_id, filename="model.safetensors.index.json"), timeout=timeout)
    except Exception:
        return None
    if not isinstance(index, dict):
        return None
    try:
        total_bytes = float((index.get("metadata") or {}).get("total_size") or 0)
    except (TypeError, ValueError):
        return None
    per_weight = _BYTES_PER_WEIGHT.get(str(dtype or "").lower(), 2.0)
    estimated = int(total_bytes / per_weight)
    return estimated if estimated > 0 else None


def _parameters_from_config(config: Dict[str, Any]) -> Optional[int]:
    """
    A dimensional estimate, for repositories that publish neither total.

    Counts the embedding matrices plus a transformer block's projections; it
    ignores biases and norms, so it lands within a few percent of the truth for
    a standard decoder and is only ever used as a last resort.
    """
    try:
        hidden = int(config.get("hidden_size") or 0)
        layers = int(config.get("num_hidden_layers") or 0)
        vocab = int(config.get("vocab_size") or 0)
    except (TypeError, ValueError):
        return None
    if not hidden or not layers:
        return None
    intermediate = int(config.get("intermediate_size") or hidden * 4)
    heads = int(config.get("num_attention_heads") or 1) or 1
    kv_heads = int(config.get("num_key_value_heads") or heads) or heads
    head_dim = hidden // heads if heads else hidden
    attention = hidden * hidden + 2 * hidden * (kv_heads * head_dim) + hidden * hidden
    mlp = 3 * hidden * intermediate
    return int(2 * vocab * hidden + layers * (attention + mlp))


def model_detail(model_id: str, timeout: float = 15.0) -> Dict[str, Any]:
    """
    Resolve a repository's context window, size, and terms of use.

    ``tokenizer_config.json`` wins over ``config.json``: a tokenizer's
    ``model_max_length`` reflects the length the model was actually trained to
    serve, which is often shorter than the architectural maximum.
    """
    detail: Dict[str, Any] = {"id": model_id, "context_length": None, "is_embedding": False}

    info: Dict[str, Any] = {}
    try:
        loaded = _get_json(f"{HF_API}/{model_id}", timeout=timeout)
        info = loaded if isinstance(loaded, dict) else {}
        detail["pipeline_tag"] = info.get("pipeline_tag") or ""
        detail["downloads"] = info.get("downloads") or 0
        detail["likes"] = info.get("likes") or 0
        detail["gated"] = bool(info.get("gated"))
        detail["license"] = _license_of(info)
        detail["library_name"] = info.get("library_name") or ""
        detail["is_embedding"] = detail["pipeline_tag"] == "feature-extraction"
    except Exception as exc:
        detail["error"] = f"Could not read the repository: {exc}"
        detail.setdefault("gated", False)
        detail.setdefault("license", "")

    architectural: Optional[int] = None
    config: Dict[str, Any] = {}
    try:
        loaded_config = _get_json(HF_RAW.format(model_id=model_id, filename="config.json"), timeout=timeout)
        if isinstance(loaded_config, dict):
            config = loaded_config
            for key in ("max_position_embeddings", "max_sequence_length", "n_positions"):
                architectural = _sane_context(config.get(key))
                if architectural:
                    break
            detail["architecture"] = config.get("architectures") or []
            detail["model_type"] = config.get("model_type") or ""
            detail["torch_dtype"] = str(config.get("torch_dtype") or "")
            quant = config.get("quantization_config") or {}
            detail["quantization_method"] = str(quant.get("quant_method") or "") if isinstance(quant, dict) else ""
    except Exception:
        architectural = None

    served: Optional[int] = None
    try:
        tokenizer = _get_json(HF_RAW.format(model_id=model_id, filename="tokenizer_config.json"), timeout=timeout)
        if isinstance(tokenizer, dict):
            served = _sane_context(tokenizer.get("model_max_length"))
    except Exception:
        served = None

    detail["context_length"] = served or architectural
    detail["architectural_context_length"] = architectural

    parameters = _parameters_from_info(info)
    source = "model_info" if parameters else ""
    if not parameters:
        parameters = _parameters_from_index(model_id, detail.get("torch_dtype") or "", timeout=timeout)
        source = "safetensors_index" if parameters else ""
    if not parameters and config:
        parameters = _parameters_from_config(config)
        source = "config_estimate" if parameters else ""
    detail["parameter_count"] = parameters
    detail["parameter_count_source"] = source
    detail["parameter_count_is_estimate"] = source in ("safetensors_index", "config_estimate")
    return detail
