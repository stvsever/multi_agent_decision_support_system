"""HuggingFace lookup, used when the backend is Local."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Query

from ..hf_catalog import model_detail, search_models

router = APIRouter(prefix="/hf", tags=["catalog"])


@router.get("/models")
def models(
    q: str = Query("", description="Search text"),
    task: str = Query("", description='Set to "embedding" to restrict to feature extraction'),
    limit: int = Query(60, ge=1, le=200),
) -> Dict[str, Any]:
    return search_models(query=q, task=task, limit=limit)


@router.get("/model/{model_id:path}")
def detail(model_id: str) -> Dict[str, Any]:
    return model_detail(model_id)
