"""System prompt inspection and editing."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ..prompt_store import list_prompts, read_prompt, restore_prompt, write_prompt
from ..schemas import PromptUpdate

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.get("")
def index() -> Dict[str, Any]:
    return {"prompts": list_prompts()}


@router.get("/{scope}/{name}")
def read(scope: str, name: str) -> Dict[str, Any]:
    try:
        return read_prompt(scope, name)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{scope}/{name}")
def write(scope: str, name: str, update: PromptUpdate) -> Dict[str, Any]:
    if not update.content.strip():
        raise HTTPException(status_code=400, detail="A prompt cannot be empty.")
    try:
        return write_prompt(scope, name, update.content)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{scope}/{name}/restore")
def restore(scope: str, name: str) -> Dict[str, Any]:
    try:
        return restore_prompt(scope, name)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
