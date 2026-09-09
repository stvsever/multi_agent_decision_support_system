"""Participant discovery, validation, and input inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from ..config_store import load_config
from ..datasets import discover, inspect_participant, read_input_file
from ..ontology import build_ontology
from ..safe_paths import require_directory, to_path

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/participants")
def participants(refresh: bool = Query(False)) -> Dict[str, Any]:
    config = load_config(refresh=refresh)
    return discover(config.workspace.data_roots)


@router.get("/participant")
def participant(directory: str = Query(..., description="Absolute participant folder")) -> Dict[str, Any]:
    return inspect_participant(require_directory(directory))


@router.get("/participant/file")
def participant_file(directory: str = Query(...), key: str = Query(...)) -> Dict[str, Any]:
    path = require_directory(directory)
    try:
        return read_input_file(path, key)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ontology")
def ontology(directory: str = Query(...)) -> Dict[str, Any]:
    return build_ontology(require_directory(directory))


@router.post("/roots")
def add_root(payload: Dict[str, str]) -> Dict[str, Any]:
    """Register a folder to scan for participants."""
    raw = str(payload.get("path") or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="A path is required.")
    path = require_directory(raw)

    from ..config_store import patch_config
    from ..schemas import ConfigPatch

    config = load_config(refresh=True)
    roots: List[str] = list(config.workspace.data_roots)
    resolved = str(path.resolve())
    if resolved not in roots:
        roots.append(resolved)
    patch_config(ConfigPatch(workspace={"data_roots": roots}))
    return discover(roots)


@router.delete("/roots")
def remove_root(path: str = Query(...)) -> Dict[str, Any]:
    from ..config_store import patch_config
    from ..schemas import ConfigPatch

    config = load_config(refresh=True)
    roots = [r for r in config.workspace.data_roots if r != path]
    patch_config(ConfigPatch(workspace={"data_roots": roots}))
    return discover(roots)
