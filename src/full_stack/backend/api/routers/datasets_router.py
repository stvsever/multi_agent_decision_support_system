"""Participant discovery, validation, and input inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from ..config_store import load_config
from ..datasets import browse, discover, inspect_participant, read_input_file
from ..ontology import aggregate_ontology, build_ontology, leaf_distribution
from ..safe_paths import require_directory, to_path
from ..schemas import OntologyAggregateRequest, OntologyDistributionRequest

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _directories(raw: List[str]) -> List[Path]:
    if not raw:
        raise HTTPException(status_code=400, detail="Select at least one participant.")
    return [require_directory(directory) for directory in raw]


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


@router.get("/browse")
def browse_folder(path: str = Query("", description="Absolute folder; blank starts at home")) -> Dict[str, Any]:
    """List one folder so a participant can be picked without typing a path."""
    target = to_path(path) if path.strip() else None
    if path.strip() and target is None:
        raise HTTPException(status_code=400, detail=f"Not a usable path: {path}")
    try:
        return browse(target)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (NotADirectoryError, PermissionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/ontology")
def ontology(directory: str = Query(...)) -> Dict[str, Any]:
    return build_ontology(require_directory(directory))


@router.post("/ontology/aggregate")
def ontology_aggregate(request: OntologyAggregateRequest) -> Dict[str, Any]:
    """One tree across several participants, keeping what only some of them have."""
    return aggregate_ontology(_directories(request.directories))


@router.post("/ontology/distribution")
def ontology_distribution(request: OntologyDistributionRequest) -> Dict[str, Any]:
    """Every participant's reading of one node, typed and ready to plot."""
    if not request.path:
        raise HTTPException(status_code=400, detail="A node path is required.")
    return leaf_distribution(_directories(request.directories), request.path)


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
    already_present = resolved in roots
    if not already_present:
        roots.append(resolved)
        patch_config(ConfigPatch(workspace={"data_roots": roots}))
    # "Added but empty" and "already listed" look identical in a participant
    # count, and only one of them is worth telling the user about.
    return {**discover(roots), "added": not already_present, "already_present": already_present, "root": resolved}


@router.delete("/roots")
def remove_root(path: str = Query(...)) -> Dict[str, Any]:
    from ..config_store import patch_config
    from ..schemas import ConfigPatch

    config = load_config(refresh=True)
    roots = [r for r in config.workspace.data_roots if r != path]
    patch_config(ConfigPatch(workspace={"data_roots": roots}))
    return discover(roots)
