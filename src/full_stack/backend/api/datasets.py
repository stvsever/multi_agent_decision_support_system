"""
Participant discovery and input validation.

A participant folder is defined by four files. Rather than failing at run time
with a stack trace, the dashboard validates a folder up front and reports
exactly which file is missing, malformed, or empty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config.settings import PROJECT_ROOT, get_settings
from .cost import participant_input_tokens
from .paths import PSEUDO_INPUTS_DIR

REQUIRED_FILES = {
    "data_overview": "data_overview.json",
    "hierarchical_deviation": "hierarchical_deviation_map.json",
    "multimodal_data": "multimodal_data.json",
    "non_numerical_data": "non_numerical_data.txt",
}

_SCAN_EXCLUDE = {".git", "node_modules", "__pycache__", ".venv", "dist", "build", ".idea"}
MAX_SCAN_DEPTH = 4


def _validate_file(path: Path, key: str) -> Dict[str, Any]:
    if not path.exists():
        return {"file": path.name, "key": key, "present": False, "valid": False, "issue": "File is missing."}
    size = path.stat().st_size
    if size == 0:
        return {"file": path.name, "key": key, "present": True, "valid": False, "size": 0, "issue": "File is empty."}
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text() or "")
        except json.JSONDecodeError as exc:
            return {
                "file": path.name,
                "key": key,
                "present": True,
                "valid": False,
                "size": size,
                "issue": f"Invalid JSON: {exc.msg} at line {exc.lineno}.",
            }
        if not isinstance(payload, dict) or not payload:
            return {
                "file": path.name,
                "key": key,
                "present": True,
                "valid": False,
                "size": size,
                "issue": "Expected a non-empty JSON object.",
            }
    return {"file": path.name, "key": key, "present": True, "valid": True, "size": size, "issue": ""}


def inspect_participant(directory: Path) -> Dict[str, Any]:
    """Full validation report for one candidate participant folder."""
    directory = Path(directory).expanduser()
    files = [_validate_file(directory / name, key) for key, name in REQUIRED_FILES.items()]
    valid = all(f["valid"] for f in files)

    overview: Dict[str, Any] = {}
    overview_path = directory / REQUIRED_FILES["data_overview"]
    if overview_path.exists():
        try:
            loaded = json.loads(overview_path.read_text() or "{}")
            overview = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            overview = {}

    coverage = overview.get("domain_coverage") or {}
    return {
        "id": overview.get("participant_id") or directory.name,
        "directory": str(directory.resolve()),
        "name": directory.name,
        "valid": valid,
        "files": files,
        "missing": [f["file"] for f in files if not f["present"]],
        "input_tokens": participant_input_tokens(directory) if valid else 0,
        "domains": list(overview.get("available_domains") or coverage.keys()),
        "domain_coverage": coverage,
        "token_budget": overview.get("token_budget"),
    }


def is_participant_dir(directory: Path) -> bool:
    return all((directory / name).exists() for name in REQUIRED_FILES.values())


def _walk(root: Path, depth: int = 0) -> Iterable[Path]:
    if depth > MAX_SCAN_DEPTH or not root.is_dir():
        return
    try:
        entries = sorted(p for p in root.iterdir() if p.is_dir())
    except (PermissionError, OSError):
        return
    for entry in entries:
        if entry.name.startswith(".") or entry.name in _SCAN_EXCLUDE:
            continue
        if is_participant_dir(entry):
            yield entry
        else:
            yield from _walk(entry, depth + 1)


def default_roots() -> List[Path]:
    roots = [PSEUDO_INPUTS_DIR]
    results = get_settings().paths.output_dir
    if results.exists():
        roots.append(results)
    return [r for r in roots if r.exists()]


def discover(extra_roots: Optional[List[str]] = None) -> Dict[str, Any]:
    """Scan configured roots for valid participant folders."""
    roots: List[Path] = list(default_roots())
    for raw in extra_roots or []:
        candidate = Path(str(raw)).expanduser()
        if candidate.is_dir() and candidate not in roots:
            roots.append(candidate)

    seen: Dict[str, Dict[str, Any]] = {}
    scanned: List[Dict[str, Any]] = []
    for root in roots:
        found = 0
        if is_participant_dir(root):
            record = inspect_participant(root)
            seen.setdefault(record["directory"], record)
            found = 1
        else:
            for directory in _walk(root):
                record = inspect_participant(directory)
                if record["directory"] not in seen:
                    seen[record["directory"]] = record
                    found += 1
        scanned.append(
            {
                "root": str(root),
                "label": _root_label(root),
                "found": found,
                "bundled": root == PSEUDO_INPUTS_DIR,
            }
        )

    participants = sorted(seen.values(), key=lambda p: p["id"])
    return {"participants": participants, "roots": scanned, "count": len(participants)}


def _root_label(root: Path) -> str:
    if root == PSEUDO_INPUTS_DIR:
        return "Bundled sample participants"
    try:
        return str(root.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(root)


def read_input_file(directory: Path, key: str) -> Dict[str, Any]:
    """Raw contents of one input file, for the inspector panels."""
    name = REQUIRED_FILES.get(key)
    if not name:
        raise KeyError(f"Unknown input file: {key}")
    path = Path(directory).expanduser() / name
    if not path.exists():
        raise FileNotFoundError(f"{name} is missing from {directory}")
    text = path.read_text(errors="replace")
    if path.suffix == ".json":
        try:
            return {"key": key, "file": name, "format": "json", "content": json.loads(text or "{}")}
        except json.JSONDecodeError as exc:
            return {"key": key, "file": name, "format": "text", "content": text, "error": str(exc)}
    return {"key": key, "file": name, "format": "text", "content": text}
