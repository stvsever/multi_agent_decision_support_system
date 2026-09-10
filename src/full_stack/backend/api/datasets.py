"""
Participant discovery and input validation.

A participant folder is defined by four files. Rather than failing at run time
with a stack trace, the dashboard validates a folder up front and reports
exactly which file is missing, malformed, or empty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config.settings import PROJECT_ROOT, get_settings
from .config_store import load_config
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
#: Enough near misses to see the pattern, few enough to stay a readable list.
MAX_NEAR_MISSES = 20
#: A browser listing is for choosing a folder, not for reading a directory that
#: happens to hold a hundred thousand files.
MAX_BROWSE_ENTRIES = 500


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


def _present_and_missing(directory: Path) -> Tuple[List[str], List[str]]:
    present = [name for name in REQUIRED_FILES.values() if (directory / name).exists()]
    missing = [name for name in REQUIRED_FILES.values() if name not in present]
    return present, missing


def _scan(root: Path) -> Dict[str, Any]:
    """
    One root's participants plus the evidence for why the rest are not.

    A folder that yields nothing is the single most confusing thing the picker
    can do, so the scan also reports how much it looked at and which folders
    came close, which is what lets the interface say what is actually wrong.
    """
    participants: List[Path] = []
    near_misses: List[Dict[str, Any]] = []
    scanned = 0

    def visit(directory: Path, depth: int) -> None:
        nonlocal scanned
        scanned += 1
        if is_participant_dir(directory):
            participants.append(directory)
            return
        present, missing = _present_and_missing(directory)
        if present and len(near_misses) < MAX_NEAR_MISSES:
            near_misses.append(
                {"directory": str(directory), "present": present, "missing": missing}
            )
        # The depth counts folders below the root, and the root itself is depth
        # zero, so the last level worth descending into is MAX_SCAN_DEPTH.
        if depth > MAX_SCAN_DEPTH:
            return
        try:
            children = sorted(p for p in directory.iterdir() if p.is_dir())
        except (PermissionError, OSError):
            return
        for child in children:
            if child.name.startswith(".") or child.name in _SCAN_EXCLUDE:
                continue
            visit(child, depth + 1)

    if root.is_dir():
        visit(root, 0)
    return {"participants": participants, "scanned_dir_count": scanned, "near_misses": near_misses}


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
        report = _scan(root)
        found = 0
        for directory in report["participants"]:
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
                "scanned_dir_count": report["scanned_dir_count"],
                "near_misses": report["near_misses"],
            }
        )

    participants = sorted(seen.values(), key=lambda p: p["id"])
    return {
        "participants": participants,
        "roots": scanned,
        "count": len(participants),
        "required_files": list(REQUIRED_FILES.values()),
    }


def _root_label(root: Path) -> str:
    if root == PSEUDO_INPUTS_DIR:
        return "Bundled sample participants"
    try:
        return str(root.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(root)


def browse_roots() -> List[Dict[str, Any]]:
    """Sensible places to start browsing from, in the order a user would try."""
    entries: List[Tuple[str, Path]] = [
        ("Bundled sample participants", PSEUDO_INPUTS_DIR),
        ("Home", Path.home()),
        ("Project", PROJECT_ROOT),
        ("Results", get_settings().paths.output_dir),
    ]
    for raw in load_config().workspace.data_roots:
        candidate = Path(str(raw)).expanduser()
        entries.append((candidate.name or str(candidate), candidate))

    seen: set[str] = set()
    roots: List[Dict[str, Any]] = []
    for label, path in entries:
        try:
            resolved = path.expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        roots.append({"label": label, "path": key})
    return roots


def browse(directory: Optional[Path] = None) -> Dict[str, Any]:
    """
    One directory listing, for picking a folder without typing a path.

    Names and structure only: nothing here opens a file, so pointing the browser
    at a directory of private data reveals no more than its listing.
    """
    target = Path(directory).expanduser() if directory is not None else Path.home()
    try:
        target = target.resolve()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Not a usable path: {directory}") from exc
    if not target.exists():
        raise FileNotFoundError(f"No directory at {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"{target} is a file, not a folder.")

    try:
        children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError as exc:
        raise PermissionError(f"This folder cannot be read: {target}") from exc
    except OSError as exc:
        raise OSError(f"This folder cannot be listed: {target}") from exc

    entries: List[Dict[str, Any]] = []
    for child in children[:MAX_BROWSE_ENTRIES]:
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        child_dirs = 0
        participant = False
        if is_dir:
            participant = is_participant_dir(child)
            try:
                child_dirs = sum(1 for p in child.iterdir() if p.is_dir())
            except (PermissionError, OSError):
                child_dirs = 0
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "is_dir": is_dir,
                "is_participant": participant,
                "child_dir_count": child_dirs,
            }
        )

    present, missing = _present_and_missing(target)
    parent = str(target.parent) if target.parent != target else None
    return {
        "path": str(target),
        "parent": parent,
        "entries": entries,
        "truncated": len(children) > MAX_BROWSE_ENTRIES,
        "is_participant": not missing,
        "present": present,
        "missing": missing,
        "roots": browse_roots(),
        "required_files": list(REQUIRED_FILES.values()),
    }


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
