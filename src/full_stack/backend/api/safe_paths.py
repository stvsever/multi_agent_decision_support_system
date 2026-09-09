"""
Turning client-supplied text into a filesystem path.

`Path.expanduser()` raises RuntimeError for `~unknownuser/...` rather than
leaving the string alone, which would surface as a 500 on every endpoint that
accepts a directory. Everything that reads a path from a request goes through
here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def to_path(raw: str) -> Optional[Path]:
    """Return an absolute path, or None when the text cannot name one."""
    text = str(raw or "").strip()
    if not text:
        return None
    candidate = Path(text)
    try:
        candidate = candidate.expanduser()
    except (RuntimeError, OSError):
        # `~someone` with no such user: treat it as an unusable path, not a crash.
        return None
    try:
        return candidate.resolve()
    except (OSError, ValueError):
        return None


def require_directory(raw: str) -> Path:
    """Resolve to an existing directory or raise a 404-shaped error."""
    from fastapi import HTTPException

    path = to_path(raw)
    if path is None:
        raise HTTPException(status_code=400, detail=f"Not a usable path: {raw}")
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"No directory at {path}")
    return path
