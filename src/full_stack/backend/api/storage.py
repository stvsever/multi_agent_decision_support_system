"""
What COMPASS has written on this machine, and how to take it back.

Everything the dashboard persists lives outside the repository: preferences and
run history under the config directory, embeddings under the OS cache, reports
under the configured output directory. None of it is discoverable by looking at
the project, so the interface needs a way to show it and clear it.

Deletion is deliberately narrow. Each entry names a fixed location derived from
the same helpers the rest of the service uses, and a delete only ever removes
inside that location, so nothing here can be pointed at an arbitrary path.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .paths import catalog_cache_file, config_dir, config_file, prompt_overrides_dir, runs_dir


@dataclass(frozen=True)
class StorageEntry:
    key: str
    label: str
    description: str
    #: Removing this loses work rather than a cache that will simply refill.
    #: Reported as `loses_work`, which is not the same question as whether the
    #: service is willing to delete the entry at all.
    loses_work: bool
    locate: Callable[[], Optional[Path]]


def _embedding_cache() -> Optional[Path]:
    """
    The embedding store the engine writes between runs.

    Resolved through the engine's own helper so the two cannot disagree about
    where the file lives on a given platform. The enclosing directory is
    reported rather than the file, because SQLite leaves write-ahead siblings
    next to it that would otherwise survive a clear.
    """
    try:
        from ..utils.core.embedding_store import _default_db_path

        path = Path(_default_db_path()).expanduser()
    except Exception:
        return None
    return path.parent if path.parent.name == "compass" else path


def _output_dir() -> Optional[Path]:
    try:
        from .config_store import load_config

        configured = str(load_config().workspace.output_dir or "").strip()
        if configured:
            return Path(configured).expanduser()
    except Exception:
        pass
    try:
        from ..config.settings import get_settings

        return Path(get_settings().paths.output_dir)
    except Exception:
        return None


ENTRIES: tuple[StorageEntry, ...] = (
    StorageEntry(
        key="model_catalog",
        label="Model catalog cache",
        description="The provider's model list and prices, cached so the interface works offline. It refetches on demand.",
        loses_work=False,
        locate=catalog_cache_file,
    ),
    StorageEntry(
        key="runs",
        label="Run history",
        description="Every recorded run: its events, logs, token ledger and result. Removing this empties the runs list.",
        loses_work=True,
        locate=runs_dir,
    ),
    StorageEntry(
        key="prompts",
        label="Edited prompts",
        description="Your edits to the agent and tool prompts. Removing them restores the shipped text everywhere.",
        loses_work=True,
        locate=prompt_overrides_dir,
    ),
    StorageEntry(
        key="embeddings",
        label="Embedding cache",
        description="Vectors computed during earlier runs, kept so repeated evidence is not embedded twice.",
        loses_work=False,
        locate=_embedding_cache,
    ),
    StorageEntry(
        key="outputs",
        label="Generated reports",
        description="Reports, performance records and audits written into the output directory.",
        loses_work=True,
        locate=_output_dir,
    ),
    StorageEntry(
        key="config",
        label="Saved configuration",
        description="Your settings file. Removing it restores every default; stored credentials are not touched.",
        loses_work=True,
        locate=config_file,
    ),
)

_BY_KEY: Dict[str, StorageEntry] = {entry.key: entry for entry in ENTRIES}


def _size_of(path: Path) -> tuple[int, Optional[int]]:
    """
    Bytes and item count, tolerating a tree that changes while it is walked.

    A single file has no item count: reporting 1 would invite the interface to
    say "1 item" about something that is one file and nothing else.
    """
    if path.is_file():
        try:
            return path.stat().st_size, None
        except OSError:
            return 0, None
    total = 0
    items = 0
    if not path.is_dir():
        return 0, None
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
                items += 1
        except OSError:
            continue
    return total, items


def describe() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for entry in ENTRIES:
        try:
            path = entry.locate()
        except Exception:
            path = None
        exists = bool(path and path.exists())
        size, items = _size_of(path) if exists and path else (0, None)
        rows.append(
            {
                "key": entry.key,
                "label": entry.label,
                "description": entry.description,
                "path": str(path) if path else "",
                "bytes": size,
                "item_count": items,
                "exists": exists,
                # Every entry here can be deleted; whether that costs work is a
                # separate question the interface phrases in its own words.
                "removable": True,
                "loses_work": entry.loses_work,
            }
        )
    return {"entries": rows, "config_dir": str(config_dir()), "total_bytes": sum(r["bytes"] for r in rows)}


def clear(key: str) -> Dict[str, Any]:
    """
    Remove one entry's contents.

    A directory has its children removed rather than the directory itself, so
    the code that expects it to exist keeps working without a re-create dance.
    """
    entry = _BY_KEY.get(key)
    if entry is None:
        raise KeyError(key)

    path = entry.locate()
    if path is None or not path.exists():
        # Nothing on disk is the state the caller asked for, so this is a
        # success that freed nothing rather than a failure.
        return {"key": key, "deleted": True, "removed": 0, "freed_bytes": 0, "path": str(path) if path else ""}

    size, items = _size_of(path)
    if path.is_file():
        path.unlink(missing_ok=True)
    else:
        for child in path.iterdir():
            try:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except OSError:
                continue

    if key == "model_catalog":
        # The in-memory copy would otherwise outlive the file it mirrors.
        try:
            from . import catalog

            with catalog._lock:  # noqa: SLF001 - the module owns this lock for exactly this reason
                catalog._memory.pop("catalog", None)
        except Exception:
            pass

    if key == "config":
        # Same reason: the cached configuration would keep serving the deleted
        # file's contents to every later read, including the run workers.
        try:
            from .config_store import load_config

            load_config(refresh=True)
        except Exception:
            pass

    return {"key": key, "deleted": True, "removed": items or 0, "freed_bytes": size, "path": str(path)}
