"""Filesystem locations the dashboard service reads and writes."""

from __future__ import annotations

import os
from pathlib import Path

from ..config.settings import BACKEND_ROOT, PROJECT_ROOT

#: Bundled synthetic participants, always discoverable.
PSEUDO_INPUTS_DIR = BACKEND_ROOT / "data" / "pseudo_data" / "inputs"
PSEUDO_OUTPUTS_DIR = BACKEND_ROOT / "data" / "pseudo_data" / "outputs"

#: Where the compiled web client lands after `npm run build`.
WEB_CLIENT_DIR = PROJECT_ROOT / "src" / "full_stack" / "frontend" / "dist"

#: Pristine copies of the shipped prompts, used by "restore default".
AGENT_PROMPTS_DIR = BACKEND_ROOT / "agents" / "prompts"
TOOL_PROMPTS_DIR = BACKEND_ROOT / "tools" / "prompts"


def config_dir() -> Path:
    """User-scoped directory holding dashboard preferences and run history."""
    override = os.getenv("COMPASS_HOME", "").strip()
    root = Path(override).expanduser() if override else Path.home() / ".compass"
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    return root


def config_file() -> Path:
    return config_dir() / "dashboard.json"


def prompt_overrides_dir() -> Path:
    path = config_dir() / "prompts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runs_dir() -> Path:
    path = config_dir() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def catalog_cache_file() -> Path:
    return config_dir() / "model_catalog.json"


def default_results_dir() -> Path:
    return PROJECT_ROOT / "results"
