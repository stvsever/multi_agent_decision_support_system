"""
Persisted dashboard configuration and credential handling.

Secrets live in a 0600 file next to the config and never travel back to the
browser: the client only ever sees whether a key is configured, where it came
from, and a masked preview.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config.settings import _resolve_secret_from_env_or_dotenv
from .paths import config_dir, config_file
from .schemas import ConfigPatch, CredentialStatus, DashboardConfig

_SECRET_FILE = "credentials.json"
#: Environment names accepted as a fallback for each provider, most specific
#: first. HuggingFace tooling reads either name, so both are honoured.
_PROVIDER_ENV: Dict[str, Tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "huggingface": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
}

_lock = threading.RLock()
_cached: Optional[DashboardConfig] = None
#: Explanations for anything the loader had to repair, shown once by the client.
_notices: List[str] = []


def _secret_file() -> Path:
    return config_dir() / _SECRET_FILE


def _write_private(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# --- configuration -----------------------------------------------------------


def _int_or(value: Any, fallback: int) -> int:
    """A number written as text, a float, or nonsense, reduced to one integer."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _section(healed: Dict[str, Any], name: str, label: str, notices: List[str]) -> Dict[str, Any]:
    """
    One configuration section as a dictionary, whatever was actually stored.

    A build that wrote a section as a string or a list would otherwise make the
    repair itself the thing that raises, which is the failure this whole
    function exists to prevent.
    """
    value = healed.get(name)
    if isinstance(value, dict):
        return dict(value)
    if value is not None:
        notices.append(f"The stored {label} settings were not readable, so that section went back to its defaults.")
    return {}


def _heal(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Repair a stored config that describes a state the service can no longer run.

    Refusing to load would lock the user out of the one screen that can fix the
    problem, so every repair is applied silently and explained in a notice.
    """
    if not isinstance(raw, dict):
        return {}, ["The stored settings file was unreadable, so the defaults were restored."]

    healed = dict(raw)
    notices: List[str] = []

    connection = _section(healed, "connection", "connection", notices)
    local = _section(healed, "local", "local backend", notices)
    raw_backend = connection.get("backend")
    backend = raw_backend if isinstance(raw_backend, str) else ""

    if backend == "openai":
        connection["backend"] = "openrouter"
        backend = "openrouter"
        notices.append(
            "OpenAI is no longer a hosting option, so this workspace was moved to OpenRouter."
        )

    model_name = local.get("model_name")
    if backend == "local" and not (isinstance(model_name, str) and model_name.strip()):
        connection["backend"] = "openrouter"
        notices.append(
            "The backend was set to Local with no model chosen, which cannot run. "
            "It was switched back to OpenRouter: pick a local model to switch again."
        )

    # A config written before GPU count existed can describe more parallelism
    # than the new field's default allows, and that is a description of a real
    # machine rather than a mistake.
    needed = _int_or(local.get("tensor_parallel_size") or 1, 1) * _int_or(local.get("pipeline_parallel_size") or 1, 1)
    if needed > _int_or(local.get("gpu_count") or 1, 1):
        local["gpu_count"] = min(64, max(1, needed))
        notices.append(
            f"The saved parallelism needs {needed} GPUs, so the GPU count was raised to match."
        )

    healed["connection"] = connection
    healed["local"] = local
    return healed, notices


#: Environment names that seed the very first load. A self-hosted compose stack
#: points these at the vLLM service it starts alongside the dashboard, and the
#: run workers are handed the saved configuration rather than the environment,
#: so without this seeding the container would quietly call the public provider.
_CONNECTION_ENV: Dict[str, str] = {
    "openrouter_base_url": "OPENROUTER_BASE_URL",
    "openrouter_app_name": "OPENROUTER_APP_NAME",
    "openrouter_site_url": "OPENROUTER_SITE_URL",
}


def _environment_seed() -> Dict[str, Any]:
    """
    First-boot connection settings taken from the environment.

    Only consulted when no settings file exists, so a value the user has
    deliberately saved is never overridden by the environment it happens to run
    in. Nothing is written to disk either: the seed is recomputed on each load
    until the first real save replaces it.
    """
    connection: Dict[str, Any] = {}
    for field, env_name in _CONNECTION_ENV.items():
        value = os.getenv(env_name, "").strip()
        if value:
            connection[field] = value
    return {"connection": connection} if connection else {}


def load_config(refresh: bool = False) -> DashboardConfig:
    global _cached
    with _lock:
        if _cached is not None and not refresh:
            return _cached
        path = config_file()
        raw: Dict[str, Any] = {}
        notices: List[str] = []
        stored = path.exists()
        if stored:
            try:
                raw = json.loads(path.read_text() or "{}")
            except json.JSONDecodeError:
                raw = {}
                notices.append("The stored settings file was not valid JSON, so the defaults were restored.")
        else:
            raw = _environment_seed()
        try:
            # Healing reads a file another build wrote, so it makes assumptions
            # about shape that the file is under no obligation to honour. It
            # belongs inside the guard, not before it.
            healed, repairs = _heal(raw)
            _cached = DashboardConfig.model_validate(healed)
            notices.extend(repairs)
        except Exception as exc:
            # A config written by an incompatible build must not lock the user out.
            _cached = DashboardConfig()
            notices.append(f"The stored settings could not be read ({exc.__class__.__name__}), so the defaults were restored.")
        _notices[:] = notices
        if notices and stored:
            # Persist the repair, or the same notice reappears on every load.
            _write_private(config_file(), _cached.model_dump(mode="json"))
        return _cached


def load_notices() -> List[str]:
    """What the last load had to repair, for a one-time message in the client."""
    with _lock:
        return list(_notices)


def save_config(config: DashboardConfig) -> DashboardConfig:
    global _cached
    with _lock:
        _write_private(config_file(), config.model_dump(mode="json"))
        _cached = config
        # Whatever the loader repaired has just been superseded by a deliberate
        # write, so the explanation is no longer worth showing.
        _notices.clear()
        return config


def patch_config(patch: ConfigPatch) -> DashboardConfig:
    current = load_config().model_dump(mode="json")
    delta = {k: v for k, v in patch.model_dump(exclude_unset=True).items() if v is not None}
    merged = _deep_merge(current, delta)
    return save_config(DashboardConfig.model_validate(merged))


def reset_config() -> DashboardConfig:
    return save_config(DashboardConfig())


# --- credentials -------------------------------------------------------------


def _read_secrets() -> Dict[str, str]:
    path = _secret_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def set_credential(provider: str, api_key: str) -> CredentialStatus:
    with _lock:
        secrets = _read_secrets()
        key = (api_key or "").strip()
        if key:
            secrets[provider] = key
        else:
            secrets.pop(provider, None)
        _write_private(_secret_file(), secrets)
    return credential_status(provider)


def _from_environment(provider: str) -> str:
    for env_name in _PROVIDER_ENV.get(provider, ()):
        value = _resolve_secret_from_env_or_dotenv(env_name).strip()
        if value:
            return value
    return ""


def get_credential(provider: str) -> str:
    """Stored key wins; the environment and .env remain a valid fallback."""
    stored = _read_secrets().get(provider, "").strip()
    if stored:
        return stored
    return _from_environment(provider)


def mask(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if len(text) <= 12:
        return f"{text[:2]}{'.' * 6}"
    return f"{text[:6]}{'.' * 6}{text[-4:]}"


def credential_status(provider: str) -> CredentialStatus:
    stored = _read_secrets().get(provider, "").strip()
    if stored:
        return CredentialStatus(provider=provider, configured=True, masked=mask(stored), source="stored")
    env_value = _from_environment(provider)
    if env_value:
        return CredentialStatus(
            provider=provider, configured=True, masked=mask(env_value), source="environment"
        )
    return CredentialStatus(provider=provider, configured=False, masked="", source="none")


# --- engine handoff ----------------------------------------------------------


def resolve_role(config: DashboardConfig, role: str) -> str:
    """The model a given agent role will actually use."""
    explicit = str(getattr(config.models.role_models, role, "") or "").strip()
    return explicit or config.models.default_model


def worker_environment(config: DashboardConfig) -> Dict[str, str]:
    """Environment overlay handed to a run worker process."""
    env = dict(os.environ)
    env["COMPASS_EXECUTOR_MAX_WORKERS"] = str(config.engine.executor_max_workers)
    key = get_credential("openrouter")
    if key:
        env["OPENROUTER_API_KEY"] = key
    hf_token = get_credential("huggingface")
    if hf_token:
        # Both names, because the hub client and the tooling around it disagree
        # on which one they read.
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if config.connection.openrouter_base_url:
        env["OPENROUTER_BASE_URL"] = config.connection.openrouter_base_url
    if config.connection.openrouter_site_url:
        env["OPENROUTER_SITE_URL"] = config.connection.openrouter_site_url
    if config.connection.openrouter_app_name:
        env["OPENROUTER_APP_NAME"] = config.connection.openrouter_app_name
    env["PYTHONUNBUFFERED"] = "1"
    return env
