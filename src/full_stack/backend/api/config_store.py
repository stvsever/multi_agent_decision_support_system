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
from typing import Any, Dict, Optional

from ..config.settings import _resolve_secret_from_env_or_dotenv
from .paths import config_dir, config_file
from .schemas import ConfigPatch, CredentialStatus, DashboardConfig

_SECRET_FILE = "credentials.json"
_PROVIDER_ENV = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}

_lock = threading.RLock()
_cached: Optional[DashboardConfig] = None


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


def load_config(refresh: bool = False) -> DashboardConfig:
    global _cached
    with _lock:
        if _cached is not None and not refresh:
            return _cached
        path = config_file()
        raw: Dict[str, Any] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text() or "{}")
            except json.JSONDecodeError:
                raw = {}
        try:
            _cached = DashboardConfig.model_validate(raw)
        except Exception:
            # A config written by an incompatible build must not lock the user out.
            _cached = DashboardConfig()
        return _cached


def save_config(config: DashboardConfig) -> DashboardConfig:
    global _cached
    with _lock:
        _write_private(config_file(), config.model_dump(mode="json"))
        _cached = config
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


def get_credential(provider: str) -> str:
    """Stored key wins; the environment and .env remain a valid fallback."""
    stored = _read_secrets().get(provider, "").strip()
    if stored:
        return stored
    env_name = _PROVIDER_ENV.get(provider, "")
    if not env_name:
        return ""
    return _resolve_secret_from_env_or_dotenv(env_name).strip()


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
    env_name = _PROVIDER_ENV.get(provider, "")
    env_value = _resolve_secret_from_env_or_dotenv(env_name).strip() if env_name else ""
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
    openai_key = get_credential("openai")
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key
    if config.connection.openrouter_base_url:
        env["OPENROUTER_BASE_URL"] = config.connection.openrouter_base_url
    if config.connection.openrouter_site_url:
        env["OPENROUTER_SITE_URL"] = config.connection.openrouter_site_url
    if config.connection.openrouter_app_name:
        env["OPENROUTER_APP_NAME"] = config.connection.openrouter_app_name
    env["PYTHONUNBUFFERED"] = "1"
    return env
