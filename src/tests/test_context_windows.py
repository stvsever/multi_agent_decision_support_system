"""
Context window resolution per role.

Order: a decision model's state limit, a per-role override, the served length on
the local backend, the cached OpenRouter catalog, the built-in table, the
explicitly configured public window, then the generic fallback. Pure settings
arithmetic: nothing leaves the process.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.config import settings as settings_module
from src.full_stack.backend.config.settings import (
    GENERIC_FALLBACK_CONTEXT_TOKENS,
    KNOWN_CONTEXT_WINDOWS,
    LLMBackend,
    Settings,
    known_context_window,
    llm_predictor_input_budget,
    parse_role_context_windows,
)

DEEPSEEK = "deepseek/deepseek-v4-flash-0731"
CATALOG = [
    {"id": DEEPSEEK, "context_length": 1_310_720},
    {"id": "openai/gpt-4.1", "context_length": 1_047_576},
    {"id": "openai/gpt-5", "context_length": 400_000},
    {"id": "anthropic/claude-sonnet-4.5", "context_length": 1_000_000},
    {"id": "qwen/qwen3-14b", "context_length": 131_072},
]


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(settings_module, "_CATALOG_CONTEXT_CACHE", {})


@pytest.fixture
def settings():
    s = Settings()
    s.openrouter_base_url = "https://openrouter.ai/api/v1"
    return s


@pytest.fixture
def catalog(tmp_path):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "model_catalog.json").write_text(json.dumps({"models": CATALOG, "fetched_at": 0}))


# --- built-in table (no catalog) -------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        (DEEPSEEK, 1_048_576),
        ("deepseek/deepseek-v4-flash", 1_048_576),
        ("openai/gpt-4.1", 1_047_576),
        ("gpt-4.1-mini", 1_047_576),
        ("openai/gpt-5", 272_000),
        ("gpt-5-nano", 272_000),
        ("openai/gpt-4o", 128_000),
        ("anthropic/claude-sonnet-4.5", 200_000),
        ("anthropic/claude-3.5-sonnet", 200_000),
        ("Qwen/Qwen3-14B-AWQ", 40_960),
        ("qwen/qwen3-32b", 40_960),
        ("Qwen/Qwen2.5-0.5B-Instruct", 32_768),
        ("openai/gpt-4o:free", 128_000),
        ("~deepseek/deepseek-v4-flash-0731", 1_048_576),
    ],
)
def test_common_models_resolve_from_the_table_without_a_catalog(settings, model, expected):
    resolved = settings.context_window_resolution(model)
    assert (resolved.tokens, resolved.source) == (expected, "known_table")


@pytest.mark.parametrize("model", ["acme/unknown-model", "my-finetune", "", None])
def test_unknown_ids_get_the_configured_window_then_the_fallback(settings, model):
    settings.models.public_model_name = "acme/unknown-default"
    assert settings.context_window_resolution(model).source == "fallback"
    assert settings.effective_context_window(model) == GENERIC_FALLBACK_CONTEXT_TOKENS
    settings.models.public_max_context_tokens = 64_000
    assert settings.context_window_resolution(model).source == "configured"
    assert settings.effective_context_window(model) == 64_000


def test_the_table_keys_are_normalized():
    assert all(key == key.lower() and "/" not in key for key in KNOWN_CONTEXT_WINDOWS)
    assert known_context_window("unknown/model") == 0
    assert known_context_window("Qwen/Qwen3-14B-GPTQ-Int4") == 40_960


# --- catalog ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        (DEEPSEEK, 1_310_720),
        ("openai/gpt-4.1", 1_047_576),
        ("openai/gpt-5", 400_000),
        ("gpt-5", 400_000),
        ("anthropic/claude-sonnet-4.5", 1_000_000),
        ("qwen/qwen3-14b", 131_072),
    ],
)
def test_the_catalog_comes_first(settings, catalog, model, expected):
    resolved = settings.context_window_resolution(model)
    assert (resolved.tokens, resolved.source) == (expected, "catalog")


def test_a_configured_window_does_not_override_a_known_model(settings, catalog):
    settings.models.public_max_context_tokens = 32_000
    assert settings.effective_context_window(DEEPSEEK) == 1_310_720
    assert settings.effective_context_window("Qwen/Qwen3-14B-AWQ") == 40_960
    assert settings.effective_context_window("acme/unknown") == 32_000


def test_the_catalog_is_skipped_for_a_self_hosted_endpoint(settings, catalog):
    settings.openrouter_base_url = "http://gpu042.cluster.local:8000/v1"
    assert settings.context_window_resolution("qwen/qwen3-14b").source == "known_table"
    settings.models.backend = LLMBackend.OPENAI
    assert settings.context_window_resolution("openai/gpt-5").source == "catalog"


# --- overrides -------------------------------------------------------------------


def test_role_overrides_win_and_all_covers_every_role(settings, catalog):
    settings.models.role_context_windows = {"all": 65_536, "predictor": 128_000}
    assert settings.effective_context_window(DEEPSEEK, role="predictor") == 128_000
    assert settings.effective_context_window(DEEPSEEK, role="tool") == 65_536
    # No role: only "all" applies (the derived token budgets use this).
    assert settings.effective_context_window(DEEPSEEK) == 65_536
    assert settings.context_window_resolution(DEEPSEEK, role="critic").source == "role_override"


def test_a_decision_model_keeps_its_state_limit_whatever_the_override(settings):
    settings.models.role_context_windows = {"predictor": 500_000, "all": 500_000}
    resolved = settings.context_window_resolution("typesafe/jev-1.13", role="predictor")
    assert (resolved.tokens, resolved.source) == (int(32_000 / 1.2), "decision_state")


def test_the_local_backend_uses_the_served_length(settings, catalog):
    settings.models.backend = LLMBackend.LOCAL
    settings.models.local_max_model_len = 24_576
    assert settings.context_window_resolution("Qwen/Qwen3-14B-AWQ").tokens == 24_576
    assert settings.context_window_resolution("qwen/qwen3-14b").source == "local_served"
    settings.models.local_max_model_len = 0
    settings.models.local_max_tokens = 16_384
    assert settings.effective_context_window("Qwen/Qwen3-14B-AWQ") == 16_384
    settings.models.role_context_windows = {"tool": 8_192}
    assert settings.effective_context_window("Qwen/Qwen3-14B-AWQ", role="tool") == 8_192


def test_every_role_is_reported(settings, catalog):
    settings.models.tool_model = "openai/gpt-4o"
    windows = settings.role_context_windows()
    assert set(windows) == {"orchestrator", "critic", "predictor", "integrator", "communicator", "tool"}
    assert windows["tool"].tokens == 128_000 and windows["tool"].source == "known_table"
    assert windows["predictor"].tokens == 1_310_720


def test_the_predictor_input_budget_follows_the_resolved_window(settings, catalog):
    settings.models.predictor_max_tokens = 12_000
    settings.token_budget.max_agent_input_tokens = 0
    assert llm_predictor_input_budget(settings) == 1_310_720 - 8_192
    settings.models.role_context_windows = {"predictor": 32_000}
    assert llm_predictor_input_budget(settings) == 32_000 - 8_192
    settings.token_budget.max_agent_input_tokens = 10_000
    assert llm_predictor_input_budget(settings) == 10_000


# --- parsing ---------------------------------------------------------------------


def test_role_window_pairs_parse():
    assert parse_role_context_windows("predictor=128000, tool=32_000") == {"predictor": 128_000, "tool": 32_000}
    assert parse_role_context_windows(["all=65536", "critic=40000"]) == {"all": 65_536, "critic": 40_000}
    assert parse_role_context_windows(None) == {}


@pytest.mark.parametrize(
    "raw, message",
    [
        ("predictor", "ROLE=TOKENS"),
        ("pred=1000", "Unknown role"),
        ("tool=big", "whole number"),
        ("tool=100", "at least 1024"),
    ],
)
def test_bad_role_window_pairs_are_refused(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_role_context_windows(raw)


# --- dashboard schema ------------------------------------------------------------


def test_the_dashboard_role_windows_accept_zero_or_a_real_window():
    from pydantic import ValidationError

    from src.full_stack.backend.api.schemas import DashboardConfig, RoleWindowMap

    assert DashboardConfig().models.role_context_windows.model_dump() == {
        "orchestrator": 0,
        "integrator": 0,
        "predictor": 0,
        "critic": 0,
        "communicator": 0,
        "tool": 0,
    }
    assert RoleWindowMap(predictor=32_768).predictor == 32_768
    with pytest.raises(ValidationError, match="at least 1024"):
        RoleWindowMap(tool=500)
