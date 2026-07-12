import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.config.settings import LLMBackend, reload_settings
from main import _clamp_role_token_limits
from src.full_stack.backend.utils.llm_client import LLMClient, TokenTracker
import src.full_stack.backend.utils.core.explainability_runner as runner


def test_effective_context_window_known_public_model():
    settings = reload_settings()
    settings.models.backend = LLMBackend.OPENROUTER
    settings.models.public_model_name = "gpt-5-nano"
    settings.models.public_max_context_tokens = 99999
    assert settings.effective_context_window("gpt-5-nano") == 99999


def test_effective_context_window_does_not_apply_public_ctx_to_other_model():
    settings = reload_settings()
    settings.models.backend = LLMBackend.OPENROUTER
    settings.models.public_model_name = "gpt-5-nano"
    settings.models.public_max_context_tokens = 400000
    # Communicator/model-specific context should not inherit gpt-5-nano's context
    assert settings.effective_context_window("gpt-5-mini") == 128000


def test_effective_context_window_local_override():
    settings = reload_settings()
    settings.models.backend = LLMBackend.LOCAL
    settings.models.local_max_tokens = 4096
    settings.models.local_max_model_len = 16384
    assert settings.effective_context_window() == 16384


def test_role_token_limits_are_clamped_to_context():
    settings = reload_settings()
    settings.models.backend = LLMBackend.LOCAL
    settings.models.local_max_tokens = 2048
    settings.models.local_max_model_len = 0
    settings.models.orchestrator_model = "Qwen/Qwen2.5-0.5B-Instruct"
    settings.models.orchestrator_max_tokens = 64000
    _clamp_role_token_limits(settings)
    assert settings.models.orchestrator_max_tokens <= settings.auto_output_token_limit(settings.models.orchestrator_model)


def test_openrouter_model_prefix_resolution():
    client = LLMClient.__new__(LLMClient)
    client.backend = LLMBackend.OPENROUTER
    assert client._resolve_model_name("gpt-5-nano") == "openai/gpt-5-nano"
    assert client._resolve_model_name("anthropic/claude-3.5-sonnet") == "anthropic/claude-3.5-sonnet"


def test_openrouter_runtime_fallback_switches_backend(monkeypatch):
    settings = reload_settings()
    settings.models.backend = LLMBackend.OPENROUTER
    settings.openai_api_key = "test-openai-key"
    settings.models.public_model_name = "openai/gpt-5-nano"
    settings.models.orchestrator_model = "openai/gpt-5-nano"
    settings.models.tool_model = "openai/gpt-5-nano"

    import src.full_stack.backend.utils.llm_client as llm_client_mod
    monkeypatch.setattr(llm_client_mod, "OpenAI", lambda api_key=None, **kwargs: object())

    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    client.backend = LLMBackend.OPENROUTER
    client.client = None
    client.embedding_client = None

    client._switch_to_openai_fallback("ssl failure")

    assert client.backend == LLMBackend.OPENAI
    assert settings.models.backend == LLMBackend.OPENAI
    assert settings.models.public_model_name == "gpt-5-nano"
    assert settings.models.orchestrator_model == "gpt-5-nano"


def test_openrouter_transient_error_detection_includes_clerk_502():
    err = RuntimeError("Error code: 502 - {'error': {'message': 'Failed to authenticate request with Clerk'}}")
    assert LLMClient._is_transient_public_api_error(err) is True


def test_openrouter_model_fallback_candidate_for_provider_model():
    settings = reload_settings()
    settings.models.backend = LLMBackend.OPENROUTER

    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    client.backend = LLMBackend.OPENROUTER

    err = RuntimeError("Error code: 502 - {'error': {'message': 'Failed to authenticate request with Clerk'}}")
    assert client._openrouter_model_fallback_candidate("x-ai/grok-4.1-fast", err) == "openai/gpt-5-nano"
    assert client._openrouter_model_fallback_candidate("openai/gpt-5-nano", err) == ""


def test_openrouter_call_retries_with_openrouter_fallback_model():
    settings = reload_settings()
    settings.models.backend = LLMBackend.OPENROUTER
    settings.openrouter_api_key = "or-key"
    settings.openai_api_key = ""

    call_models = []
    call_kwargs = []

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5

    class _Message:
        content = '{"ok": true}'

    class _Choice:
        message = _Message()
        finish_reason = "stop"

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    def _fake_create(**kwargs):
        call_models.append(kwargs.get("model"))
        call_kwargs.append(kwargs)
        if len(call_models) == 1:
            raise RuntimeError("Error code: 502 - {'error': {'message': 'Failed to authenticate request with Clerk'}}")
        return _Response()

    client = LLMClient.__new__(LLMClient)
    client.settings = settings
    client.backend = LLMBackend.OPENROUTER
    client.token_tracker = TokenTracker()
    client.local_llm = None
    client.embedding_client = None
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_fake_create)
        )
    )

    result = client.call(
        messages=[{"role": "user", "content": "ping"}],
        model="x-ai/grok-4.1-fast",
        max_tokens=128,
        temperature=0.3,
    )

    assert call_models == ["x-ai/grok-4.1-fast", "openai/gpt-5-nano"]
    assert "temperature" in call_kwargs[0]
    assert "temperature" not in call_kwargs[1]
    assert result.model == "openai/gpt-5-nano"


def test_explainability_hybrid_client_kwargs_resolution():
    settings = reload_settings()
    settings.openrouter_api_key = "or-key"
    settings.openrouter_base_url = "https://openrouter.ai/api/v1"
    settings.openrouter_site_url = "https://example.org"
    settings.openrouter_app_name = "COMPASS"

    kwargs = runner._resolve_hybrid_client_kwargs(settings)
    assert kwargs["api_key"] == "or-key"
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert kwargs["default_headers"]["HTTP-Referer"] == "https://example.org"
    assert kwargs["default_headers"]["X-Title"] == "COMPASS"


def test_explainability_runner_partial_failure(monkeypatch, tmp_path):
    settings = reload_settings()
    settings.explainability.enabled = True
    settings.explainability.methods = ["internal", "external"]

    feature_space = {
        "root_node": "__XAI_ROOT__",
        "hierarchy_children": {
            "__XAI_ROOT__": ["dom::BRAIN", "dom::BIO"],
            "dom::BRAIN": ["leaf::brain_a"],
            "dom::BIO": ["leaf::bio_a"],
        },
        "leaf_nodes": ["leaf::brain_a", "leaf::bio_a"],
        "parent_nodes": ["dom::BRAIN", "dom::BIO"],
        "leaf_to_parent": {
            "leaf::brain_a": "dom::BRAIN",
            "leaf::bio_a": "dom::BIO",
        },
        "leaf_to_feature": {
            "leaf::brain_a": {
                "domain": "BRAIN",
                "feature_name": "brain_a",
                "path_in_hierarchy": ["MRI"],
                "value": 1.0,
                "z_score": 1.2,
            },
            "leaf::bio_a": {
                "domain": "BIOLOGICAL_ASSAY",
                "feature_name": "bio_a",
                "path_in_hierarchy": ["Serum"],
                "value": -0.4,
                "z_score": -0.8,
            },
        },
    }
    selected_attempt = {
        "iteration": 2,
        "executor_output": {
            "predictor_call_context": {
                "mode": "direct",
                "high_priority_context": "core context",
                "non_core_context": "non core context",
            }
        },
    }

    class BrokenIG:
        @staticmethod
        def load_model(model_name=None):
            raise RuntimeError("broken iga model")

    class FakeHFR:
        @staticmethod
        def monte_carlo_hfr_tokenshap(**kwargs):
            return {"leaf::brain_a": 0.2, "leaf::bio_a": 0.8}

    def fake_loader(module_name, filename):
        if filename == "ig_attribution.py":
            return BrokenIG
        if filename == "aHFR_TokenSHAP.py":
            return FakeHFR
        raise AssertionError(f"Unexpected module file: {filename}")

    monkeypatch.setattr(runner, "_load_script_module", fake_loader)
    monkeypatch.setattr(runner, "get_llm_client", lambda: object())

    out = runner.run_explainability_methods(
        settings=settings,
        participant_id="SUBJ",
        target_condition="major depressive disorder",
        control_condition="control",
        selected_attempt=selected_attempt,
        feature_space=feature_space,
        output_dir=tmp_path,
    )

    assert out["status"] == "success"
    assert out["methods"]["internal"]["status"] == "failed"
    assert out["methods"]["external"]["status"] == "success"
