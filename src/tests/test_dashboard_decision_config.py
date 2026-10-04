"""Decision models and orchestration routing in the dashboard configuration."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.api import DEFAULT_MODEL
from src.full_stack.backend.api import catalog as catalog_module
from src.full_stack.backend.api import config_store
from src.full_stack.backend.api import engine_bridge
from src.full_stack.backend.api.schemas import ConfigPatch, DashboardConfig, RunOverrides

JEV = "typesafe/jev-1.13"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path))
    config_store._cached = None
    catalog_module._memory.clear()
    yield
    config_store._cached = None
    catalog_module._memory.clear()


@pytest.fixture
def client():
    from src.full_stack.backend.api.app import create_app

    return TestClient(create_app())


def _with_keys(monkeypatch, **keys):
    """Credentials as given, whatever the developer's environment holds."""
    monkeypatch.setattr(engine_bridge, "get_credential", lambda provider: keys.get(provider, ""))


# --- validation ----------------------------------------------------------------


def test_the_new_sections_have_the_agreed_defaults():
    config = DashboardConfig()
    assert config.engine.orchestration_mode == "auto"
    assert config.engine.orchestration_threshold_tokens == 0
    assert config.decision.model_dump() == {
        "provider": "openrouter",
        "choice_orders": 3,
        "score_levels": 10,
        "regression_refine": True,
        "stability_threshold": 0.20,
        "sufficiency_threshold": 0.0,
        "compiler_model": "",
    }


def test_a_decision_model_may_serve_the_predictor():
    config = DashboardConfig.model_validate({"models": {"role_models": {"predictor": JEV}}})
    assert config.models.role_models.predictor == JEV


@pytest.mark.parametrize(
    "models,named",
    [
        ({"default_model": JEV}, "default model"),
        ({"role_models": {"critic": JEV}}, "critic"),
        ({"role_models": {"orchestrator": "typesafe/jev-latest", "tool": JEV}}, "orchestrator, tool"),
    ],
)
def test_a_decision_model_outside_the_predictor_is_rejected(models, named):
    with pytest.raises(ValidationError) as excinfo:
        DashboardConfig.model_validate({"models": models})
    assert "only serve the Predictor" in str(excinfo.value)
    assert named in str(excinfo.value)


def test_the_question_book_compiler_must_be_a_conventional_llm():
    with pytest.raises(ValidationError) as excinfo:
        DashboardConfig.model_validate({"decision": {"compiler_model": JEV}})
    assert "question book compiler" in str(excinfo.value)


@pytest.mark.parametrize(
    "patch,expected",
    [
        ({"engine": {"orchestration_mode": "sometimes"}}, "orchestration_mode"),
        ({"engine": {"orchestration_threshold_tokens": -1}}, "orchestration_threshold_tokens"),
        ({"decision": {"provider": "elsewhere"}}, "provider"),
        ({"decision": {"choice_orders": 7}}, "choice_orders"),
        ({"decision": {"score_levels": 11}}, "score_levels"),
        ({"decision": {"stability_threshold": 0.01}}, "stability_threshold"),
        ({"decision": {"sufficiency_threshold": 0.95}}, "sufficiency_threshold"),
    ],
)
def test_out_of_range_routing_and_decision_settings_are_a_422(client, patch, expected):
    response = client.patch("/api/settings", json=patch)
    assert response.status_code == 422
    assert expected in response.json()["detail"]


def test_the_settings_api_rejects_a_decision_default_model_naming_the_roles(client):
    response = client.patch("/api/settings", json={"models": {"default_model": JEV, "role_models": {"critic": JEV}}})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "Predictor" in detail and "default model" in detail and "critic" in detail
    # Nothing was stored.
    assert client.get("/api/settings").json()["config"]["models"]["default_model"] == DEFAULT_MODEL


def test_the_decision_section_patches_like_any_other(client):
    body = client.patch(
        "/api/settings",
        json={
            "models": {"role_models": {"predictor": JEV}},
            "engine": {"orchestration_mode": "always"},
            "decision": {"choice_orders": 4},
        },
    ).json()
    assert body["config"]["decision"]["choice_orders"] == 4
    assert body["config"]["decision"]["provider"] == "openrouter"  # untouched
    assert body["config"]["engine"]["orchestration_mode"] == "always"
    assert body["effective"]["predictor_kind"] == "decision"
    assert body["effective"]["orchestration_mode"] == "always"
    assert body["effective"]["decision"]["choice_orders"] == 4


def test_patch_and_run_overrides_carry_a_decision_section():
    assert ConfigPatch(decision={"score_levels": 5}).decision == {"score_levels": 5}
    merged = engine_bridge.merge_overrides(
        DashboardConfig(),
        RunOverrides(decision={"provider": "typesafe"}, engine={"orchestration_mode": "never"}),
    )
    assert merged.decision.provider == "typesafe"
    assert merged.engine.orchestration_mode == "never"


def test_a_run_override_cannot_smuggle_a_decision_model_into_another_role():
    with pytest.raises(ValidationError):
        engine_bridge.merge_overrides(DashboardConfig(), RunOverrides(models={"role_models": {"integrator": JEV}}))


# --- stored configuration ------------------------------------------------------


def test_a_stored_decision_default_model_is_moved_to_the_predictor():
    config_store.config_file().write_text(
        json.dumps({"models": {"default_model": JEV, "role_models": {"critic": "typesafe/jev-latest"}}})
    )
    config = config_store.load_config(refresh=True)
    assert config.models.default_model == DEFAULT_MODEL
    assert config.models.role_models.predictor == JEV
    assert config.models.role_models.critic == ""
    notices = " ".join(config_store.load_notices())
    assert "only serve the Predictor" in notices


def test_the_typesafe_key_reaches_the_worker_and_is_never_returned(client):
    secret = "ts-test-0123456789abcdef"
    body = client.put("/api/settings/credentials", json={"provider": "typesafe", "api_key": secret}).json()
    assert body["credential"]["configured"] is True
    assert secret not in json.dumps(body)
    assert client.get("/api/settings").json()["credentials"]["typesafe"]["source"] == "stored"
    assert secret not in client.get("/api/settings").text
    env = config_store.worker_environment(DashboardConfig())
    assert env["TYPESAFE_API_KEY"] == secret


# --- engine handoff ------------------------------------------------------------


@pytest.fixture
def engine_settings(monkeypatch):
    """Apply a config to a fresh settings singleton, and restore it afterwards."""
    from src.full_stack.backend.config.settings import reload_settings

    monkeypatch.setattr(engine_bridge, "_catalog_context_window", lambda model_id: 1_048_576)
    reload_settings()
    yield engine_bridge.apply_config_to_settings
    reload_settings()


def test_orchestration_and_decision_settings_reach_the_engine(engine_settings):
    config = DashboardConfig.model_validate(
        {
            "models": {"role_models": {"predictor": JEV}},
            "engine": {"orchestration_mode": "never", "orchestration_threshold_tokens": 12_000},
            "decision": {
                "provider": "typesafe",
                "choice_orders": 5,
                "score_levels": 6,
                "regression_refine": False,
                "stability_threshold": 0.3,
                "sufficiency_threshold": 0.4,
                "compiler_model": " vendor/compiler ",
            },
        }
    )
    settings = engine_settings(config)
    assert settings.orchestration.mode == "never"
    assert settings.orchestration.threshold_tokens == 12_000
    assert settings.decision.provider == "typesafe"
    assert settings.decision.choice_orders == 5
    assert settings.decision.score_levels == 6
    assert settings.decision.regression_refine is False
    assert settings.decision.stability_threshold == pytest.approx(0.3)
    assert settings.decision.sufficiency_threshold == pytest.approx(0.4)
    assert settings.decision.compiler_model == "vendor/compiler"
    assert settings.models.predictor_model == JEV
    assert settings.models.orchestrator_model == DEFAULT_MODEL


def test_the_local_backend_keeps_a_decision_predictor(engine_settings):
    config = DashboardConfig.model_validate(
        {"connection": {"backend": "local"}, "models": {"role_models": {"predictor": JEV, "critic": "vendor/x"}}}
    )
    settings = engine_settings(config)
    assert settings.models.predictor_model == JEV
    # Every other role still runs on the local weights.
    assert settings.models.critic_model == config.local.model_name
    assert settings.models.orchestrator_model == config.local.model_name


def test_the_local_backend_still_replaces_a_conventional_predictor(engine_settings):
    config = DashboardConfig.model_validate(
        {"connection": {"backend": "local"}, "models": {"role_models": {"predictor": "vendor/remote"}}}
    )
    assert engine_settings(config).models.predictor_model == config.local.model_name


def test_the_defaults_leave_routing_on_auto(engine_settings):
    settings = engine_settings(DashboardConfig())
    assert settings.orchestration.mode == "auto"
    assert settings.orchestration.threshold_tokens == 0


# --- run blockers --------------------------------------------------------------


def test_a_typesafe_route_without_its_key_blocks_the_run(monkeypatch):
    _with_keys(monkeypatch, openrouter="sk-or")
    config = DashboardConfig.model_validate(
        {"models": {"role_models": {"predictor": JEV}}, "decision": {"provider": "typesafe"}}
    )
    problems = engine_bridge.run_blockers(config)
    assert len(problems) == 1 and "TypeSafe" in problems[0]
    _with_keys(monkeypatch, openrouter="sk-or", typesafe="ts")
    assert engine_bridge.run_blockers(config) == []


def test_a_decision_predictor_on_the_local_backend_needs_an_openrouter_key(monkeypatch):
    _with_keys(monkeypatch)
    config = DashboardConfig.model_validate(
        {"connection": {"backend": "local"}, "models": {"role_models": {"predictor": JEV}}}
    )
    problems = engine_bridge.run_blockers(config)
    assert len(problems) == 1 and "OpenRouter" in problems[0]
    # Without a decision model the Local backend needs no key at all.
    assert engine_bridge.run_blockers(DashboardConfig.model_validate({"connection": {"backend": "local"}})) == []


def test_a_missing_openrouter_key_is_reported_once(monkeypatch):
    _with_keys(monkeypatch)
    config = DashboardConfig.model_validate({"models": {"role_models": {"predictor": JEV}}})
    problems = engine_bridge.run_blockers(config)
    assert len(problems) == 1 and "OpenRouter key" in problems[0]


def test_a_decision_model_set_in_place_for_another_role_blocks_the_run(monkeypatch):
    """Assignment skips validation, so the service checks again before it spawns a worker."""
    _with_keys(monkeypatch, openrouter="sk-or")
    config = DashboardConfig()
    config.models.role_models.communicator = JEV
    problems = engine_bridge.run_blockers(config)
    assert any("communicator" in p and "Predictor" in p for p in problems)
    with pytest.raises(engine_bridge.ConfigurationProblem):
        engine_bridge.ensure_runnable(config)
    # A structural audit calls no model, so nothing blocks it.
    assert engine_bridge.run_blockers(config, audit=True) == []


def test_the_effective_snapshot_reports_routing_and_the_decision_section():
    config = DashboardConfig.model_validate(
        {"models": {"role_models": {"predictor": JEV}}, "engine": {"orchestration_threshold_tokens": 9_000}}
    )
    snapshot = engine_bridge.effective_settings_snapshot(config)
    assert snapshot["predictor_kind"] == "decision"
    assert snapshot["models"]["predictor"] == JEV
    assert snapshot["orchestration_mode"] == "auto"
    assert snapshot["orchestration_threshold_tokens"] == 9_000
    # A blank compiler resolves to the model that will actually write the question book.
    assert snapshot["decision"]["compiler_model"] == DEFAULT_MODEL
    assert snapshot["decision"]["choice_orders"] == 3
    assert engine_bridge.effective_settings_snapshot(DashboardConfig())["predictor_kind"] == "llm"


# --- task specification ----------------------------------------------------------


def test_output_scales_and_class_definitions_reach_the_engine_task():
    from src.full_stack.backend.api.schemas import TaskSpecInput

    task = TaskSpecInput.model_validate(
        {
            "prediction_type": "hierarchical",
            "root": {
                "node_id": "root",
                "display_name": "Profile",
                "mode": "univariate_regression",
                "regression_outputs": ["madrs_total"],
                "output_scales": {"madrs_total": {"min": 0, "max": 60, "integer": True}},
                "children": [
                    {
                        "node_id": "dx",
                        "display_name": "Diagnosis",
                        "mode": "binary_classification",
                        "class_labels": ["CASE", "CONTROL"],
                        "class_definitions": {"CASE": "Meets criteria."},
                    }
                ],
            },
        }
    )
    spec = engine_bridge.build_task_spec(task)
    assert spec.root.output_scales == {"madrs_total": {"min": 0, "max": 60, "integer": True}}
    assert spec.root.children[0].class_definitions == {"CASE": "Meets criteria."}
