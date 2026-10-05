"""The cost projection follows the evidence route and prices decision models input-only."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.api import catalog as catalog_module
from src.full_stack.backend.api import config_store
from src.full_stack.backend.api import cost as cost_module
from src.full_stack.backend.api.schemas import DashboardConfig

REPO = Path(__file__).resolve().parent.parent.parent
SUBJ_001 = REPO / "src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO"

DEFAULT = "deepseek/deepseek-v4-flash-0731"
JEV = "typesafe/jev-1.13"
PRICE = 0.065
COMPLETION_PRICE = 0.18
# int(32000 / 1.2) state tokens, minus what the longest question needs.
DECISION_BUDGET = 26_666 - cost_module.DECISION_STATE_QUESTION_RESERVE


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path))
    config_store._cached = None
    catalog_module._memory.clear()
    yield
    config_store._cached = None
    catalog_module._memory.clear()


@pytest.fixture
def prices(monkeypatch, tmp_path):
    """The provider catalog without any decision model, as it may well be."""
    index = {
        DEFAULT: {
            "prompt_usd_per_mtok": PRICE,
            "completion_usd_per_mtok": COMPLETION_PRICE,
            "context_length": 1_310_720,
        }
    }
    monkeypatch.setattr(cost_module, "pricing_index", lambda cached_only=False: index)
    # The engine reads context windows from the catalog cache the dashboard writes.
    (tmp_path / "model_catalog.json").write_text(
        json.dumps({"models": [{"id": DEFAULT, "context_length": 1_310_720}], "fetched_at": 0})
    )
    return index


def _decision_config(**engine) -> DashboardConfig:
    config = DashboardConfig()
    config.models.role_models.predictor = JEV
    for key, value in engine.items():
        setattr(config.engine, key, value)
    return config


def _line(estimate, role):
    return next(line for line in estimate["lines"] if line["role"] == role)


def _roles(estimate):
    return [line["role"] for line in estimate["lines"]]


# --- route resolution ----------------------------------------------------------


def test_a_record_that_fits_the_llm_predictor_is_projected_direct(prices):
    estimate = cost_module.estimate_run(
        config=DashboardConfig(), input_tokens=8680, iterations=1, include_deep_report=False
    )
    assert estimate["route"] == "direct"
    assert estimate["predictor_kind"] == "llm"
    # No Orchestrator and no tool steps on the direct route.
    assert _roles(estimate) == ["predictor", "critic"]
    predictor = _line(estimate, "predictor")
    profile = cost_module.AGENT_PROFILES["predictor"]
    assert predictor["prompt_tokens"] == profile.prompt_base + 8680
    assert predictor["completion_tokens"] == profile.completion_base
    assert estimate["plan_steps"] == 0
    assert estimate["fully_priced"] is True


def test_the_direct_predictor_prompt_does_not_scale_with_plan_size(prices):
    small = cost_module.estimate_run(config=DashboardConfig(), input_tokens=8680, iterations=1, plan_steps=4)
    large = cost_module.estimate_run(config=DashboardConfig(), input_tokens=8680, iterations=1, plan_steps=28)
    assert small["route"] == large["route"] == "direct"
    assert small["total_tokens"] == large["total_tokens"]


def test_the_llm_budget_is_the_context_minus_the_output_reserve_capped_by_the_input_limit(prices):
    config = DashboardConfig()
    # 1,310,720 tokens at 0.95 is the derived agent input limit.
    assert cost_module.predictor_input_budget(config, "llm", prices) == int(1_310_720 * 0.95)
    config.token_budget.max_agent_input_tokens = 50_000
    assert cost_module.predictor_input_budget(config, "llm", prices) == 50_000
    config.models.role_context_windows.predictor = 32_000
    config.token_budget.max_agent_input_tokens = 0
    # 32,000 minus the 8,192 reserve for a 12,000-token Predictor ceiling.
    assert cost_module.predictor_input_budget(config, "llm", prices) == 32_000 - 8_192
    # The configured window only covers models the catalog and the table do not know.
    config.models.role_context_windows.predictor = 0
    config.models.context_window = 32_000
    assert cost_module.predictor_input_budget(config, "llm", prices) == int(1_310_720 * 0.95)


def test_a_record_larger_than_the_budget_is_projected_orchestrated(prices):
    config = DashboardConfig()
    config.token_budget.max_agent_input_tokens = 5_000
    estimate = cost_module.estimate_run(config=config, input_tokens=8680, iterations=1, include_deep_report=False)
    assert estimate["route"] == "orchestrated"
    assert _roles(estimate)[:2] == ["orchestrator", "tool"]
    explicit = cost_module.estimate_run(
        config=config, input_tokens=8680, iterations=1, include_deep_report=False, route="orchestrated"
    )
    assert estimate["total_tokens"] == explicit["total_tokens"]


def test_the_configured_mode_and_threshold_decide_the_auto_route(prices):
    always = DashboardConfig()
    always.engine.orchestration_mode = "always"
    assert cost_module.estimate_run(config=always, input_tokens=10, iterations=1)["route"] == "orchestrated"

    never = DashboardConfig()
    never.engine.orchestration_mode = "never"
    assert cost_module.estimate_run(config=never, input_tokens=10**8, iterations=1)["route"] == "direct"

    lowered = DashboardConfig()
    lowered.engine.orchestration_threshold_tokens = 5_000
    assert cost_module.estimate_run(config=lowered, input_tokens=8680, iterations=1)["route"] == "orchestrated"
    assert cost_module.estimate_run(config=lowered, input_tokens=4_000, iterations=1)["route"] == "direct"


def test_an_explicit_route_wins_over_the_mode(prices):
    config = DashboardConfig()
    config.engine.orchestration_mode = "always"
    assert cost_module.estimate_run(config=config, input_tokens=10, route="direct")["route"] == "direct"


# --- decision models -----------------------------------------------------------


def test_a_decision_predictor_on_the_direct_route_is_priced_input_only(prices):
    estimate = cost_module.estimate_run(
        config=_decision_config(), input_tokens=8680, iterations=1, include_deep_report=False
    )
    assert estimate["route"] == "direct"
    assert estimate["predictor_kind"] == "decision"
    assert estimate["predictor_budget_tokens"] == DECISION_BUDGET

    predictor = _line(estimate, "predictor")
    assert predictor["model"] == JEV
    assert predictor["calls"] == cost_module.DECISION_REQUESTS_PER_ATTEMPT
    assert predictor["prompt_tokens"] == 2 * (8680 + cost_module.DECISION_QUESTION_TOKENS)
    assert predictor["completion_tokens"] == 0
    assert predictor["completion_usd_per_mtok"] == 0
    assert predictor["usd"] == pytest.approx(predictor["prompt_tokens"] / 1_000_000 * 0.042, abs=1e-6)

    book = _line(estimate, "question_book")
    assert book["model"] == DEFAULT  # the Orchestrator's model compiles it
    assert (book["calls"], book["prompt_tokens"], book["completion_tokens"]) == (1, 4_000, 1_600)

    # The decision critic calls no model, and nothing was left unpriced.
    assert "critic" not in _roles(estimate)
    assert estimate["fully_priced"] is True
    assert estimate["usd"] is not None


def test_the_question_book_uses_the_configured_compiler(prices):
    config = _decision_config()
    config.decision.compiler_model = "vendor/compiler"
    estimate = cost_module.estimate_run(config=config, input_tokens=100, iterations=1, include_deep_report=False)
    assert _line(estimate, "question_book")["model"] == "vendor/compiler"


def test_a_record_over_the_state_budget_orchestrates_and_packs_to_the_budget(prices):
    estimate = cost_module.estimate_run(
        config=_decision_config(), input_tokens=50_000, iterations=2, include_deep_report=False
    )
    assert estimate["route"] == "orchestrated"
    assert {"orchestrator", "tool", "predictor", "question_book"} <= set(_roles(estimate))
    predictor = _line(estimate, "predictor")
    assert predictor["calls"] == 2 * cost_module.DECISION_REQUESTS_PER_ATTEMPT
    assert predictor["prompt_tokens"] == 4 * (DECISION_BUDGET + cost_module.DECISION_QUESTION_TOKENS)
    assert predictor["completion_tokens"] == 0


def test_the_kind_can_be_forced_for_a_what_if_estimate(prices):
    estimate = cost_module.estimate_run(config=DashboardConfig(), input_tokens=100, predictor_kind="decision")
    assert estimate["predictor_kind"] == "decision"


def test_decision_spend_is_never_unpriced(monkeypatch):
    """The worker tracks decision calls under the model id; the catalog may not list it."""
    monkeypatch.setattr(cost_module, "pricing_index", lambda cached_only=False: {})
    actual = cost_module.actual_cost_from_usage({JEV: {"prompt": 1_000_000, "completion": 37}}, cached_only=True)
    assert actual["fully_priced"] is True
    assert actual["usd"] == pytest.approx(0.042)
    assert actual["lines"][0]["completion_usd_per_mtok"] == 0


def test_a_provider_row_without_a_completion_price_is_still_priced(monkeypatch):
    monkeypatch.setattr(
        cost_module,
        "pricing_index",
        lambda cached_only=False: {JEV: {"prompt_usd_per_mtok": 0.05, "completion_usd_per_mtok": None}},
    )
    actual = cost_module.actual_cost_from_usage({JEV: {"prompt": 1_000_000, "completion": 10}})
    assert actual["fully_priced"] is True
    assert actual["usd"] == pytest.approx(0.05)


def test_the_hot_path_index_carries_the_registry_without_a_cache():
    index = catalog_module.pricing_index(cached_only=True)
    assert index[JEV]["completion_usd_per_mtok"] == 0


# --- the estimate endpoint ------------------------------------------------------


@pytest.fixture
def client(monkeypatch, prices):
    from src.full_stack.backend.api.app import create_app

    return TestClient(create_app())


def test_the_estimate_endpoint_reports_the_route_and_predictor_kind(client):
    body = client.post(
        "/api/runs/estimate",
        json={
            "participant_dirs": [str(SUBJ_001)],
            "task": {"prediction_type": "binary", "target_label": "T", "control_label": "C"},
        },
    ).json()
    assert body["route"] == "direct"
    assert body["predictor_kind"] == "llm"
    assert body["participants"][0]["route"] == "direct"
    assert set(body["effective_models"]) == {
        "orchestrator", "integrator", "predictor", "critic", "communicator", "tool"
    }


def test_the_estimate_endpoint_prices_a_decision_predictor_override(client):
    body = client.post(
        "/api/runs/estimate",
        json={
            "participant_dirs": [str(SUBJ_001)],
            "task": {"prediction_type": "binary", "target_label": "T", "control_label": "C"},
            "overrides": {"models": {"role_models": {"predictor": JEV}}, "decision": {"choice_orders": 2}},
        },
    ).json()
    assert body["predictor_kind"] == "decision"
    assert body["effective_models"]["predictor"] == JEV
    line = body["participants"][0]
    assert line["predictor_kind"] == "decision"
    assert any(row["role"] == "question_book" for row in line["lines"])
    assert body["totals"]["fully_priced"] is True


def test_the_estimate_endpoint_says_mixed_when_participants_take_different_routes(client, monkeypatch, tmp_path):
    sizes = {"small": 1_000, "large": 10**8}
    for name in sizes:
        (tmp_path / name).mkdir()
    monkeypatch.setattr(
        "src.full_stack.backend.api.routers.runs_router.participant_input_tokens",
        lambda directory: sizes.get(Path(directory).name, 0),
    )
    body = client.post(
        "/api/runs/estimate", json={"participant_dirs": [str(tmp_path / name) for name in sizes]}
    ).json()
    assert [line["route"] for line in body["participants"]] == ["direct", "orchestrated"]
    assert body["route"] == "mixed"


def test_an_empty_estimate_has_no_route(client):
    body = client.post("/api/runs/estimate", json={"participant_dirs": []}).json()
    assert body["route"] is None


def test_a_performance_report_attributes_decision_requests_to_their_model():
    """The engine's ledger names a decision model only in the tool field."""
    from src.full_stack.backend.api.reports import usage_by_model_from_report

    report = {
        "token_usage": {
            "calls": [
                {"component": "orchestrator", "prompt_tokens": 100, "completion_tokens": 10},
                {
                    "component": "predictor",
                    "prompt_tokens": 30_000,
                    "completion_tokens": 0,
                    "step_tool": f"DecisionModel:{JEV}:coarse",
                },
                {
                    "component": "predictor",
                    "prompt_tokens": 4_000,
                    "completion_tokens": 1_600,
                    "step_tool": "QuestionBookCompiler",
                },
            ]
        }
    }
    usage = usage_by_model_from_report(report, DEFAULT)
    assert usage[JEV] == {"prompt": 30_000, "completion": 0, "calls": 1}
    assert usage[DEFAULT]["calls"] == 2
