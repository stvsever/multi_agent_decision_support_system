"""
Parity between the engine and the dashboard cost estimate.

For the bundled pseudo participants, several budgets and both Predictor kinds
(a conventional LLM and the structured decision model TypeSafe Jev), the engine
side runs what a dashboard run worker runs before its first attempt: the
configuration stamped onto the engine settings singleton, then the pipeline's
own DataLoader, direct executor output, Predictor `measure_direct` and route
decision. The dashboard side is the Studio estimate (`cost.measure_participant`
plus `cost.estimate_run`, and once through the HTTP endpoint). Both must agree on
the route, the measured Predictor input and the budget, and the projected
Predictor prompt on the direct route must be the measured one. The engine side
is `main.run_compass_pipeline` itself, stopped right after its route decision.

No model is called and the network is blocked. The decision model's question
book is the deterministic one on both sides (a run compiles a book of the same
shape once per task with the companion LLM).
"""

import contextlib
import io
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import main as compass_main
from src.full_stack.backend.agents.decision.predictor import DecisionPredictor
from src.full_stack.backend.agents.decision.questions import deterministic_book
from src.full_stack.backend.agents.predictor import Predictor
from src.full_stack.backend.api import catalog as catalog_module
from src.full_stack.backend.api import config_store
from src.full_stack.backend.api import cost as cost_module
from src.full_stack.backend.api import engine_bridge
from src.full_stack.backend.api.engine_bridge import apply_config_to_settings, build_task_spec, normalise_instructions
from src.full_stack.backend.api.schemas import DashboardConfig, TaskSpecInput
from src.full_stack.backend.config import settings as settings_module
from src.full_stack.backend.config.settings import reload_settings


REPO = Path(__file__).resolve().parent.parent.parent
PSEUDO = sorted((REPO / "src/full_stack/backend/data/pseudo_data/inputs").glob("SUBJ_*_PSEUDO"))
DEFAULT = "deepseek/deepseek-v4-flash-0731"
JEV = "typesafe/jev-1.13"

BINARY = TaskSpecInput(prediction_type="binary", target_label="CASE", control_label="CONTROL")
REGRESSION = TaskSpecInput(
    prediction_type="regression_univariate",
    target_label="total_iq",
    regression_outputs=["total_iq"],
    output_scales={"total_iq": {"min": 40, "max": 160}},
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    def _blocked(*_args, **_kwargs):
        raise RuntimeError("network access is not allowed in these tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path))
    monkeypatch.setattr(settings_module, "_CATALOG_CONTEXT_CACHE", {})
    (tmp_path / "model_catalog.json").write_text(
        json.dumps({"models": [{"id": DEFAULT, "context_length": 1_310_720}], "fetched_at": 0})
    )
    index = {DEFAULT: {"prompt_usd_per_mtok": 0.065, "completion_usd_per_mtok": 0.18, "context_length": 1_310_720}}
    monkeypatch.setattr(cost_module, "pricing_index", lambda cached_only=False: index)
    monkeypatch.setattr(engine_bridge, "_catalog_context_window", lambda model_id: None)
    config_store._cached = None
    catalog_module._memory.clear()
    cost_module._MEASURE_CACHE.clear()
    yield
    config_store._cached = None
    catalog_module._memory.clear()
    cost_module._MEASURE_CACHE.clear()
    reload_settings()


class _Stop(Exception):
    """Raised right after the pipeline's route decision: nothing past it is needed."""


class _StubAgent:
    def __init__(self, *_args, **_kwargs):
        self.integrator = SimpleNamespace(set_runtime_instruction=lambda _text: None)

    def set_runtime_instruction(self, _text):
        return None


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """
    Run main.run_compass_pipeline up to its route decision, as a dashboard run
    worker does, and return the measurement and the decision it made. The real
    Predictor (or DecisionPredictor) measures the record; every other agent and
    the provider client are stubs, and the pipeline stops right after routing.
    """
    from src.full_stack.backend.agents import base_agent

    offline_client = SimpleNamespace(ping=lambda: None, token_tracker=SimpleNamespace(add=lambda *a, **k: None))
    monkeypatch.setattr(compass_main, "get_llm_client", lambda: offline_client)
    monkeypatch.setattr(base_agent, "get_llm_client", lambda: offline_client)
    for name in ("Orchestrator", "Executor", "Critic", "Communicator"):
        monkeypatch.setattr(compass_main, name, _StubAgent)
    monkeypatch.setattr(
        DecisionPredictor, "question_book", lambda self, spec: deterministic_book(spec, self.task_context())
    )
    captured = {}
    real_decide_route = compass_main.decide_route

    def _capture(**kwargs):
        captured["measurement"] = kwargs["measurement"]
        captured["decision"] = real_decide_route(**kwargs)
        raise _Stop()

    monkeypatch.setattr(compass_main, "decide_route", _capture)

    def _measure(config: DashboardConfig, directory: Path, task: TaskSpecInput):
        captured.clear()
        reload_settings()
        settings = apply_config_to_settings(config)
        settings.paths.logs_dir = tmp_path / "logs"
        settings.paths.logs_dir.mkdir(exist_ok=True)
        task_spec = build_task_spec(task)
        target, control = task_spec.legacy_target_control()
        with pytest.raises(_Stop), contextlib.redirect_stdout(io.StringIO()):
            compass_main.run_compass_pipeline(
                participant_dir=directory,
                target_condition=target,
                control_condition=control,
                prediction_task_spec=task_spec,
                agent_instructions=normalise_instructions(config.instructions),
                max_iterations=2,
                verbose=False,
                orchestration_mode=config.engine.orchestration_mode,
                orchestration_threshold_tokens=config.engine.orchestration_threshold_tokens,
            )
        assert captured["measurement"] is not None, "the pipeline could not measure the record"
        return captured["measurement"], captured["decision"]

    return _measure


def _config(kind: str, *, max_agent_input: int = 0, threshold: int = 0, predictor_window: int = 0, mode: str = "auto"):
    config = DashboardConfig()
    config.instructions = {"global": "Research use only.", "predictor": "Weigh cognition first."}
    config.token_budget.max_agent_input_tokens = max_agent_input
    config.engine.orchestration_threshold_tokens = threshold
    config.engine.orchestration_mode = mode
    config.models.role_context_windows.predictor = predictor_window
    if kind == "decision":
        config.models.role_models.predictor = JEV
    return config


CASES = [
    # (kind, task, max_agent_input, threshold, predictor_window, mode)
    ("llm", BINARY, 0, 0, 0, "auto"),
    ("llm", BINARY, 6_000, 0, 0, "auto"),
    ("llm", BINARY, 3_000, 0, 0, "auto"),
    ("llm", BINARY, 0, 7_000, 0, "auto"),
    ("llm", BINARY, 0, 0, 16_000, "auto"),
    ("llm", REGRESSION, 200_000, 0, 0, "auto"),
    ("llm", BINARY, 3_000, 0, 0, "never"),
    ("decision", BINARY, 0, 0, 0, "auto"),
    ("decision", BINARY, 0, 2_500, 0, "auto"),
    ("decision", REGRESSION, 0, 0, 0, "auto"),
    ("decision", REGRESSION, 0, 0, 0, "always"),
]


@pytest.mark.parametrize("kind, task, max_agent_input, threshold, predictor_window, mode", CASES)
def test_the_dashboard_predicts_the_engine_route_and_tokens(engine, kind, task, max_agent_input, threshold, predictor_window, mode):
    config = _config(
        kind, max_agent_input=max_agent_input, threshold=threshold, predictor_window=predictor_window, mode=mode
    )
    routes = set()
    for directory in PSEUDO:
        measured_by_engine, engine_route = engine(config, directory, task)
        reload_settings()  # the dashboard side must not lean on the stamped singleton

        measured = cost_module.measure_participant(config, directory, build_task_spec(task))
        assert measured is not None, directory.name
        assert measured["input_tokens"] == measured_by_engine["input_tokens"], directory.name
        assert measured["budget_tokens"] == measured_by_engine["budget_tokens"], directory.name
        assert measured["route"] == engine_route.route, directory.name
        assert measured["threshold_tokens"] == engine_route.budget_tokens, directory.name

        estimate = cost_module.estimate_run(
            config=config,
            input_tokens=cost_module.participant_input_tokens(directory),
            iterations=2,
            include_deep_report=False,
            measurement=measured,
        )
        assert estimate["measured"] is True
        assert estimate["route"] == engine_route.route
        assert estimate["predictor_kind"] == kind
        assert estimate["predictor_budget_tokens"] == measured_by_engine["budget_tokens"]
        assert estimate["predictor_input_tokens"] == measured_by_engine["input_tokens"]
        predictor_line = next(line for line in estimate["lines"] if line["role"] == "predictor")
        if engine_route.route == "direct" and kind == "llm":
            assert predictor_line["prompt_tokens"] == 2 * measured_by_engine["input_tokens"]
        if engine_route.route == "direct" and kind == "decision":
            state = min(measured_by_engine["input_tokens"], measured_by_engine["budget_tokens"])
            questions = list(measured["question_tokens_per_request"])
            if measured["round_two_expected"]:
                questions = questions + questions
            assert predictor_line["prompt_tokens"] == 2 * sum(state + q for q in questions)
            assert predictor_line["completion_tokens"] == 0
        routes.add(engine_route.route)
    if mode == "always":
        assert routes == {"orchestrated"}
    if mode == "never":
        assert routes == {"direct"}


def test_the_budgets_split_the_pseudo_cohort():
    """The cases above are only meaningful if a budget puts records on both routes."""
    config = _config("llm", max_agent_input=6_000)
    routes = {cost_module.measure_participant(config, d, build_task_spec(BINARY))["route"] for d in PSEUDO}
    assert routes == {"direct", "orchestrated"}


def test_the_budget_without_a_record_is_the_engine_budget():
    for kind in ("llm", "decision"):
        for max_input in (0, 5_000):
            config = _config(kind, max_agent_input=max_input)
            reload_settings()
            settings = apply_config_to_settings(config)
            if kind == "llm":
                expected = Predictor(llm_client=SimpleNamespace()).input_budget_tokens()
                assert cost_module.predictor_input_budget(config, "llm") == expected
            else:
                spec_state = int(32_000 / float(settings.decision.tokenizer_ratio))
                assert cost_module.predictor_input_budget(config, "decision") == (
                    spec_state - cost_module.DECISION_STATE_QUESTION_RESERVE
                )


def test_the_estimate_endpoint_carries_the_engine_measurement(engine):
    from fastapi.testclient import TestClient

    from src.full_stack.backend.api.app import create_app

    client = TestClient(create_app())
    for kind, overrides in (
        ("llm", {"token_budget": {"max_agent_input_tokens": 6_000}}),
        ("decision", {"models": {"role_models": {"predictor": JEV}}, "engine": {"orchestration_threshold_tokens": 2_500}}),
    ):
        body = client.post(
            "/api/runs/estimate",
            json={
                "participant_dirs": [str(d) for d in PSEUDO],
                "task": BINARY.model_dump(),
                "overrides": overrides,
            },
        ).json()
        config = _config(
            kind,
            max_agent_input=6_000 if kind == "llm" else 0,
            threshold=2_500 if kind == "decision" else 0,
        )
        config.instructions = {}
        for line, directory in zip(body["participants"], PSEUDO):
            measured_by_engine, route = engine(config, directory, BINARY)
            assert line["measured"] is True
            assert line["route"] == route.route, directory.name
            assert line["predictor_input_tokens"] == measured_by_engine["input_tokens"], directory.name
            assert line["predictor_budget_tokens"] == measured_by_engine["budget_tokens"], directory.name


def test_the_settings_screen_shows_the_engine_windows():
    from fastapi.testclient import TestClient

    from src.full_stack.backend.api.app import create_app

    client = TestClient(create_app())
    effective = client.get("/api/settings").json()["effective"]
    windows = effective["context_windows"]
    assert windows["predictor"] == {"tokens": 1_310_720, "source": "catalog", "model": DEFAULT}
    assert effective["predictor_input_budget"] == int(1_310_720 * 0.95)

    client.patch("/api/settings", json={"models": {"role_context_windows": {"tool": 32_768}}})
    windows = client.get("/api/settings").json()["effective"]["context_windows"]
    assert windows["tool"]["tokens"] == 32_768 and windows["tool"]["source"] == "role_override"
    assert windows["critic"]["source"] == "catalog"
