"""
CLI choice of the Predictor model.

A structured decision model (TypeSafe Jev) can only serve the Predictor; every
other role needs a conventional companion LLM. These tests drive
`main._apply_predictor_choice` with an argparse Namespace, the role guard
`enforce_role_models`, and the full argument parser of `main.main` with the
pipeline itself replaced by a stub. Nothing is run and no model is called.
"""

import argparse
import builtins
import io
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import main as main_mod
from main import DEFAULT_COMPANION_MODEL, _apply_predictor_choice
from src.full_stack.backend.config.settings import LLMBackend, get_settings, reload_settings
from src.full_stack.backend.agents.decision import (
    NON_PREDICTOR_ROLES,
    decision_role_conflicts,
    enforce_role_models,
)

REPO = Path(__file__).resolve().parents[2]
SUBJ_001 = REPO / "src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO"
JEV = "typesafe/jev-1.13"
LLM = "deepseek/deepseek-v4-flash-0731"
OTHER_LLM = "openai/gpt-5-mini"
ROLES = ("orchestrator", "critic", "integrator", "communicator", "tool")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    def _blocked(*_args, **_kwargs):
        raise RuntimeError("network access is not allowed in these tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    yield
    reload_settings()


class _Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def no_terminal(monkeypatch):
    """stdin is not a terminal, and any prompt fails the test."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    def _no_prompt(*_args, **_kwargs):
        raise AssertionError("must not prompt without a terminal")

    monkeypatch.setattr(builtins, "input", _no_prompt)


@pytest.fixture
def terminal(monkeypatch):
    """stdin is a terminal; `answers` feeds input(), `prompts` records what was asked."""
    monkeypatch.setattr(sys, "stdin", _Terminal(""))
    state = {"answers": [], "prompts": []}

    def _input(prompt=""):
        state["prompts"].append(prompt)
        answer = state["answers"].pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(builtins, "input", _input)
    return state


def _args(**overrides) -> argparse.Namespace:
    values = dict(
        public_model=LLM,
        predictor_model=None,
        companion_model=None,
        orchestration=None,
        orchestration_threshold=None,
        decision_provider=None,
        decision_choice_orders=None,
        decision_score_levels=None,
        decision_refine=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _cli_settings(public: str):
    """Settings as main() leaves them before the Predictor choice: every role on --public_model."""
    s = reload_settings()
    s.models.backend = LLMBackend.OPENROUTER
    s.models.public_model_name = public
    for role in ROLES + ("predictor",):
        setattr(s.models, f"{role}_model", public)
    return s


def _roles(s):
    return {role: getattr(s.models, f"{role}_model") for role in ROLES}


# ---------------------------------------------------------------------------
# Decision model as --public_model
# ---------------------------------------------------------------------------
def test_decision_public_model_without_terminal_uses_the_default_companion(no_terminal, capsys):
    s = _cli_settings(JEV)
    _apply_predictor_choice(s, _args(public_model=JEV))
    assert s.models.predictor_model == JEV
    assert _roles(s) == {role: DEFAULT_COMPANION_MODEL for role in ROLES}
    assert s.models.public_model_name == DEFAULT_COMPANION_MODEL
    out = capsys.readouterr().out
    assert f"other roles use {DEFAULT_COMPANION_MODEL}" in out
    # The Predictor summary line is printed once, when the pipeline initializes its agents.
    assert "(structured decision model, state limit" not in out


def test_decision_public_model_asks_for_the_companion_on_a_terminal(terminal):
    terminal["answers"].append(OTHER_LLM)
    s = _cli_settings(JEV)
    _apply_predictor_choice(s, _args(public_model=JEV))
    assert len(terminal["prompts"]) == 1
    assert f"{JEV} is a structured decision model" in terminal["prompts"][0]
    assert DEFAULT_COMPANION_MODEL in terminal["prompts"][0]
    assert s.models.predictor_model == JEV
    assert _roles(s) == {role: OTHER_LLM for role in ROLES}


@pytest.mark.parametrize("answer", ["", "   ", EOFError()])
def test_empty_or_closed_terminal_answer_uses_the_default_companion(terminal, answer):
    terminal["answers"].append(answer)
    s = _cli_settings(JEV)
    _apply_predictor_choice(s, _args(public_model=JEV))
    assert _roles(s) == {role: DEFAULT_COMPANION_MODEL for role in ROLES}
    assert s.models.predictor_model == JEV


def test_given_companion_is_used_without_asking(terminal):
    s = _cli_settings(JEV)
    _apply_predictor_choice(s, _args(public_model=JEV, companion_model=OTHER_LLM))
    assert terminal["prompts"] == []
    assert _roles(s) == {role: OTHER_LLM for role in ROLES}
    assert s.models.public_model_name == OTHER_LLM
    assert s.models.predictor_model == JEV


def test_explicit_predictor_model_wins_over_the_decision_public_model(no_terminal):
    s = _cli_settings(JEV)
    _apply_predictor_choice(s, _args(public_model=JEV, predictor_model="~typesafe/jev-latest"))
    assert s.models.predictor_model == "~typesafe/jev-latest"
    assert _roles(s) == {role: DEFAULT_COMPANION_MODEL for role in ROLES}


@pytest.mark.parametrize("public", [JEV, LLM])
@pytest.mark.parametrize("companion", [JEV, "~typesafe/jev-latest", "jev-1.13"])
def test_a_decision_model_as_companion_is_refused(no_terminal, public, companion):
    s = _cli_settings(public)
    with pytest.raises(ValueError, match="--companion_model must be a conventional LLM"):
        _apply_predictor_choice(s, _args(public_model=public, companion_model=companion))


# ---------------------------------------------------------------------------
# Decision model as --predictor_model only
# ---------------------------------------------------------------------------
def test_decision_predictor_model_with_an_llm_public_model_keeps_the_other_roles(terminal):
    s = _cli_settings(LLM)
    _apply_predictor_choice(s, _args(public_model=LLM, predictor_model=JEV))
    assert terminal["prompts"] == []
    assert s.models.predictor_model == JEV
    assert _roles(s) == {role: LLM for role in ROLES}
    assert s.models.public_model_name == LLM


def test_decision_predictor_model_with_a_companion(no_terminal):
    s = _cli_settings(LLM)
    _apply_predictor_choice(s, _args(public_model=LLM, predictor_model=JEV, companion_model=OTHER_LLM))
    assert s.models.predictor_model == JEV
    assert _roles(s) == {role: OTHER_LLM for role in ROLES}
    assert s.models.public_model_name == OTHER_LLM


def test_llm_choices_are_left_alone(no_terminal):
    s = _cli_settings(LLM)
    _apply_predictor_choice(s, _args(public_model=LLM))
    assert s.models.predictor_model == LLM
    assert _roles(s) == {role: LLM for role in ROLES}

    s = _cli_settings(LLM)
    _apply_predictor_choice(s, _args(public_model=LLM, predictor_model=OTHER_LLM))
    assert s.models.predictor_model == OTHER_LLM
    assert _roles(s) == {role: LLM for role in ROLES}

    # A companion alone moves the other roles; an LLM Predictor stays where it is.
    s = _cli_settings(LLM)
    _apply_predictor_choice(s, _args(public_model=LLM, companion_model=OTHER_LLM))
    assert s.models.predictor_model == LLM
    assert _roles(s) == {role: OTHER_LLM for role in ROLES}


def test_local_backend_ignores_a_decision_public_model(no_terminal):
    s = reload_settings()
    s.models.backend = LLMBackend.LOCAL
    for role in ROLES + ("predictor",):
        setattr(s.models, f"{role}_model", "Qwen/Qwen3-14B-AWQ")
    _apply_predictor_choice(s, _args(public_model=JEV))
    assert s.models.predictor_model == "Qwen/Qwen3-14B-AWQ"
    assert _roles(s) == {role: "Qwen/Qwen3-14B-AWQ" for role in ROLES}


# ---------------------------------------------------------------------------
# Role guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", ROLES)
def test_enforce_role_models_rejects_a_decision_model_outside_the_predictor(role):
    s = _cli_settings(LLM)
    setattr(s.models, f"{role}_model", JEV)
    with pytest.raises(ValueError, match=f"Set a conventional LLM for: {role}"):
        enforce_role_models(s)


def test_enforce_role_models_accepts_a_decision_predictor():
    s = _cli_settings(LLM)
    s.models.predictor_model = JEV
    enforce_role_models(s)


def test_enforce_role_models_lists_every_conflict():
    s = _cli_settings(JEV)
    with pytest.raises(ValueError) as info:
        enforce_role_models(s)
    for role in ROLES:
        assert role in str(info.value)
    assert set(NON_PREDICTOR_ROLES) == set(ROLES)


def test_decision_role_conflicts():
    assert decision_role_conflicts({"critic": JEV, "orchestrator": LLM, "predictor": JEV}) == ["critic"]
    assert decision_role_conflicts({"tool": "typesafe/jev-router"}) == []
    assert decision_role_conflicts({}) == []
    assert decision_role_conflicts(None) == []


def test_apply_predictor_choice_rejects_a_leftover_decision_role(no_terminal):
    s = _cli_settings(LLM)
    s.models.critic_model = JEV
    with pytest.raises(ValueError, match="critic"):
        _apply_predictor_choice(s, _args(public_model=LLM))


# ---------------------------------------------------------------------------
# Routing and decision options
# ---------------------------------------------------------------------------
def test_routing_and_decision_options_are_applied(no_terminal):
    s = _cli_settings(LLM)
    _apply_predictor_choice(
        s,
        _args(
            predictor_model=JEV,
            orchestration="Force",
            orchestration_threshold=12_000,
            decision_provider="typesafe",
            decision_choice_orders=5,
            decision_score_levels=8,
            decision_refine=False,
        ),
    )
    assert s.orchestration.mode == "always"
    assert s.orchestration.threshold_tokens == 12_000
    assert s.decision.provider == "typesafe"
    assert s.decision.choice_orders == 5
    assert s.decision.score_levels == 8
    assert s.decision.regression_refine is False


def test_routing_and_decision_options_are_clamped(no_terminal):
    s = _cli_settings(LLM)
    _apply_predictor_choice(s, _args(orchestration_threshold=-5, decision_choice_orders=0, decision_score_levels=25))
    assert s.orchestration.threshold_tokens == 0
    assert s.decision.choice_orders == 1
    assert s.decision.score_levels == 10
    s = _cli_settings(LLM)
    _apply_predictor_choice(s, _args(decision_score_levels=1, decision_refine=True))
    assert s.decision.score_levels == 2
    assert s.decision.regression_refine is True


def test_unset_options_keep_the_settings_defaults(no_terminal):
    s = _cli_settings(LLM)
    before = (s.orchestration.mode, s.orchestration.threshold_tokens, s.decision.provider, s.decision.choice_orders, s.decision.score_levels, s.decision.regression_refine)
    _apply_predictor_choice(s, _args())
    after = (s.orchestration.mode, s.orchestration.threshold_tokens, s.decision.provider, s.decision.choice_orders, s.decision.score_levels, s.decision.regression_refine)
    assert after == before


def test_unknown_orchestration_mode_is_refused(no_terminal):
    s = _cli_settings(LLM)
    with pytest.raises(ValueError, match="orchestration mode"):
        _apply_predictor_choice(s, _args(orchestration="sometimes"))


# ---------------------------------------------------------------------------
# The full command line, with the pipeline replaced by a stub
# ---------------------------------------------------------------------------
def _run_main(monkeypatch, argv):
    captured = {}

    def _pipeline(**kwargs):
        s = get_settings()
        captured["kwargs"] = kwargs
        captured["models"] = {role: getattr(s.models, f"{role}_model") for role in ROLES + ("predictor",)}
        captured["public_model_name"] = s.models.public_model_name
        captured["orchestration"] = (s.orchestration.mode, s.orchestration.threshold_tokens)
        captured["decision"] = (s.decision.provider, s.decision.choice_orders, s.decision.score_levels, s.decision.regression_refine)
        captured["max_agent_input"] = s.token_budget.max_agent_input_tokens
        return {}

    monkeypatch.setattr(main_mod, "run_compass_pipeline", _pipeline)
    monkeypatch.setattr(main_mod, "reset_llm_client", lambda: None)
    monkeypatch.setattr(sys, "argv", ["main.py", str(SUBJ_001), *argv])
    reload_settings()
    with pytest.raises(SystemExit) as info:
        main_mod.main()
    assert info.value.code == 0, "main() reported an error"
    return captured


def test_command_line_with_a_decision_public_model(monkeypatch, no_terminal):
    captured = _run_main(
        monkeypatch,
        [
            "--public_model",
            JEV,
            "--orchestration",
            "never",
            "--orchestration_threshold",
            "9000",
            "--decision_provider",
            "typesafe",
            "--decision_choice_orders",
            "2",
            "--decision_score_levels",
            "9",
            "--no-decision_refine",
            "--quiet",
        ],
    )
    assert captured["models"]["predictor"] == JEV
    assert all(captured["models"][role] == DEFAULT_COMPANION_MODEL for role in ROLES)
    assert captured["public_model_name"] == DEFAULT_COMPANION_MODEL
    assert captured["orchestration"] == ("never", 9000)
    assert captured["decision"] == ("typesafe", 2, 9, False)
    # Token budgets follow the companion's context window, not the decision state limit.
    assert captured["max_agent_input"] > int(32_000 / 1.2)


def test_command_line_with_a_decision_predictor_and_companion(monkeypatch, no_terminal):
    captured = _run_main(
        monkeypatch,
        ["--public_model", LLM, "--predictor_model", JEV, "--companion_model", OTHER_LLM, "--decision_refine", "--quiet"],
    )
    assert captured["models"]["predictor"] == JEV
    assert all(captured["models"][role] == OTHER_LLM for role in ROLES)
    assert captured["decision"][3] is True
    assert captured["orchestration"][0] in ("auto", "always", "never")


def test_command_line_rejects_an_unknown_orchestration_mode(monkeypatch, no_terminal, capsys):
    monkeypatch.setattr(main_mod, "run_compass_pipeline", lambda **_kwargs: pytest.fail("must not run"))
    monkeypatch.setattr(sys, "argv", ["main.py", str(SUBJ_001), "--orchestration", "sometimes"])
    with pytest.raises(SystemExit) as info:
        main_mod.main()
    assert info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
