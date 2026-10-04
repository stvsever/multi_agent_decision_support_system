"""
Evidence routing: whether a participant record goes straight to the Predictor
(direct) or through the orchestration workflow first.

Covers the route decision, the direct executor output built from the pseudo
participants without any model call, the LLM Predictor's measurement of the
direct input (and that the direct prompt is never capped per section), and the
ROUTE event the dashboard store keeps. No socket may open.
"""

import contextlib
import io
import shutil
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.agents.executor import Executor
from src.full_stack.backend.agents.predictor import Predictor
from src.full_stack.backend.config.settings import ORCHESTRATION_MODES, reload_settings
from src.full_stack.backend.data.models.prediction_task import (
    PredictionMode,
    PredictionTaskNode,
    PredictionTaskSpec,
    build_binary_task_spec,
)
from src.full_stack.backend.decision.predictor import DecisionPredictor
from src.full_stack.backend.decision.questions import deterministic_book
from src.full_stack.backend.runtime import event_bus
from src.full_stack.backend.runtime.event_bus import EventStore, RunEventEmitter
from src.full_stack.backend.utils.core.data_loader import DataLoader
from src.full_stack.backend.utils.core.input_routing import (
    RouteDecision,
    build_direct_executor_output,
    decide_route,
    normalize_mode,
)
from src.full_stack.backend.utils.token_packer import count_tokens

REPO = Path(__file__).resolve().parents[2]
PSEUDO_INPUTS = REPO / "src/full_stack/backend/data/pseudo_data/inputs"
PARTICIPANTS = sorted(p.name for p in PSEUDO_INPUTS.iterdir() if p.is_dir())
MIDDLE_MARKER = "MIDDLE-MARKER-ALPHA"
TAIL_MARKER = "TAIL-MARKER-OMEGA"


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    def _blocked(*_args, **_kwargs):
        raise RuntimeError("network access is not allowed in these tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setenv("COMPASS_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def settings():
    s = reload_settings()
    yield s
    reload_settings()


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _load(participant_dir: Path):
    return _quiet(DataLoader().load, participant_dir)


def _direct(participant, spec, *, target="MDD", control="CONTROL"):
    return _quiet(
        build_direct_executor_output,
        participant,
        target_condition=target,
        control_condition=control,
        prediction_task_spec=spec,
    )


def _binary_spec():
    return build_binary_task_spec(target_label="MDD", control_label="CONTROL")


def _multiclass_spec():
    return PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="severity",
            display_name="Depression severity",
            mode=PredictionMode.MULTICLASS_CLASSIFICATION,
            class_labels=["none", "mild", "severe"],
        )
    )


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, "auto"),
        ("", "auto"),
        ("auto", "auto"),
        ("  AUTO ", "auto"),
        ("true", "auto"),
        ("on", "auto"),
        ("default", "auto"),
        ("always", "always"),
        ("Force", "always"),
        ("never", "never"),
        ("off", "never"),
        ("false", "never"),
        ("direct", "never"),
    ],
)
def test_normalize_mode_aliases(raw, expected):
    assert normalize_mode(raw) == expected
    assert expected in ORCHESTRATION_MODES


@pytest.mark.parametrize("raw", ["sometimes", "orchestrated", "yes please", "1"])
def test_normalize_mode_rejects_unknown_modes(raw):
    with pytest.raises(ValueError, match="orchestration mode must be one of auto, always, never"):
        normalize_mode(raw)


# ---------------------------------------------------------------------------
# Route decision
# ---------------------------------------------------------------------------
def _measure(tokens, budget=None):
    m = {"input_tokens": tokens}
    if budget is not None:
        m["budget_tokens"] = budget
    return m


def test_auto_goes_direct_when_the_record_fits():
    decision = decide_route(mode="auto", measurement=_measure(1000, 5000))
    assert isinstance(decision, RouteDecision)
    assert decision.route == "direct"
    assert decision.mode == "auto"
    assert (decision.input_tokens, decision.budget_tokens) == (1000, 5000)
    assert decision.threshold_source == "predictor_budget"
    assert decision.predictor_kind == "llm"
    assert decision.escalated is False
    assert "fits" in decision.reason and "orchestration skipped" in decision.reason
    # Exactly at the budget still fits.
    assert decide_route(mode="auto", measurement=_measure(5000, 5000)).route == "direct"


def test_auto_orchestrates_when_the_record_overflows():
    decision = decide_route(mode="auto", measurement=_measure(6000, 5000), predictor_kind="decision")
    assert decision.route == "orchestrated"
    assert decision.predictor_kind == "decision"
    assert "exceeds" in decision.reason


def test_always_orchestrates_even_when_the_record_fits():
    decision = decide_route(mode="always", measurement=_measure(10, 5000))
    assert decision.route == "orchestrated"
    assert decision.mode == "always"
    assert decide_route(mode="force", measurement=_measure(10, 5000)).mode == "always"


def test_never_goes_direct_even_when_the_record_overflows():
    over = decide_route(mode="never", measurement=_measure(6000, 5000))
    assert over.route == "direct"
    assert "packed to fit" in over.reason
    fits = decide_route(mode="never", measurement=_measure(100, 5000))
    assert fits.route == "direct"
    assert "fits the Predictor input" in fits.reason
    assert decide_route(mode="never", measurement=None).route == "direct"


def test_override_lowers_the_budget():
    lowered = decide_route(mode="auto", measurement=_measure(1000, 5000), threshold_override=500)
    assert lowered.budget_tokens == 500
    assert lowered.threshold_source == "override"
    assert lowered.route == "orchestrated"
    still_fits = decide_route(mode="auto", measurement=_measure(1000, 5000), threshold_override=2000)
    assert (still_fits.route, still_fits.budget_tokens) == ("direct", 2000)


def test_override_never_raises_the_budget_above_the_predictor_input():
    decision = decide_route(mode="auto", measurement=_measure(6000, 5000), threshold_override=10_000)
    assert decision.budget_tokens == 5000
    assert decision.threshold_source == "override"
    assert decision.route == "orchestrated"


def test_override_without_a_measured_budget_is_used_as_is():
    decision = decide_route(mode="auto", measurement=_measure(100), threshold_override=500)
    assert (decision.route, decision.budget_tokens, decision.threshold_source) == ("direct", 500, "override")


@pytest.mark.parametrize("override", [0, -100, None])
def test_non_positive_override_is_ignored(override):
    decision = decide_route(mode="auto", measurement=_measure(1000, 5000), threshold_override=override)
    assert decision.threshold_source == "predictor_budget"
    assert decision.budget_tokens == 5000


def test_missing_measurement_in_auto_orchestrates():
    decision = decide_route(mode="auto", measurement=None)
    assert decision.route == "orchestrated"
    assert (decision.input_tokens, decision.budget_tokens) == (0, 0)


def test_route_decision_to_dict():
    payload = decide_route(mode="auto", measurement=_measure(1, 2)).to_dict()
    assert set(payload) == {
        "route",
        "mode",
        "input_tokens",
        "budget_tokens",
        "threshold_source",
        "predictor_kind",
        "reason",
        "escalated",
    }


def test_invalid_mode_fails_the_route_decision():
    with pytest.raises(ValueError):
        decide_route(mode="bogus", measurement=_measure(1, 2))


# ---------------------------------------------------------------------------
# Direct executor output
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("participant_name", PARTICIPANTS)
def test_direct_executor_output_on_pseudo_participants(settings, monkeypatch, participant_name):
    def _no_constructor(*_args, **_kwargs):
        raise AssertionError("the direct route must not construct agents (they create model clients)")

    monkeypatch.setattr(Executor, "__init__", _no_constructor)
    participant = _load(PSEUDO_INPUTS / participant_name)
    spec = _binary_spec()
    out = _direct(participant, spec)

    assert out["route"] == "direct"
    assert out["chunking_skipped"] is True
    assert out["chunking_reason"] == "direct_route"
    assert out["chunk_evidence"] == []
    assert out["step_outputs"] == {}
    assert out["predictor_chunk_count"] == 0
    assert out["plan_id"] == "direct"
    assert out["execution_result"] is None
    assert out["total_tokens_used"] == 0
    assert out["participant_id"] == participant.participant_id
    assert (out["target_condition"], out["control_condition"]) == ("MDD", "CONTROL")
    # Same layout as the orchestrated Executor output: the dumped task spec.
    assert out["prediction_task_spec"] == spec.model_dump()

    assert out["domains_processed"]
    assert all(isinstance(d, str) and d for d in out["domains_processed"])
    assert set(out["domains_processed"]) <= set(out["multimodal_data"])

    ledger = out["coverage_ledger"]
    assert ledger["all_features"]
    assert ledger["missing_features"] == []
    assert ledger["summary"]["missing_count"] == 0
    assert sorted(ledger["covered_features"]) == sorted(ledger["all_features"])
    assert out["predictor_input"]["coverage_ledger"] is ledger

    predictor_input = out["predictor_input"]
    assert predictor_input["non_numerical_data_raw"] == out["non_numerical_data"]
    assert out["non_numerical_data"] == participant.non_numerical_data.raw_text
    assert predictor_input["multimodal_unprocessed_raw"]
    assert out["fusion_result"].skipped_fusion is True
    assert out["fusion_result"].context_fill_report == {"route": "direct"}
    assert isinstance(out["non_core_context_text"], str)


def test_every_feature_reaches_the_predictor_on_the_direct_route(settings):
    participant = _load(PSEUDO_INPUTS / PARTICIPANTS[0])
    out = _direct(participant, _binary_spec())
    predictor = Predictor(llm_client=SimpleNamespace())
    summary = predictor._validate_feature_representation(
        coverage_ledger=out["coverage_ledger"],
        predictor_input=out["predictor_input"],
        chunk_evidence=[],
    )
    assert summary["invariant_ok"] is True
    assert summary["missing_feature_count"] == 0
    assert summary["all_feature_count"] == len(out["coverage_ledger"]["all_features"])


def test_direct_executor_output_accepts_a_decision_model_token_hint(settings):
    # main.py passes the Predictor model as the token hint; tiktoken does not know Jev.
    participant = _load(PSEUDO_INPUTS / PARTICIPANTS[0])
    spec = _binary_spec()
    baseline = _direct(participant, spec)
    hinted = _quiet(
        build_direct_executor_output,
        participant,
        target_condition="MDD",
        control_condition="CONTROL",
        prediction_task_spec=spec,
        model_hint="typesafe/jev-1.13",
    )
    assert hinted["route"] == "direct"
    assert hinted["non_core_context_text"] == baseline["non_core_context_text"]
    assert hinted["coverage_ledger"]["all_features"] == baseline["coverage_ledger"]["all_features"]


def test_direct_executor_output_keeps_a_non_binary_task_spec(settings):
    participant = _load(PSEUDO_INPUTS / PARTICIPANTS[0])
    spec = _multiclass_spec()
    out = _direct(participant, spec, target="Depression severity", control="")
    assert PredictionTaskSpec(**out["prediction_task_spec"]) == spec
    assert out["route"] == "direct"


# ---------------------------------------------------------------------------
# LLM Predictor: measurement of the direct input
# ---------------------------------------------------------------------------
def test_llm_predictor_measures_the_direct_input(settings):
    participant = _load(PSEUDO_INPUTS / PARTICIPANTS[0])
    spec = _binary_spec()
    out = _direct(participant, spec)
    predictor = Predictor(llm_client=SimpleNamespace())
    measurement = predictor.measure_direct(dict(out), target_condition="MDD", control_condition="CONTROL", prediction_task_spec=spec)
    assert measurement["predictor_kind"] == "llm"
    assert measurement["model"] == settings.models.predictor_model
    assert measurement["input_tokens"] > 0
    assert measurement["budget_tokens"] > 0
    assert measurement["budget_tokens"] == predictor.input_budget_tokens()
    notes_tokens = count_tokens(out["non_numerical_data"], model_hint="cl100k_base")
    assert measurement["input_tokens"] > notes_tokens


def test_llm_predictor_input_budget_follows_the_settings(settings):
    predictor = Predictor(llm_client=SimpleNamespace())
    settings.token_budget.max_agent_input_tokens = 12_345
    assert predictor.input_budget_tokens() == 12_345
    settings.token_budget.max_agent_input_tokens = 0
    reserve = max(1024, min(8192, int(settings.models.predictor_max_tokens) + 1024))
    expected = settings.effective_context_window(settings.models.predictor_model) - reserve
    assert predictor.input_budget_tokens() == expected


def test_llm_measurement_drives_the_route(settings):
    participant = _load(PSEUDO_INPUTS / PARTICIPANTS[0])
    spec = _binary_spec()
    out = _direct(participant, spec)
    predictor = Predictor(llm_client=SimpleNamespace())

    settings.token_budget.max_agent_input_tokens = 0  # the full context window minus the output reserve
    roomy = predictor.measure_direct(dict(out), target_condition="MDD", control_condition="CONTROL", prediction_task_spec=spec)
    assert decide_route(mode="auto", measurement=roomy).route == "direct"
    assert decide_route(mode="auto", measurement=roomy, threshold_override=roomy["input_tokens"] - 1).route == "orchestrated"

    settings.token_budget.max_agent_input_tokens = 2048
    tight = predictor.measure_direct(dict(out), target_condition="MDD", control_condition="CONTROL", prediction_task_spec=spec)
    assert tight["input_tokens"] == roomy["input_tokens"]
    assert tight["budget_tokens"] == 2048
    assert decide_route(mode="auto", measurement=tight).route == "orchestrated"


def test_decision_predictor_measurement_drives_the_route(settings):
    settings.models.predictor_model = "typesafe/jev-1.13"
    participant = _load(PSEUDO_INPUTS / PARTICIPANTS[0])
    spec = _binary_spec()
    out = _direct(participant, spec)
    predictor = DecisionPredictor(llm_client=SimpleNamespace())
    predictor.question_book = lambda task_spec: deterministic_book(task_spec, predictor.task_context())
    measurement = predictor.measure_direct(dict(out), prediction_task_spec=spec)
    assert 0 < measurement["input_tokens"] < measurement["budget_tokens"] < int(32_000 / 1.2)
    decision = decide_route(mode="auto", measurement=measurement, predictor_kind="decision")
    assert (decision.route, decision.predictor_kind) == ("direct", "decision")
    assert decide_route(mode="auto", measurement=measurement, threshold_override=1000).route == "orchestrated"


# ---------------------------------------------------------------------------
# LLM Predictor: the direct prompt is never capped per section
# ---------------------------------------------------------------------------
def _long_notes() -> str:
    lines = ["## Clinical notes"]
    for i in range(320):
        if i == 160:
            lines.append(f"Session {i:03d}: {MIDDLE_MARKER} the participant described early morning waking and guilt.")
        else:
            lines.append(
                f"Session {i:03d}: the participant described low mood, poor concentration and reduced appetite "
                f"during week {i % 12}, with variable sleep."
            )
    lines.append(f"Final note: {TAIL_MARKER} follow-up planned in four weeks.")
    return "\n".join(lines)


@pytest.fixture
def long_record(tmp_path):
    folder = tmp_path / "SUBJ_LONG_NOTES"
    shutil.copytree(PSEUDO_INPUTS / PARTICIPANTS[0], folder)
    notes = _long_notes()
    (folder / "non_numerical_data.txt").write_text(notes)
    return folder, notes


def _prompt(predictor, out, spec):
    summary = predictor._validate_feature_representation(
        coverage_ledger=out["coverage_ledger"],
        predictor_input=out["predictor_input"],
        chunk_evidence=[],
    )
    prompt, context = predictor._compose_final_prompt(
        executor_output=out,
        predictor_input=out["predictor_input"],
        target_condition="MDD",
        control_condition="CONTROL",
        prediction_task_spec=spec,
        coverage_summary=summary,
        chunk_evidence=[],
        chunking_skipped=True,
    )
    return prompt, context


@pytest.mark.parametrize("spec_factory", [_binary_spec, _multiclass_spec])
def test_direct_route_prompt_keeps_notes_longer_than_the_section_cap(settings, long_record, spec_factory):
    folder, notes = long_record
    assert count_tokens(notes, model_hint="cl100k_base") > 2 * 2500
    spec = spec_factory()
    out = _direct(_load(folder), spec)
    predictor = Predictor(llm_client=SimpleNamespace())

    prompt, context = _prompt(predictor, out, spec)
    assert context["mode"] == "direct"
    assert notes.strip() in prompt
    assert MIDDLE_MARKER in prompt and TAIL_MARKER in prompt
    assert "middle context omitted" not in prompt

    # The same record on any other route is capped at 2,500 tokens per section.
    capped_out = dict(out)
    capped_out["route"] = "orchestrated"
    capped, _ = _prompt(predictor, capped_out, spec)
    assert MIDDLE_MARKER not in capped
    assert "middle context omitted" in capped
    assert len(capped) < len(prompt)

    measurement = predictor.measure_direct(dict(out), target_condition="MDD", control_condition="CONTROL", prediction_task_spec=spec)
    assert measurement["input_tokens"] > count_tokens(notes, model_hint="cl100k_base")


class _Stop(Exception):
    pass


def test_execute_sends_the_uncapped_prompt_and_appends_revision_feedback(settings, long_record):
    folder, _notes = long_record
    spec = _binary_spec()
    out = _direct(_load(folder), spec)
    predictor = Predictor(llm_client=SimpleNamespace())
    sent = []

    def _capture(**kwargs):
        sent.append(kwargs)
        raise _Stop()

    predictor._call_predictor_json = _capture
    with pytest.raises(_Stop):
        _quiet(predictor.execute, dict(out), "MDD", "CONTROL", prediction_task_spec=spec)
    first = sent[0]["user_prompt"]
    assert MIDDLE_MARKER in first and TAIL_MARKER in first
    assert "## Critic feedback on the previous attempt" not in first

    predictor.set_revision_feedback("  The probability ignores the sleep findings.  ")
    assert predictor.revision_feedback == "The probability ignores the sleep findings."
    executor_output = dict(out)
    with pytest.raises(_Stop):
        _quiet(predictor.execute, executor_output, "MDD", "CONTROL", prediction_task_spec=spec)
    revised = sent[1]["user_prompt"]
    assert revised.startswith(first)
    assert "## Critic feedback on the previous attempt\nThe probability ignores the sleep findings." in revised
    assert revised.rstrip().endswith("keeping the same JSON contract.")
    call_context = executor_output["predictor_call_context"]
    assert call_context["mode"] == "direct"
    assert call_context["final_prompt_token_estimate"] == predictor._token_count(revised)

    predictor.set_revision_feedback(None)
    with pytest.raises(_Stop):
        _quiet(predictor.execute, dict(out), "MDD", "CONTROL", prediction_task_spec=spec)
    assert sent[2]["user_prompt"] == first


# ---------------------------------------------------------------------------
# ROUTE event
# ---------------------------------------------------------------------------
def test_route_event_resets_attempt_state_for_a_direct_attempt():
    store = EventStore()
    store.add_event("STEP_START", {"id": 1, "tool": "PhenotypeRepresentation", "desc": "Represent"})
    assert store.state["steps"]
    decision = decide_route(mode="auto", measurement=_measure(100, 1000)).to_dict()
    store.add_event("ROUTE", {**decision, "iteration": 2})

    assert store.state["route"]["route"] == "direct"
    assert store.state["routes"] == [store.state["route"]]
    assert store.state["iteration"] == 2
    assert store.state["steps"] == []
    assert [s["id"] for s in store.state["history"]] == [1]
    assert store.state["current_stage"] == 4
    assert store.state["max_steps"] == 0
    assert store.state["prediction"] is None and store.state["critic"] is None
    assert store.state["status"] == "Direct prediction: orchestration skipped"


def test_route_event_for_an_orchestrated_attempt_keeps_the_plan_state():
    store = EventStore()
    store.add_event("STEP_START", {"id": 7, "tool": "FeatureSynthesizer", "desc": "Synthesize"})
    stage_before = store.state["current_stage"]
    escalated = {**decide_route(mode="auto", measurement=_measure(9000, 1000)).to_dict(), "escalated": True}
    store.add_event("ROUTE", escalated)
    assert store.state["route"]["route"] == "orchestrated"
    assert store.state["route"]["escalated"] is True
    assert [s["id"] for s in store.state["steps"]] == [7]
    assert store.state["current_stage"] == stage_before
    assert len(store.state["routes"]) == 1


def test_emitter_sends_the_route_decision_with_its_iteration(monkeypatch):
    store = EventStore()
    monkeypatch.setattr(event_bus, "_event_store", store)
    decision = decide_route(mode="always", measurement=_measure(10, 1000)).to_dict()
    RunEventEmitter().on_route_decision(decision, iteration=3)
    RunEventEmitter().on_route_decision(None)
    events = [e for e in store.events if e["type"] == "ROUTE"]
    assert len(events) == 2
    assert events[0]["data"]["iteration"] == 3
    assert events[0]["data"]["route"] == "orchestrated"
    assert "iteration" not in decision  # the caller's dict is not mutated
    assert events[1]["data"] == {}
    assert len(store.state["routes"]) == 2
