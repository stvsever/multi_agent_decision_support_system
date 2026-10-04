"""
Evidence routing: whether a participant record goes straight to the Predictor
(direct) or through the orchestration workflow first.

Covers the route decision, the direct executor output built from the pseudo
participants without any model call, the LLM Predictor's measurement of the
direct input (and that the direct prompt is never capped per section), and the
ROUTE event the dashboard store keeps. No socket may open.
"""

import contextlib
import copy
import io
import json
import re
import shutil
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import main as main_mod
from src.full_stack.backend.decision.client import DecisionRequestError
from src.full_stack.backend.utils.core.predictor_input_assembler import PredictorInputAssembler
from src.full_stack.backend.utils.llm_client import is_context_length_error
from src.full_stack.backend.agents.executor import Executor
from src.full_stack.backend.agents.predictor import Predictor
from src.full_stack.backend.config.settings import ORCHESTRATION_MODES, LLMBackend, reload_settings
from src.full_stack.backend.data.models.prediction_result import (
    BinaryClassification,
    ClassificationPrediction,
    NodePrediction,
    PredictionResult,
    Verdict,
)
from src.full_stack.backend.data.models.prediction_task import (
    PredictionMode,
    PredictionTaskNode,
    PredictionTaskSpec,
    build_binary_task_spec,
)
from src.full_stack.backend.decision.predictor import DecisionPredictor
from src.full_stack.backend.decision.quality import evaluate_decision_prediction
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
from src.full_stack.backend.utils.core.record_rendering import DIRECT_RECORD_HEADER, measurement_groups, render_deviation_profile
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
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path / "compass_home"))


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
    # The payload estimate goes to the Predictor input only, not into the fusion result.
    assert out["fusion_result"].context_fill_report == {"route": "direct"}
    assert predictor_input["context_fill_report"] is not out["fusion_result"].context_fill_report

    # The compact, lossless rendering the Predictor reads on the direct route.
    record = out["direct_record"]
    assert set(record) == {"clinical_record", "deviation_profile", "data_overview", "measurements"}
    assert record["clinical_record"] == participant.non_numerical_data.raw_text.strip()
    leaf_lines = [line for line in record["measurements"].splitlines() if line.startswith("- ")]
    # Every feature leaf once, nothing summarised or left out.
    assert len(leaf_lines) == len(ledger["all_features"])
    assert out["direct_fits"] is True

    non_core = out["non_core_context_text"]
    assert non_core == f"## measurements\n{DIRECT_RECORD_HEADER}\n\n{record['measurements']}"
    # build_direct_executor_output counts with the default model hint, gpt-5.
    assert out["non_core_context_tokens"] == count_tokens(non_core, model_hint="gpt-5") > 0
    estimate = predictor_input["context_fill_report"]["predictor_payload_estimate"]
    assert estimate == {
        "final_tokens": out["non_core_context_tokens"]
        + sum(count_tokens(record[key], model_hint="gpt-5") for key in ("clinical_record", "deviation_profile", "data_overview")),
        "strategy": "direct_route_compact_record",
        "chunked_two_pass_required": False,
    }
    assert predictor_input["context_fill_report"]["route"] == "direct"


def test_direct_record_renders_leaves_groups_and_coverage(settings):
    participant = _load(PSEUDO_INPUTS / PARTICIPANTS[0])
    out = _direct(participant, _binary_spec())
    record = out["direct_record"]
    # Leaves: "label = value (z +1.23)" or "label (z +1.23)", grouped under "[path]".
    leaf = re.compile(r"^- .+ \(z [+-]\d+\.\d{2}\)$")
    lines = record["measurements"].splitlines()
    assert lines[0].startswith("[") and lines[0].endswith("]")
    assert any(leaf.match(line) for line in lines)
    assert all(line.startswith("[") or line.startswith("- ") for line in lines)
    # Group scores of the parsed deviation tree (leaf scores live in the measurements).
    assert out["deviation_tree"] is participant.hierarchical_deviation
    assert record["deviation_profile"] == render_deviation_profile(
        participant.hierarchical_deviation, leaf_groups=measurement_groups(out["predictor_input"]["multimodal_unprocessed_raw"])
    )
    # Coverage: "<domain>: <present>/<total> leaves present (<pct>%)".
    overview = out["data_overview"]["domain_coverage"]
    for line in record["data_overview"].splitlines():
        name = line.split(":", 1)[0]
        cov = overview[name]
        assert line.startswith(f"{name}: {int(cov['present_leaves'])}/{int(cov['total_leaves'])} leaves present")
    # Far fewer tokens than the raw JSON renderings it replaces.
    raw_json = json.dumps(out["predictor_input"]["multimodal_unprocessed_raw"])
    assert count_tokens(record["measurements"], model_hint="gpt-5") < count_tokens(raw_json, model_hint="gpt-5")


PROFILE_LINE = re.compile(r"^[^:]+(?: > [^:]+)*: (?:mean \|z\| \d+\.\d{2}|[+-]\d+\.\d{2})$")


@pytest.mark.parametrize("participant_name", PARTICIPANTS)
def test_deviation_profile_keeps_the_group_scores_of_every_pseudo_participant(settings, participant_name):
    participant = _load(PSEUDO_INPUTS / participant_name)
    out = _direct(participant, _binary_spec())
    lines = out["direct_record"]["deviation_profile"].splitlines()
    assert lines, participant_name
    assert all(PROFILE_LINE.match(line) for line in lines), lines[:5]
    # No profile line repeats a measurement leaf: every path is a group, not a leaf.
    leaf_names = {
        line[2:].split(" = ")[0].split(" (z ")[0]
        for line in out["direct_record"]["measurements"].splitlines()
        if line.startswith("- ")
    }
    assert not any(line.split(": ")[0].split(" > ")[-1] in leaf_names for line in lines)


def test_subj_001_deviation_profile_lists_signed_group_scores(settings):
    out = _direct(_load(PSEUDO_INPUTS / "SUBJ_001_PSEUDO"), _binary_spec())
    lines = out["direct_record"]["deviation_profile"].splitlines()
    assert lines[:3] == [
        "BIOLOGICAL_ASSAY > lipidomics > sphingolipids: +1.70",
        "BIOLOGICAL_ASSAY > lipidomics > phospholipids: -0.70",
        "BIOLOGICAL_ASSAY > lipidomics > fatty_acids: +1.00",
    ]
    assert "BRAIN_MRI > structural > subcortical_volumes: -1.40" in lines
    # The same tree reaches the Predictor prompt on the direct route.
    predictor = Predictor(llm_client=SimpleNamespace())
    prompt, _ = _prompt(predictor, out, _binary_spec())
    assert "BIOLOGICAL_ASSAY > lipidomics > sphingolipids: +1.70" in prompt


def test_leaf_qualifiers_are_rendered_in_the_direct_record(settings):
    out = _direct(_load(PSEUDO_INPUTS / "SUBJ_002_PSEUDO"), _binary_spec())
    measurements = out["direct_record"]["measurements"].splitlines()
    assert "- fasting_glucose = 92 mg/dL (z -0.20) [ref range 70-99]" in measurements
    assert "- HbA1c = 5.2% (z -0.10) [ref range <5.7]" in measurements
    assert "[qualifiers]" in DIRECT_RECORD_HEADER


def _context_rows(participant):
    executor = Executor.__new__(Executor)
    context = executor._build_context(participant, "MDD", "CONTROL", prediction_task_spec=_binary_spec(), agent_instructions=None)
    return [row for rows in context["multimodal_data"].values() for row in rows]


BASE_LEAF_KEYS = {"domain", "feature_id", "field_name", "path_in_hierarchy", "unit", "value", "z_score"}


@pytest.mark.parametrize("participant_name", PARTICIPANTS)
def test_executor_context_has_a_qualifiers_key_only_when_a_leaf_has_qualifiers(settings, participant_name):
    rows = _context_rows(_load(PSEUDO_INPUTS / participant_name))
    for row in rows:
        if "qualifiers" in row:
            assert row["qualifiers"] and isinstance(row["qualifiers"], dict)
            assert set(row) == BASE_LEAF_KEYS | {"qualifiers"}
        else:
            assert set(row) == BASE_LEAF_KEYS


def _strip_leaf_extras(node):
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key == "_leaves" and isinstance(value, list):
                node[key] = [{k: v for k, v in leaf.items() if k in ("feature", "value", "z_score", "unit")} for leaf in value]
            else:
                _strip_leaf_extras(value)
    return node


_CORE_SECTIONS = {
    "non_numerical_data_raw",
    "hierarchical_deviation_raw",
    "data_overview",
    "phenotype_representation",
    "feature_synthesizer",
    "differential_diagnosis",
}


def _orchestrated_prompt(out, spec):
    """The Predictor prompt for this record on the orchestrated route: the generic
    JSON renderings and the Integrator's pass-through non-core sections."""
    assembler = PredictorInputAssembler(max_chunk_tokens=10**9, model_hint="gpt-5")
    sections = assembler.build_sections(
        executor_output={
            "step_outputs": {},
            "data_overview": out["data_overview"],
            "hierarchical_deviation": out["hierarchical_deviation"],
            "non_numerical_data": out["non_numerical_data"],
        },
        predictor_input=out["predictor_input"],
        coverage_ledger=out["coverage_ledger"],
    )
    non_core = [section for section in sections if section.name.split("#", 1)[0] not in _CORE_SECTIONS]
    orchestrated = {k: v for k, v in out.items() if k not in ("direct_record", "direct_fits")}
    orchestrated["route"] = "orchestrated"
    orchestrated["non_core_context_text"] = assembler.chunk_to_text(non_core, 1, 1)
    prompt, _ = _prompt(Predictor(llm_client=SimpleNamespace()), orchestrated, spec)
    return prompt


def test_orchestrated_prompt_is_unchanged_for_leaves_without_qualifiers(settings, tmp_path):
    folder = tmp_path / "SUBJ_002_NO_QUALIFIERS"
    shutil.copytree(PSEUDO_INPUTS / "SUBJ_002_PSEUDO", folder)
    data_path = folder / "multimodal_data.json"
    data_path.write_text(json.dumps(_strip_leaf_extras(json.loads(data_path.read_text()))))
    participant = _load(folder)
    rows = _context_rows(participant)
    assert rows and all(set(row) == BASE_LEAF_KEYS for row in rows)
    out = _direct(participant, _binary_spec())
    assert "qualifiers" not in json.dumps(out["predictor_input"]["multimodal_unprocessed_raw"], default=str)
    prompt = _orchestrated_prompt(out, _binary_spec())
    assert "fasting_glucose" in prompt
    # No trace of the new field when no leaf carries a qualifier.
    assert "qualifiers" not in prompt and "ref_range" not in prompt
    # The original participant's leaves do carry them, and only then do they appear.
    original = _direct(_load(PSEUDO_INPUTS / "SUBJ_002_PSEUDO"), _binary_spec())
    assert "ref_range" in _orchestrated_prompt(original, _binary_spec())


def test_an_anchored_label_keeps_a_value_it_does_not_state(settings, tmp_path):
    folder = tmp_path / "SUBJ_PHQ"
    shutil.copytree(PSEUDO_INPUTS / PARTICIPANTS[0], folder)
    data_path = folder / "multimodal_data.json"
    data = json.loads(data_path.read_text())
    data["MOOD"] = {"_leaves": [{"feature": "PHQ-9 total score", "value": 9, "z_score": 1.1}, {"feature": "Sex: Female", "value": "Female"}]}
    data_path.write_text(json.dumps(data))
    out = _direct(_load(folder), _binary_spec())
    lines = out["direct_record"]["measurements"].splitlines()
    assert "- PHQ-9 total score = 9 (z +1.10)" in lines
    assert "- Sex: Female" in lines


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


LEAF_LINE = re.compile(r"^- .+ \(z [+-]\d+\.\d{2}\)$", re.MULTILINE)
RAW_HEADINGS = (
    "## non_numerical_data_raw",
    "## hierarchical_deviation_raw",
    "## phenotype_representation",
    "## feature_synthesizer",
    "## differential_diagnosis",
)


@pytest.mark.parametrize("spec_factory", [_binary_spec, _multiclass_spec])
def test_direct_route_prompt_uses_the_compact_record(settings, spec_factory):
    spec = spec_factory()
    out = _direct(_load(PSEUDO_INPUTS / PARTICIPANTS[0]), spec)
    predictor = Predictor(llm_client=SimpleNamespace())
    prompt, context = _prompt(predictor, out, spec)
    assert context["mode"] == "direct"
    record = out["direct_record"]
    assert "## clinical_record (free text)\n" + record["clinical_record"] in prompt
    assert "## deviation_profile (aggregate z of each ontology group" in prompt
    assert record["deviation_profile"] in prompt
    assert "## data_overview (leaves present per domain)\n" + record["data_overview"] in prompt
    # The measurements follow as the non-core context, every leaf with its z.
    assert "## measurements\n" + DIRECT_RECORD_HEADER in prompt
    assert record["measurements"] in prompt
    assert LEAF_LINE.search(prompt)
    # No raw JSON renderings and no tool sections on the direct route.
    for heading in RAW_HEADINGS:
        assert heading not in prompt
    assert context["high_priority_context"].startswith("## clinical_record (free text)")

    # The same output on the orchestrated route reads the raw renderings and tool sections.
    orchestrated = dict(out)
    orchestrated["route"] = "orchestrated"
    raw_prompt, _ = _prompt(predictor, orchestrated, spec)
    for heading in RAW_HEADINGS:
        assert heading in raw_prompt
    assert "## clinical_record (free text)" not in raw_prompt


def test_direct_record_that_was_not_measured_to_fit_is_capped(settings, long_record):
    folder, _notes = long_record
    spec = _binary_spec()
    out = _direct(_load(folder), spec)
    predictor = Predictor(llm_client=SimpleNamespace())
    fits, _ = _prompt(predictor, out, spec)
    assert MIDDLE_MARKER in fits and TAIL_MARKER in fits
    # "never" mode sends a record that does not fit: it is capped per section.
    out["direct_fits"] = False
    capped, _ = _prompt(predictor, out, spec)
    assert MIDDLE_MARKER not in capped
    assert "middle context omitted" in capped
    assert "## clinical_record (free text)" in capped


def test_local_backend_caps_are_skipped_when_the_direct_record_fits(settings, long_record):
    settings.models.backend = LLMBackend.LOCAL
    folder, _notes = long_record
    spec = _binary_spec()
    out = _direct(_load(folder), spec)
    predictor = Predictor(llm_client=SimpleNamespace())
    assert predictor._is_local_backend()
    fits, _ = _prompt(predictor, out, spec)
    assert MIDDLE_MARKER in fits and TAIL_MARKER in fits
    assert out["direct_record"]["measurements"] in fits
    out["direct_fits"] = False
    capped, _ = _prompt(predictor, out, spec)
    # The local cap is 900 tokens per section.
    assert MIDDLE_MARKER not in capped
    assert "middle context omitted" in capped


def _valid_binary_json() -> str:
    return json.dumps(
        {
            "prediction_id": "p1",
            "binary_classification": "CASE",
            "probability_score": 0.7,
            "confidence_level": "MEDIUM",
            "key_findings": [],
            "reasoning_chain": ["step"],
            "clinical_summary": "summary",
            "supporting_evidence": {"for_case": [], "for_control": []},
            "uncertainty_factors": [],
        }
    )


class _ScriptedLLM:
    """Returns (or raises) the scripted items in order and records every prompt."""

    def __init__(self, items):
        self.items = list(items)
        self.prompts = []

    def call(self, *, messages, **_kwargs):
        self.prompts.append(messages[1]["content"])
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(content=item, prompt_tokens=10, completion_tokens=10)


def _long_prompt(predictor, folder):
    spec = _binary_spec()
    out = _direct(_load(folder), spec)
    prompt, _ = _prompt(predictor, out, spec)
    predictor._active_prediction_task_spec = spec
    return prompt


@pytest.mark.parametrize("keep_full", [True, False])
def test_direct_route_retry_after_a_parse_error_keeps_the_full_prompt(settings, long_record, keep_full):
    folder, _notes = long_record
    llm = _ScriptedLLM(["this is not JSON", _valid_binary_json()])
    predictor = Predictor(llm_client=llm)
    prompt = _long_prompt(predictor, folder)
    predictor._keep_full_prompt = keep_full
    with contextlib.redirect_stdout(io.StringIO()):
        parsed = predictor._call_predictor_json(system_prompt="sys", user_prompt=prompt, max_retries=2)
    assert parsed["probability_score"] == 0.7
    first, retry = llm.prompts
    assert first == prompt
    assert "PREVIOUS_ERROR:" in retry
    if keep_full:
        # Direct route: a parse error is no reason to drop part of the record.
        assert retry.startswith(prompt)
        assert MIDDLE_MARKER in retry and TAIL_MARKER in retry
    else:
        # Any other route shrinks the prompt on a retry (keep ratio 0.92).
        assert "[TRUNCATED FOR RETRY: middle context omitted" in retry
        assert len(retry) < len(prompt)


def test_direct_route_retry_after_a_length_error_shrinks_the_prompt(settings, long_record):
    folder, _notes = long_record
    llm = _ScriptedLLM([RuntimeError("Empty response (finish_reason=length)"), _valid_binary_json()])
    predictor = Predictor(llm_client=llm)
    prompt = _long_prompt(predictor, folder)
    predictor._keep_full_prompt = True
    with contextlib.redirect_stdout(io.StringIO()):
        predictor._call_predictor_json(system_prompt="sys", user_prompt=prompt, max_retries=2)
    retry = llm.prompts[1]
    assert "[TRUNCATED FOR RETRY: middle context omitted" in retry
    assert "PREVIOUS_ERROR:\nEmpty response (finish_reason=length)" in retry


@pytest.mark.parametrize("keep_full, calls", [(True, 1), (False, 3)])
def test_a_context_length_error_on_the_direct_route_is_raised_at_once(settings, long_record, keep_full, calls):
    folder, _notes = long_record
    error = RuntimeError("Error code: 400 - This model's maximum context length is 65536 tokens")
    llm = _ScriptedLLM([error, error, error])
    predictor = Predictor(llm_client=llm)
    prompt = _long_prompt(predictor, folder)
    predictor._keep_full_prompt = keep_full
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(RuntimeError, match="maximum context length"):
        predictor._call_predictor_json(system_prompt="sys", user_prompt=prompt, max_retries=2)
    # Direct route: retrying the same complete record cannot help, so the
    # pipeline gets the error at once and orchestrates. Elsewhere the prompt is
    # shrunk and retried as before.
    assert len(llm.prompts) == calls


def test_only_a_direct_record_measured_to_fit_is_uncapped(settings):
    predictor = Predictor(llm_client=SimpleNamespace())
    assert predictor._direct_uncapped({"route": "direct", "direct_fits": True}) is True
    # Older direct outputs without the flag were measured to fit.
    assert predictor._direct_uncapped({"route": "direct"}) is True
    assert predictor._direct_uncapped({"route": "direct", "direct_fits": False}) is False
    assert predictor._direct_uncapped({"route": "orchestrated", "direct_fits": True}) is False


def test_prompt_over_the_input_budget_can_be_cut_below_a_quarter(settings):
    predictor = Predictor(llm_client=SimpleNamespace())
    settings.token_budget.max_agent_input_tokens = 2048
    prompt = " ".join(f"word{i}" for i in range(20_000))
    total = predictor._token_count(prompt)
    assert total > 8 * 2048
    fitted = predictor._fit_prompt_to_input_budget(prompt, max_completion_tokens=1024)
    kept = predictor._token_count(fitted)
    # Kept within the budget, marker included (32 tokens are reserved for it),
    # far below the old 25 percent floor.
    assert kept <= 2048
    assert kept < 0.25 * total
    assert "middle context omitted" in fitted
    assert fitted.startswith("word0 word1")
    assert fitted.rstrip().endswith("word19999")
    # A prompt within the budget is untouched.
    assert predictor._fit_prompt_to_input_budget("short prompt", max_completion_tokens=1024) == "short prompt"


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


# ---------------------------------------------------------------------------
# Pipeline fallbacks (main.run_compass_pipeline with stub agents)
# ---------------------------------------------------------------------------
class _StubAgent:
    def __init__(self, *_args, **_kwargs):
        self.instruction = ""
        self.integrator = SimpleNamespace(set_runtime_instruction=lambda _text: None)

    def set_runtime_instruction(self, text):
        self.instruction = text


class _StubCritic(_StubAgent):
    def execute(self, *, prediction, prediction_task_spec, **_kwargs):
        # The structural decision critic accepts any complete, valid prediction.
        return evaluate_decision_prediction(prediction, prediction_task_spec)


class _StubReports:
    def generate(self, **_kwargs):
        return {}

    def save(self, *_args, **_kwargs):
        return None

    def save_markdown(self, *_args, **_kwargs):
        return None


def _stub_prediction(spec, route, quality=None):
    root = NodePrediction(
        node_id=spec.root.node_id,
        path=spec.root.node_id,
        mode=spec.root.mode,
        classification=ClassificationPrediction(predicted_label="MDD", probabilities={"MDD": 0.7, "CONTROL": 0.3}),
    )
    return PredictionResult(
        prediction_id=f"stub_{route}",
        participant_id="stub",
        target_condition="MDD",
        control_condition="CONTROL",
        prediction_task_spec=spec,
        root_prediction=root,
        binary_classification=BinaryClassification.CASE,
        probability_score=0.7,
        predictor_kind="llm",
        input_route=route,
        decision_report={"route": route, "quality": dict(quality)} if quality else {},
    )


class _RecordingUI:
    """Stands in for the dashboard emitter and records every call."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            # A copy, as the real emitter serializes the event when it is sent.
            self.calls.append((name, copy.deepcopy(args), copy.deepcopy(kwargs)))

        return _record


@pytest.fixture
def pipeline(settings, monkeypatch, tmp_path):
    """run_compass_pipeline with every model-backed part replaced by a stub."""
    state = SimpleNamespace(
        measure=None,
        measure_error=None,
        direct_error=None,
        direct_error_from=1,  # first iteration whose direct call raises direct_error
        quality={},  # iteration -> decision quality block of that attempt's prediction
        predictor_calls=[],
        call_details=[],
        orchestrated=[],
        orchestrated_feedback=[],
        ui=None,
    )

    class _StubPredictor(_StubAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.revision_feedback = ""

        def measure_direct(self, executor_output, **_kwargs):
            if state.measure_error is not None:
                raise state.measure_error
            return dict(state.measure or {"input_tokens": 100, "budget_tokens": 1000, "predictor_kind": "llm"})

        def execute(self, executor_output, target_condition, control_condition, prediction_task_spec=None, iteration=1):
            route = executor_output.get("route")
            state.predictor_calls.append({"route": route, "direct_fits": executor_output.get("direct_fits"), "iteration": iteration})
            state.call_details.append(
                {"has_deviation_tree": executor_output.get("deviation_tree") is not None, "revision_feedback": self.revision_feedback}
            )
            if route == "direct" and state.direct_error is not None and iteration >= state.direct_error_from:
                raise state.direct_error
            return _stub_prediction(prediction_task_spec, route, state.quality.get(iteration))

        def set_revision_feedback(self, text):
            self.revision_feedback = str(text or "")

    def _orchestrated_attempt(
        *, participant_data, target_condition, control_condition, prediction_task_spec, iteration, previous_feedback, **_kwargs
    ):
        state.orchestrated.append(iteration)
        state.orchestrated_feedback.append(previous_feedback)
        output = build_direct_executor_output(
            participant_data,
            target_condition=target_condition,
            control_condition=control_condition,
            prediction_task_spec=prediction_task_spec,
        )
        # A real orchestrated output has neither the direct record nor the parsed tree.
        for key in ("direct_record", "direct_fits", "deviation_tree"):
            output.pop(key, None)
        output["route"] = "orchestrated"
        plan = main_mod._direct_plan(
            participant_id=output["participant_id"],
            target_condition=target_condition,
            control_condition=control_condition,
            prediction_task_spec=prediction_task_spec,
            domains=output["domains_processed"],
            reason="stub orchestration",
            iteration=iteration,
        )
        return plan, output

    settings.paths.logs_dir = tmp_path / "logs"
    settings.paths.logs_dir.mkdir()
    monkeypatch.setattr(main_mod, "get_llm_client", lambda: SimpleNamespace(ping=lambda: None))
    for name in ("Orchestrator", "Executor", "Communicator"):
        monkeypatch.setattr(main_mod, name, _StubAgent)
    monkeypatch.setattr(main_mod, "Critic", _StubCritic)
    monkeypatch.setattr(main_mod, "Predictor", _StubPredictor)
    monkeypatch.setattr(main_mod, "DecisionPredictor", _StubPredictor)
    monkeypatch.setattr(main_mod, "_run_orchestrated_attempt", _orchestrated_attempt)
    monkeypatch.setattr(
        main_mod,
        "_run_explainability_for_selected_attempt",
        lambda **_kwargs: {"enabled": False, "status": "disabled", "methods_requested": []},
    )
    monkeypatch.setattr(main_mod, "PatientReportGenerator", _StubReports)
    monkeypatch.setattr(main_mod, "_resolve_output_dir", lambda *_args: tmp_path / "out")

    def _run(mode="auto", *, threshold=None, interactive=False, predictor_model=None):
        if predictor_model:
            settings.models.predictor_model = predictor_model
        if interactive:
            state.ui = _RecordingUI()
            monkeypatch.setattr(main_mod, "get_ui", lambda enabled=False: state.ui)
        with contextlib.redirect_stdout(io.StringIO()):
            return main_mod.run_compass_pipeline(
                participant_dir=PSEUDO_INPUTS / PARTICIPANTS[0],
                target_condition="MDD",
                control_condition="CONTROL",
                prediction_task_spec=_binary_spec(),
                max_iterations=2,
                verbose=False,
                interactive_ui=interactive,
                orchestration_mode=mode,
                orchestration_threshold_tokens=threshold,
            )

    state.run = _run
    return state


def test_pipeline_goes_direct_when_the_record_fits(pipeline):
    result = pipeline.run("auto")
    assert pipeline.predictor_calls == [{"route": "direct", "direct_fits": True, "iteration": 1}]
    assert pipeline.orchestrated == []
    assert result["input_route"] == "direct"
    assert result["verdict"] == Verdict.SATISFACTORY.value
    attempt = result["routing"]["attempts"][0]
    assert (attempt["route"], attempt["cause"], attempt["escalated"]) == ("direct", "measurement", False)
    assert attempt["reason"] == result["routing"]["initial"]["reason"]


def test_measurement_failure_falls_back_to_orchestration(pipeline):
    pipeline.measure_error = RuntimeError("tokenizer exploded")
    result = pipeline.run("auto")
    # Without a measurement auto mode cannot show that the record fits: orchestrate.
    assert pipeline.orchestrated == [1]
    assert pipeline.predictor_calls == [{"route": "orchestrated", "direct_fits": None, "iteration": 1}]
    assert result["input_route"] == "orchestrated"
    initial = result["routing"]["initial"]
    assert initial["route"] == "orchestrated"
    assert (initial["input_tokens"], initial["budget_tokens"]) == (0, 0)
    assert result["iterations"] == 1


def test_measurement_failure_in_never_mode_goes_direct_without_the_fit_flag(pipeline):
    pipeline.measure_error = RuntimeError("tokenizer exploded")
    result = pipeline.run("never")
    assert pipeline.orchestrated == []
    # Direct, but not measured to fit: the Predictor may cut the record.
    assert pipeline.predictor_calls == [{"route": "direct", "direct_fits": False, "iteration": 1}]
    assert result["input_route"] == "direct"


def test_never_mode_with_an_oversized_record_is_not_marked_as_fitting(pipeline):
    pipeline.measure = {"input_tokens": 5000, "budget_tokens": 1000, "predictor_kind": "llm"}
    pipeline.run("never")
    assert pipeline.predictor_calls == [{"route": "direct", "direct_fits": False, "iteration": 1}]


def test_context_length_error_on_the_direct_route_orchestrates_the_attempt(pipeline):
    pipeline.direct_error = RuntimeError("Error code: 400 - This model's maximum context length is 65536 tokens")
    result = pipeline.run("auto")
    # The same iteration is rerun on the orchestrated route.
    assert pipeline.predictor_calls == [
        {"route": "direct", "direct_fits": True, "iteration": 1},
        {"route": "orchestrated", "direct_fits": None, "iteration": 1},
    ]
    assert pipeline.orchestrated == [1]
    assert result["input_route"] == "orchestrated"
    assert result["iterations"] == 1
    routing = result["routing"]
    assert routing["initial"]["route"] == "direct"
    assert routing["selected_route"] == "orchestrated"
    assert len(routing["attempts"]) == 1
    attempt = routing["attempts"][0]
    assert attempt["route"] == "orchestrated"
    assert attempt["escalated"] is True
    assert attempt["iteration"] == 1
    assert attempt["cause"] == "context_length"
    assert attempt["reason"] == "The provider rejected the direct prompt as too long; orchestrating."


def test_a_decision_length_error_on_the_direct_route_also_orchestrates(pipeline):
    pipeline.direct_error = DecisionRequestError("Decision request returned HTTP 413", status=413, body="request too long")
    result = pipeline.run("auto", predictor_model="typesafe/jev-1.13")
    assert pipeline.orchestrated == [1]
    assert result["predictor_kind"] == "decision"
    assert result["routing"]["attempts"][0]["cause"] == "context_length"


def test_the_fallback_emits_a_failed_predictor_step(pipeline):
    pipeline.direct_error = RuntimeError("maximum context length exceeded")
    pipeline.run("auto", interactive=True)
    names = [name for name, _args, _kwargs in pipeline.ui.calls]
    failed = [kwargs for name, _args, kwargs in pipeline.ui.calls if name == "on_step_failed"]
    assert failed == [{"step_id": 911, "error": "Direct prompt rejected as too long: maximum context length exceeded"}]
    # The failure is shown before the route changes to orchestrated.
    routes = [(args, kwargs) for name, args, kwargs in pipeline.ui.calls if name == "on_route_decision"]
    assert [r[0][0]["route"] for r in routes] == ["direct", "orchestrated"]
    assert routes[-1][0][0]["cause"] == "context_length"
    assert names.index("on_step_failed") < max(i for i, n in enumerate(names) if n == "on_route_decision")


def test_revision_feedback_moves_to_the_orchestrator_on_a_length_fallback(pipeline):
    # Iteration 1 (direct) is rejected by the critic; the LLM Predictor gets the
    # feedback for a direct revision. Iteration 2's direct prompt is too long.
    pipeline.quality = {1: {"max_instability": 2.0, "mean_instability": 2.0}}
    pipeline.direct_error = RuntimeError("context_length_exceeded")
    pipeline.direct_error_from = 2
    result = pipeline.run("auto")
    assert pipeline.predictor_calls == [
        {"route": "direct", "direct_fits": True, "iteration": 1},
        {"route": "direct", "direct_fits": True, "iteration": 2},
        {"route": "orchestrated", "direct_fits": None, "iteration": 2},
    ]
    feedback = pipeline.call_details[1]["revision_feedback"]
    assert feedback and "presentation order" in feedback
    # The Orchestrator receives it, and the Predictor's copy is cleared.
    assert pipeline.orchestrated == [2]
    assert pipeline.orchestrated_feedback == [feedback]
    assert pipeline.call_details[2]["revision_feedback"] == ""
    assert [a["cause"] for a in result["routing"]["attempts"]] == ["measurement", "context_length"]


def test_orchestrated_outputs_carry_the_parsed_deviation_tree(pipeline):
    pipeline.measure_error = RuntimeError("tokenizer exploded")
    pipeline.run("auto")
    assert pipeline.predictor_calls[0]["route"] == "orchestrated"
    assert pipeline.call_details[0]["has_deviation_tree"] is True


def test_the_length_fallback_output_carries_the_parsed_deviation_tree(pipeline):
    pipeline.direct_error = RuntimeError("maximum context length exceeded")
    pipeline.run("auto")
    assert pipeline.predictor_calls[-1]["route"] == "orchestrated"
    assert pipeline.call_details[-1]["has_deviation_tree"] is True


def test_direct_fits_compares_with_the_predictor_budget_not_the_routing_threshold(pipeline):
    # A routing threshold below the record size, in never mode: the attempt is
    # direct, and the record does fit the Predictor itself (100 <= 1000).
    result = pipeline.run("never", threshold=50)
    assert result["routing"]["initial"]["budget_tokens"] == 50
    assert pipeline.predictor_calls == [{"route": "direct", "direct_fits": True, "iteration": 1}]


def test_a_decision_rejection_the_route_cannot_fix_stops_without_orchestrating(pipeline):
    # Only an unknown output scale: orchestration would not change it.
    pipeline.quality = {1: {"default_scale_outputs": ["root.score"]}}
    result = pipeline.run("auto", predictor_model="typesafe/jev-1.13")
    assert pipeline.orchestrated == []
    assert result["iterations"] == 1
    assert result["verdict"] == Verdict.UNSATISFACTORY.value
    assert [a["cause"] for a in result["routing"]["attempts"]] == ["measurement"]


def test_a_decision_rejection_about_the_evidence_escalates_to_orchestration(pipeline):
    pipeline.quality = {1: {"max_instability": 2.0, "mean_instability": 2.0}}
    result = pipeline.run("auto", predictor_model="typesafe/jev-1.13")
    assert pipeline.orchestrated == [2]
    assert [c["route"] for c in pipeline.predictor_calls] == ["direct", "orchestrated"]
    attempts = result["routing"]["attempts"]
    assert [(a["route"], a["cause"], a["escalated"]) for a in attempts] == [
        ("direct", "measurement", False),
        ("orchestrated", "critic", True),
    ]
    assert attempts[1]["reason"] == "The critic rejected the direct attempt; distilling the record with the orchestration workflow."
    assert pipeline.orchestrated_feedback[0]  # the critic's feedback goes to the Orchestrator


def test_context_length_error_in_never_mode_is_raised(pipeline):
    pipeline.direct_error = RuntimeError("maximum context length exceeded")
    with pytest.raises(RuntimeError, match="maximum context length exceeded"):
        pipeline.run("never")
    assert pipeline.orchestrated == []


def test_other_errors_on_the_direct_route_are_raised(pipeline):
    pipeline.direct_error = RuntimeError("HTTP 500 from the provider")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        pipeline.run("auto")
    assert pipeline.orchestrated == []
    assert [c["route"] for c in pipeline.predictor_calls] == ["direct"]


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Error code: 400 - This model's maximum context length is 65536 tokens", True),
        ("context_length_exceeded", True),
        ("prompt is too long: 250000 tokens > 200000 maximum", True),
        ("Input exceeds the context window", True),
        ("Too many tokens in the request", True),
        ("Context length of the prompt exceeded", True),
        ("HTTP 413: request too long", True),
        ("token limit reached for this model", True),
        ("input exceeds the maximum allowed size", True),
        ("Empty response (finish_reason=length)", False),
        ("HTTP 429 rate limited", False),
        ("token expired", False),
        ("state exceeds 32000 tokens", False),
    ],
)
def test_context_length_error_detection(message, expected):
    assert is_context_length_error(RuntimeError(message)) is expected
    # main uses the shared detector.
    assert main_mod.is_context_length_error is is_context_length_error


def test_a_decision_length_error_is_a_context_length_error():
    rejected = DecisionRequestError("Decision request returned HTTP 422", status=422, body="state exceeds the limit")
    assert rejected.looks_like_length_error
    assert is_context_length_error(rejected) is True
    # The same body on a server error is not a length error.
    assert is_context_length_error(DecisionRequestError("HTTP 500", status=500, body="state exceeds the limit")) is False
