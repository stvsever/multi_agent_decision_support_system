"""
Structured decision models: the frozen question book, the typed questions asked
per request, and the state (the evidence a decision model reads) packed to the
model's limit.

The companion LLM that compiles a question book is always a stub here, and the
book cache is pointed at a temporary directory.
"""

import contextlib
import io
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.config.settings import reload_settings
from src.full_stack.backend.data.models.prediction_task import (
    PredictionMode,
    PredictionTaskNode,
    PredictionTaskSpec,
)
from src.full_stack.backend.decision import questions as questions_module
from src.full_stack.backend.decision import state as state_module
from src.full_stack.backend.utils.core import record_rendering
from src.full_stack.backend.decision.questions import (
    COMPILER_SYSTEM_PROMPT,
    QuestionBook,
    QuestionSet,
    add_fine_regression_questions,
    asked_summary,
    build_question_book,
    cache_dir,
    deterministic_book,
    placeholder_definition,
    round_one_questions,
    task_hash,
)
from src.full_stack.backend.decision.state import (
    PACKING_LADDER,
    PackedState,
    StateSection,
    build_sections,
    measurement_rows,
    pack_state,
    render_deviation_profile,
    render_measurements,
)
from src.full_stack.backend.utils.core.data_loader import DataLoader
from src.full_stack.backend.utils.core.input_routing import build_direct_executor_output
from src.full_stack.backend.utils.token_packer import count_tokens

REPO = Path(__file__).resolve().parents[2]
SUBJ_001 = REPO / "src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO"
TRUNCATION_MARK = "[truncated to fit the decision model state]"


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    """No socket may open, and question books are cached in a temporary folder."""

    def _blocked(*_args, **_kwargs):
        raise RuntimeError("network access is not allowed in these tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setenv("COMPASS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path / "compass_home"))


def _cl100k(text: str) -> int:
    return count_tokens(text or "", model_hint="cl100k_base")


# ---------------------------------------------------------------------------
# Task specifications
# ---------------------------------------------------------------------------
def _binary_spec(**extra) -> PredictionTaskSpec:
    return PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="dx",
            display_name="Major depressive disorder",
            mode=PredictionMode.BINARY_CLASSIFICATION,
            class_labels=["MDD", "CONTROL"],
            **extra,
        )
    )


def _multiclass_spec(labels=("none", "mild", "severe"), **extra) -> PredictionTaskSpec:
    return PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="severity",
            display_name="Depression severity",
            mode=PredictionMode.MULTICLASS_CLASSIFICATION,
            class_labels=list(labels),
            **extra,
        )
    )


def _regression_spec(scale=None) -> PredictionTaskSpec:
    scales = {"madrs_total": scale} if scale is not None else {}
    return PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="madrs",
            display_name="MADRS total",
            mode=PredictionMode.UNIVARIATE_REGRESSION,
            regression_outputs=["madrs_total"],
            output_scales=scales,
        )
    )


def _hierarchical_spec(*, root_defs=None, madrs_scale=True) -> PredictionTaskSpec:
    scales = {"hamd_total": {}}
    if madrs_scale:
        scales["madrs_total"] = {"min": 0, "max": 60, "integer": True, "unit": "points"}
    return PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="severity",
            display_name="Depression severity",
            mode=PredictionMode.MULTICLASS_CLASSIFICATION,
            class_labels=["none", "mild", "severe"],
            class_definitions=dict(root_defs or {}),
            children=[
                PredictionTaskNode(
                    node_id="scores",
                    display_name="Symptom scores",
                    mode=PredictionMode.MULTIVARIATE_REGRESSION,
                    regression_outputs=["madrs_total", "hamd_total"],
                    output_scales=scales,
                ),
                PredictionTaskNode(
                    node_id="response",
                    display_name="Treatment response",
                    mode=PredictionMode.BINARY_CLASSIFICATION,
                    class_labels=["responder", "non_responder"],
                ),
            ],
        )
    )


def _fully_specified_spec() -> PredictionTaskSpec:
    return PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="dx",
            display_name="Major depressive disorder",
            mode=PredictionMode.BINARY_CLASSIFICATION,
            class_labels=["MDD", "CONTROL"],
            class_definitions={"MDD": "Meets DSM-5 criteria for MDD.", "CONTROL": "No psychiatric diagnosis."},
            children=[
                PredictionTaskNode(
                    node_id="madrs",
                    display_name="MADRS total",
                    mode=PredictionMode.UNIVARIATE_REGRESSION,
                    regression_outputs=["madrs_total"],
                    output_scales={"madrs_total": {"min": 0, "max": 60, "integer": True}},
                )
            ],
        )
    )


def _of_kind(qs: QuestionSet, kind: str, node_id: str = None):
    return [q for q in qs.items if q.kind == kind and (node_id is None or q.node_id == node_id)]


# ---------------------------------------------------------------------------
# Question sets
# ---------------------------------------------------------------------------
def test_binary_node_gets_a_noul_per_label_and_the_choice_in_both_orders():
    spec = _binary_spec(class_definitions={"MDD": "Meets DSM-5 criteria.", "CONTROL": "No diagnosis."})
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)

    nouls = _of_kind(qs, "binary_noul", "dx")
    choices = _of_kind(qs, "choice", "dx")
    # One Noul per label, each a single literal condition.
    assert [q.variant for q in nouls] == ["noul_1", "noul_2"]
    first, second = nouls
    assert first.order == ["MDD", "CONTROL"]
    assert second.order == ["CONTROL", "MDD"]
    for q in nouls:
        assert q.payload["type"] == "noul"
        assert set(q.payload["criteria"]) == {"true", "false"}
    # Label neutral: the node's question (which may lean toward one label) is not part of a Noul.
    assert first.payload["instructions"] == (
        "Does this participant belong to the 'MDD' group? "
        "Use `task_context` for definitions and the participant record for evidence."
    )
    assert first.payload["criteria"] == {"true": "MDD: Meets DSM-5 criteria.", "false": "CONTROL: No diagnosis."}
    assert second.payload["instructions"] == (
        "Does this participant belong to the 'CONTROL' group? "
        "Use `task_context` for definitions and the participant record for evidence."
    )
    assert second.payload["criteria"] == {"true": "CONTROL: No diagnosis.", "false": "MDD: Meets DSM-5 criteria."}

    assert [q.variant for q in choices] == ["order1", "order2"]
    assert [q.order for q in choices] == [["MDD", "CONTROL"], ["CONTROL", "MDD"]]
    for q in choices:
        assert q.payload["type"] == "choice"
        # The criteria map is presented in the question's order.
        assert list(q.payload["criteria"]) == q.order
        assert q.payload["criteria"]["MDD"] == "Meets DSM-5 criteria."
    # Four questions per binary node, then the evidence sufficiency Noul.
    assert [q.kind for q in qs.items] == ["binary_noul", "binary_noul", "choice", "choice", "sufficiency"]
    assert len(qs) == 2 + 2 + 1


def test_binary_noul_without_definitions_uses_the_generic_group_text():
    spec = _binary_spec()
    nouls = _of_kind(round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10), "binary_noul")
    assert nouls[0].payload["criteria"] == {
        "true": "MDD: The participant belongs to the 'MDD' group.",
        "false": "CONTROL: The participant belongs to the 'CONTROL' group.",
    }


def test_more_than_255_class_labels_are_refused_before_any_request():
    labels = [f"label_{i:03d}" for i in range(256)]
    spec = _multiclass_spec(labels=labels)
    with pytest.raises(ValueError, match="256 class labels.*at most 255"):
        round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    ok = _multiclass_spec(labels=labels[:255])
    qs = round_one_questions(ok, deterministic_book(ok), choice_orders=3, score_levels=10)
    assert all(len(q.payload["criteria"]) == 255 for q in _of_kind(qs, "choice"))


def test_binary_node_always_asks_both_orders_even_with_one_choice_order():
    spec = _binary_spec()
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=1, score_levels=10)
    assert [q.order for q in _of_kind(qs, "choice")] == [["MDD", "CONTROL"], ["CONTROL", "MDD"]]


def test_multiclass_rotations_put_every_label_first_and_include_both_directions():
    spec = _multiclass_spec()
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    choices = _of_kind(qs, "choice", "severity")
    orders = [q.order for q in choices]
    assert len(orders) == 3
    labels = ["none", "mild", "severe"]
    assert all(sorted(order) == sorted(labels) for order in orders)
    assert {order[0] for order in orders} == set(labels)
    assert labels in orders
    assert list(reversed(labels)) in orders
    assert not _of_kind(qs, "binary_noul")
    for q in choices:
        assert list(q.payload["criteria"]) == q.order
        assert q.payload["criteria"]["mild"] == "The participant belongs to the 'mild' group."


@pytest.mark.parametrize("choice_orders, expected", [(1, 1), (2, 2), (3, 3), (7, 3)])
def test_multiclass_order_count_is_capped_by_the_label_count(choice_orders, expected):
    spec = _multiclass_spec()
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=choice_orders, score_levels=10)
    orders = [q.order for q in _of_kind(qs, "choice")]
    assert len(orders) == expected
    assert len({tuple(o) for o in orders}) == expected
    if expected >= 2:
        assert ["severe", "mild", "none"] in orders


def test_multiclass_with_five_labels_keeps_three_distinct_orders():
    labels = ["a", "b", "c", "d", "e"]
    spec = _multiclass_spec(labels=labels)
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    orders = [q.order for q in _of_kind(qs, "choice")]
    assert len(orders) == 3
    assert labels in orders and list(reversed(labels)) in orders
    assert len({o[0] for o in orders}) == 3


def test_regression_gets_ascending_and_descending_mirror_scores():
    spec = _regression_spec({"min": 0, "max": 60, "integer": True, "unit": "points"})
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    scores = _of_kind(qs, "score_coarse", "madrs")
    assert [q.variant for q in scores] == ["ascending", "descending"]
    asc, desc = scores
    assert asc.output == desc.output == "madrs_total"
    assert asc.order == list(range(10))
    assert desc.order == list(reversed(range(10)))
    assert asc.payload["type"] == desc.payload["type"] == "score"
    assert len(asc.payload["criteria"]) == 10
    assert desc.payload["criteria"] == list(reversed(asc.payload["criteria"]))
    assert asc.payload["instructions"] == desc.payload["instructions"]
    # Integer levels: 0 to 6 (seven values), then six values each up to 55 to 60.
    assert asc.payload["criteria"][0].startswith("0 to 6 points; ")
    assert asc.payload["criteria"][1].startswith("7 to 12 points; ")
    assert desc.payload["criteria"][0].startswith("55 to 60 points; ")
    assert asc.payload["instructions"] == (
        "Where does this participant's madrs total (madrs_total) lie on its range of 0 to 60 points? "
        "Choose the level that describes the participant's value, using `task_context` and the participant record."
    )
    assert [b.center for b in asc.bins] == [b.center for b in desc.bins]


def test_coarse_score_instruction_states_the_meaning_of_the_ends():
    spec = _regression_spec(
        {
            "min": 0,
            "max": 60,
            "integer": True,
            "unit": "points",
            "description": "MADRS total score",
            "low_meaning": "no symptoms",
            "high_meaning": "most severe",
        }
    )
    asc = _of_kind(round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10), "score_coarse")[0]
    assert asc.payload["instructions"] == (
        "Where does this participant's MADRS total score (madrs_total) lie on its range of 0 to 60 points? "
        "Low values mean no symptoms; high values mean most severe. "
        "Choose the level that describes the participant's value, using `task_context` and the participant record."
    )
    assert asc.payload["criteria"][0].endswith("; low end: no symptoms")
    assert asc.payload["criteria"][-1].endswith("; high end: most severe")
    assert not any("end:" in c for c in asc.payload["criteria"][1:-1])


def test_exact_integer_output_gets_one_level_per_value():
    spec = _regression_spec({"min": 0, "max": 6, "integer": True})
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    asc, desc = _of_kind(qs, "score_coarse")
    assert len(asc.payload["criteria"]) == 7
    assert all(b.exact for b in asc.bins)
    assert desc.payload["criteria"] == list(reversed(asc.payload["criteria"]))


def test_integer_scale_instruction_announces_integer_bounds():
    spec = _regression_spec({"min": 0, "max": 6, "integer": True})
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    asc = _of_kind(qs, "score_coarse")[0]
    assert "lie on its range of 0 to 6? " in asc.payload["instructions"]
    assert "0.00" not in asc.payload["instructions"]


def test_regression_without_a_scale_uses_the_default_standardized_scale():
    spec = _regression_spec()
    book = deterministic_book(spec)
    assert book.nodes["madrs"].scales["madrs_total"].source == "default"
    assert book.default_scale_outputs() == ["madrs.madrs_total"]
    qs = round_one_questions(spec, book, choice_orders=3, score_levels=10)
    asc = _of_kind(qs, "score_coarse")[0]
    assert len(asc.payload["criteria"]) == 10
    assert "lie on its range of -3.00 to 3.00 standardized units?" in asc.payload["instructions"]


def test_every_request_carries_one_evidence_sufficiency_noul():
    spec = _hierarchical_spec()
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    suff = _of_kind(qs, "sufficiency")
    assert len(suff) == 1
    q = suff[0]
    assert q.node_id == "severity"
    assert q.payload["type"] == "noul"
    assert set(q.payload["criteria"]) == {"true", "false"}
    assert "enough relevant evidence" in q.payload["instructions"]
    assert "(and 2 related outputs)" in q.payload["instructions"]
    single = _binary_spec()
    single_q = _of_kind(round_one_questions(single, deterministic_book(single), choice_orders=3, score_levels=10), "sufficiency")[0]
    assert "related outputs" not in single_q.payload["instructions"]


def test_hierarchical_round_one_covers_every_node():
    spec = _hierarchical_spec()
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    assert len(_of_kind(qs, "choice", "severity")) == 3
    assert len(_of_kind(qs, "binary_noul", "response")) == 2
    assert len(_of_kind(qs, "choice", "response")) == 2
    # 3 severity choices, 2 x 2 coarse scores, 4 response questions, sufficiency.
    assert len(qs) == 3 + 4 + 4 + 1
    coarse = _of_kind(qs, "score_coarse", "scores")
    assert sorted({q.output for q in coarse}) == ["hamd_total", "madrs_total"]
    assert len(coarse) == 4
    summary = asked_summary(qs)
    assert len(summary) == len(qs)
    assert set(summary[0]) == {"qid", "node_id", "kind", "output", "variant", "order"}


def test_question_ids_are_unique_across_question_set_prefixes():
    spec = _hierarchical_spec()
    book = deterministic_book(spec)
    round_one = round_one_questions(spec, book, choice_orders=3, score_levels=10)
    scores_node = spec.root.children[0]
    scale = book.nodes["scores"].scales["madrs_total"]
    coarse = scale.coarse_grid(10)
    # The window of coarse levels 2 to 4 (13 to 18, 19 to 24, 25 to 30).
    lo, hi = coarse[2].lo, coarse[4].hi
    assert (lo, hi) == (12.5, 30.5)
    round_two = QuestionSet(prefix="r")
    fine = scale.fine_grid(lo, hi, 10)
    add_fine_regression_questions(round_two, scores_node, "madrs_total", scale, fine, lo, hi)
    hamd = book.nodes["scores"].scales["hamd_total"]
    add_fine_regression_questions(round_two, scores_node, "hamd_total", hamd, hamd.fine_grid(-1.2, 0.6, 10), -1.2, 0.6)

    ids_one = [q.qid for q in round_one.items]
    ids_two = [q.qid for q in round_two.items]
    assert len(set(ids_one)) == len(ids_one)
    assert len(set(ids_two)) == len(ids_two) == 4
    assert not set(ids_one) & set(ids_two)
    assert all(qid.startswith("q") for qid in ids_one)
    assert all(qid.startswith("r") for qid in ids_two)
    assert ids_one[0] == "q001" and ids_two[0] == "r001"
    assert set(round_one.payload()) == set(ids_one)

    fine_qs = [q for q in round_two.items if q.output == "madrs_total"]
    assert [q.kind for q in fine_qs] == ["score_fine", "score_fine"]
    assert [q.variant for q in fine_qs] == ["ascending", "descending"]
    assert fine_qs[1].payload["criteria"] == list(reversed(fine_qs[0].payload["criteria"]))
    # The window [12.5, 30.5] holds the integers 13 to 30, printed as integers.
    assert fine_qs[0].payload["instructions"] == (
        "Suppose this participant's madrs total (madrs_total) is somewhere from 13 to 30 points. "
        "Within that part of the range, choose the level that describes the participant's value, "
        "using `task_context` and the participant record."
    )
    assert len(fine_qs[0].payload["criteria"]) == 10
    assert fine_qs[0].payload["criteria"][0] == "13 to 14 points; in the low part of the scale"
    assert fine_qs[0].payload["criteria"][-1] == "30 points; in the middle of the scale"
    # A continuous window is printed at a tenth of its width (0.18 wide: two decimals).
    hamd_qs = [q for q in round_two.items if q.output == "hamd_total"]
    assert hamd_qs[0].payload["instructions"].startswith(
        "Suppose this participant's hamd total (hamd_total) is somewhere from -1.20 to 0.60 standardized units. "
    )


def test_question_set_size_helpers():
    spec = _binary_spec()
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    sizes = [len(json.dumps(q.payload)) for q in qs.items]
    assert qs.longest_question_chars() == max(sizes)
    assert qs.all_chars() == sum(sizes)
    assert QuestionSet().longest_question_chars() == 0


# ---------------------------------------------------------------------------
# Question book
# ---------------------------------------------------------------------------
class _CompilerStub:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def __call__(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return json.loads(json.dumps(self.payload))


def _compiled_payload():
    return {
        "nodes": {
            "severity": {
                "question": "How severe is this participant's depression?",
                "label_definitions": {
                    "none": "COMPILED none",
                    "mild": "COMPILED mild",
                    "severe": "COMPILED severe",
                },
            },
            "scores": {
                "question": "What are this participant's symptom scores?",
                "scales": {
                    "madrs_total": {"min": 0, "max": 99, "integer": True},
                    "hamd_total": {"min": 0, "max": 52, "integer": True, "unit": "points", "description": "HAM-D 17 total"},
                },
            },
            "response": {
                "question": "Will this participant respond to treatment?",
                "label_definitions": {
                    "responder": "At least 50 percent MADRS reduction.",
                    "non_responder": "Less than 50 percent MADRS reduction.",
                },
            },
        }
    }


def _placeholder_payload():
    """Every scale compiles, but one question and one class definition stay generic."""
    payload = _compiled_payload()
    payload["nodes"]["response"]["question"] = ""
    del payload["nodes"]["response"]["label_definitions"]["non_responder"]
    return payload


def _book_files():
    folder = cache_dir()
    return sorted(folder.glob("*.json")) if folder.exists() else []


def test_cache_dir_follows_the_environment(tmp_path):
    assert cache_dir() == tmp_path / "cache" / "decision_question_books"


def test_fully_specified_task_needs_no_compiler():
    spec = _fully_specified_spec()
    book = build_question_book(spec, context="Adult depression cohort.", llm_json=None)
    assert book.compiled_by == "task_spec"
    assert book.nodes["dx"].label_definitions["MDD"] == "Meets DSM-5 criteria for MDD."
    scale = book.nodes["madrs"].scales["madrs_total"]
    assert (scale.minimum, scale.maximum, scale.source) == (0.0, 60.0, "task_spec")
    assert _book_files() == []


def test_without_a_compiler_generic_definitions_and_default_scales_are_used():
    spec = _hierarchical_spec()
    book = build_question_book(spec, context="ctx", llm_json=None)
    assert book.compiled_by == "deterministic"
    assert any("No companion LLM available" in note for note in book.notes)
    assert book.nodes["severity"].label_definitions["mild"] == "The participant belongs to the 'mild' group."
    assert book.nodes["scores"].scales["hamd_total"].source == "default"
    assert book.nodes["scores"].scales["madrs_total"].source == "task_spec"
    assert _book_files() == []


def test_compiled_book_keeps_task_spec_definitions_and_scales():
    spec = _hierarchical_spec(root_defs={"none": "SPEC none: MADRS below 7."})
    compiler = _CompilerStub(_compiled_payload())
    book = build_question_book(spec, context="Adult outpatients.", llm_json=compiler, compiler_label="stub/compiler")

    assert len(compiler.calls) == 1
    system_prompt, user_prompt = compiler.calls[0]
    assert system_prompt == COMPILER_SYSTEM_PROMPT
    assert "## Task specification" in user_prompt
    assert "Adult outpatients." in user_prompt
    for node_id in ("severity", "scores", "response"):
        assert node_id in user_prompt

    assert book.compiled_by == "stub/compiler"
    severity = book.nodes["severity"]
    assert severity.question == "How severe is this participant's depression?"
    assert severity.label_definitions["none"] == "SPEC none: MADRS below 7."
    assert severity.label_definitions["mild"] == "COMPILED mild"
    assert severity.label_definitions["severe"] == "COMPILED severe"

    scores = book.nodes["scores"]
    madrs = scores.scales["madrs_total"]
    assert (madrs.maximum, madrs.source) == (60.0, "task_spec")
    hamd = scores.scales["hamd_total"]
    assert (hamd.minimum, hamd.maximum, hamd.integer, hamd.source) == (0.0, 52.0, True, "compiler")
    assert hamd.unit == "points" and hamd.description == "HAM-D 17 total"

    response = book.nodes["response"]
    assert response.question == "Will this participant respond to treatment?"
    assert response.question_is_default is False
    assert response.label_definitions["non_responder"] == "Less than 50 percent MADRS reduction."
    # Complete: no default scale, no generic wording left.
    assert book.default_scale_outputs() == [] and book.placeholder_items() == []
    assert book.notes == []


def test_deterministic_book_lists_its_placeholders():
    book = deterministic_book(_hierarchical_spec())
    # A new book version, so books cached before these rules are not reused.
    assert book.version == "2026-10-04.2"
    assert all(node.question_is_default for node in book.nodes.values())
    assert book.placeholder_items() == [
        "severity: question",
        "severity.none: definition",
        "severity.mild: definition",
        "severity.severe: definition",
        "scores: question",
        "response: question",
        "response.responder: definition",
        "response.non_responder: definition",
    ]
    assert placeholder_definition("mild") == "The participant belongs to the 'mild' group."
    # Definitions given in the task specification are not placeholders.
    spec_defs = deterministic_book(_hierarchical_spec(root_defs={"none": "MADRS below 7."}))
    assert "severity.none: definition" not in spec_defs.placeholder_items()
    # question_is_default survives the cache format.
    clone = QuestionBook.from_dict(json.loads(json.dumps(book.to_dict())))
    assert clone.placeholder_items() == book.placeholder_items()


def test_placeholder_only_compile_is_retried_and_never_cached():
    spec = _hierarchical_spec()
    compiler = _CompilerStub(_placeholder_payload())
    book = build_question_book(spec, context="ctx", llm_json=compiler, compiler_label="stub")
    # Every scale compiled; still incomplete because of the generic wording.
    assert len(compiler.calls) == 2
    assert book.default_scale_outputs() == []
    assert book.placeholder_items() == ["response: question", "response.non_responder: definition"]
    # The last merged book is returned: what did compile is kept.
    assert book.compiled_by == "stub"
    response = book.nodes["response"]
    assert response.question.startswith("Which group does this participant belong to")
    assert response.question_is_default is True
    assert response.label_definitions["responder"] == "At least 50 percent MADRS reduction."
    assert book.notes == [
        "Compilation incomplete (ValueError: compiled book left response: question, "
        "response.non_responder: definition undefined)."
    ]
    assert _book_files() == []


def test_compiled_reply_without_the_nodes_wrapper_is_accepted():
    spec = _hierarchical_spec()
    unwrapped = _compiled_payload()["nodes"]
    compiler = _CompilerStub(unwrapped)
    book = build_question_book(spec, context="ctx", llm_json=compiler, compiler_label="stub")
    assert len(compiler.calls) == 1
    assert book.notes == []
    assert book.nodes["severity"].label_definitions["mild"] == "COMPILED mild"
    assert book.nodes["scores"].scales["hamd_total"].source == "compiler"
    assert [f.name for f in _book_files()] == [f"{book.task_hash}.json"]


def test_cached_book_file_is_readable_by_others():
    spec = _hierarchical_spec()
    book = build_question_book(spec, context="ctx", llm_json=_CompilerStub(_compiled_payload()), compiler_label="stub")
    path = cache_dir() / f"{book.task_hash}.json"
    assert (path.stat().st_mode & 0o777) == 0o644


def test_compiled_book_is_cached_and_reused_without_a_second_call():
    spec = _hierarchical_spec()
    first = build_question_book(spec, context="ctx", llm_json=_CompilerStub(_compiled_payload()), compiler_label="stub")
    files = _book_files()
    assert [f.name for f in files] == [f"{first.task_hash}.json"]
    assert first.task_hash == task_hash(spec, "ctx")

    never = _CompilerStub(error=AssertionError("the cached book must be used"))
    second = build_question_book(spec, context="ctx", llm_json=never)
    assert never.calls == []
    assert second.to_dict() == first.to_dict()
    assert isinstance(QuestionBook.from_dict(json.loads(files[0].read_text())), QuestionBook)

    # Another context is another instrument.
    other = _CompilerStub(_compiled_payload())
    third = build_question_book(spec, context="another study", llm_json=other)
    assert len(other.calls) == 1
    assert third.task_hash != first.task_hash
    assert len(_book_files()) == 2


def test_cached_book_is_used_even_without_a_compiler():
    spec = _hierarchical_spec()
    compiled = build_question_book(spec, context="ctx", llm_json=_CompilerStub(_compiled_payload()), compiler_label="stub")
    again = build_question_book(spec, context="ctx", llm_json=None)
    assert again.compiled_by == "stub"
    assert again.to_dict() == compiled.to_dict()


INCOMPLETE_SCALE_NOTE = "Compilation incomplete (ValueError: compiled book left scores.hamd_total undefined)."
GARBAGE_NOTE = (
    "Compilation incomplete (ValueError: compiled book left scores.hamd_total, severity: question, "
    "severity.none: definition, severity.mild: definition, severity.severe: definition, scores: question, "
    "response: question, response.responder: definition and more undefined)."
)


def test_compiler_failure_is_retried_once_then_falls_back_uncached():
    spec = _hierarchical_spec()
    compiler = _CompilerStub(error=RuntimeError("provider down"))
    book = build_question_book(spec, context="ctx", llm_json=compiler)
    # One retry, so two calls in all.
    assert len(compiler.calls) == 2
    assert book.compiled_by == "deterministic"
    assert book.notes == ["Compilation incomplete (RuntimeError: provider down)."]
    assert book.nodes["severity"].label_definitions["mild"] == "The participant belongs to the 'mild' group."
    assert book.nodes["scores"].scales["madrs_total"].source == "task_spec"
    assert book.default_scale_outputs() == ["scores.hamd_total"]
    # A failed compilation is never cached: the next participant tries again.
    assert _book_files() == []
    again = _CompilerStub(_compiled_payload())
    compiled = build_question_book(spec, context="ctx", llm_json=again, compiler_label="stub")
    assert len(again.calls) == 1
    assert compiled.compiled_by == "stub"
    assert [f.name for f in _book_files()] == [f"{compiled.task_hash}.json"]


def test_compiler_succeeding_on_the_retry_is_cached():
    spec = _hierarchical_spec()

    class _FlakyCompiler(_CompilerStub):
        def __call__(self, system_prompt, user_prompt):
            self.calls.append((system_prompt, user_prompt))
            if len(self.calls) == 1:
                raise TimeoutError("slow provider")
            return json.loads(json.dumps(_compiled_payload()))

    compiler = _FlakyCompiler()
    book = build_question_book(spec, context="ctx", llm_json=compiler, compiler_label="stub")
    assert len(compiler.calls) == 2
    assert book.compiled_by == "stub"
    assert book.notes == []
    assert book.nodes["scores"].scales["hamd_total"].source == "compiler"
    assert book.default_scale_outputs() == []
    assert [f.name for f in _book_files()] == [f"{book.task_hash}.json"]


HAMD_MERGE_NOTE = "No usable compiled scale for 'scores.hamd_total'; default scale kept."


def test_compiler_returning_garbage_keeps_defaults_and_is_not_cached():
    spec = _hierarchical_spec()
    compiler = _CompilerStub({"unexpected": True})
    book = build_question_book(spec, context="ctx", llm_json=compiler, compiler_label="stub")
    # The merged book still had hamd_total on the default scale: retried once,
    # then the last merged book is returned with both notes, and not cached.
    assert len(compiler.calls) == 2
    assert book.compiled_by == "stub"
    assert book.nodes["scores"].scales["hamd_total"].source == "default"
    # Only the notes of the last attempt (each attempt merges into a fresh book);
    # the incomplete note names the first eight gaps.
    assert book.notes == [HAMD_MERGE_NOTE, GARBAGE_NOTE]
    assert book.default_scale_outputs() == ["scores.hamd_total"]
    assert len(book.placeholder_items()) == 8
    assert _book_files() == []


def test_unusable_compiled_scale_keeps_the_compiled_questions_but_is_not_cached():
    spec = _hierarchical_spec()
    payload = _compiled_payload()
    payload["nodes"]["scores"]["scales"]["hamd_total"] = {"min": 10, "max": 10}
    compiler = _CompilerStub(payload)
    book = build_question_book(spec, context="ctx", llm_json=compiler, compiler_label="stub")
    assert len(compiler.calls) == 2
    assert book.compiled_by == "stub"
    assert book.nodes["scores"].scales["hamd_total"].source == "default"
    # What did compile is kept: questions, definitions, the other scale.
    assert book.nodes["severity"].question == "How severe is this participant's depression?"
    assert book.nodes["severity"].label_definitions["mild"] == "COMPILED mild"
    assert book.nodes["response"].label_definitions["responder"] == "At least 50 percent MADRS reduction."
    assert book.nodes["scores"].scales["madrs_total"].source == "task_spec"
    assert book.notes == [HAMD_MERGE_NOTE, INCOMPLETE_SCALE_NOTE]
    assert _book_files() == []
    # The next participant compiles again; once complete, the book is cached.
    good = _CompilerStub(_compiled_payload())
    again = build_question_book(spec, context="ctx", llm_json=good, compiler_label="stub")
    assert len(good.calls) == 1
    assert again.notes == []
    assert [f.name for f in _book_files()] == [f"{again.task_hash}.json"]


def test_a_merged_book_survives_a_failed_retry():
    spec = _hierarchical_spec()

    class _MergedThenDown(_CompilerStub):
        def __call__(self, system_prompt, user_prompt):
            self.calls.append((system_prompt, user_prompt))
            if len(self.calls) == 1:
                payload = _compiled_payload()
                del payload["nodes"]["scores"]["scales"]["hamd_total"]
                return payload
            raise RuntimeError("provider down")

    compiler = _MergedThenDown()
    book = build_question_book(spec, context="ctx", llm_json=compiler, compiler_label="stub")
    assert len(compiler.calls) == 2
    # The first attempt's merged book, with the second attempt's error.
    assert book.compiled_by == "stub"
    assert book.nodes["severity"].label_definitions["severe"] == "COMPILED severe"
    assert book.notes == [HAMD_MERGE_NOTE, "Compilation incomplete (RuntimeError: provider down)."]
    assert _book_files() == []
    # Without a compiler the next call finds no cached book.
    assert build_question_book(spec, context="ctx", llm_json=None).compiled_by == "deterministic"


def test_book_content_hash_and_default_scale_outputs():
    spec = _hierarchical_spec()
    base = deterministic_book(spec, "ctx")
    same = deterministic_book(spec, "ctx")
    assert base.content_hash() == same.content_hash()
    assert len(base.content_hash()) == 16
    assert all(c in "0123456789abcdef" for c in base.content_hash())
    # The hash fingerprints the instrument (the nodes), not notes or provenance.
    same.notes.append("a note")
    same.compiled_by = "someone else"
    assert same.content_hash() == base.content_hash()
    changed = deterministic_book(spec, "ctx")
    changed.nodes["severity"].label_definitions["mild"] = "MADRS 7 to 19."
    assert changed.content_hash() != base.content_hash()
    # The hash survives a round trip through the cache format.
    assert QuestionBook.from_dict(json.loads(json.dumps(base.to_dict()))).content_hash() == base.content_hash()

    assert base.default_scale_outputs() == ["scores.hamd_total"]
    assert deterministic_book(_hierarchical_spec(madrs_scale=False)).default_scale_outputs() == [
        "scores.madrs_total",
        "scores.hamd_total",
    ]
    assert deterministic_book(_fully_specified_spec()).default_scale_outputs() == []


def test_failed_cache_write_leaves_no_file_behind(monkeypatch):
    spec = _hierarchical_spec()

    def _no_replace(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(questions_module.os, "replace", _no_replace)
    book = build_question_book(spec, context="ctx", llm_json=_CompilerStub(_compiled_payload()), compiler_label="stub")
    # The book is still returned; neither the cache file nor a temporary file remains.
    assert book.compiled_by == "stub"
    assert _book_files() == []
    assert not list(cache_dir().glob("*.tmp"))
    # The lock file of the task stays (it is reused by the next compilation).
    assert (cache_dir() / f"{book.task_hash}.lock").exists()


def test_cache_write_is_atomic_and_readable():
    spec = _hierarchical_spec()
    book = build_question_book(spec, context="ctx", llm_json=_CompilerStub(_compiled_payload()), compiler_label="stub")
    path = cache_dir() / f"{book.task_hash}.json"
    assert QuestionBook.from_dict(json.loads(path.read_text())).to_dict() == book.to_dict()
    assert not list(cache_dir().glob("*.tmp"))


def test_parallel_workers_compile_a_book_only_once():
    spec = _hierarchical_spec()
    calls = []
    start = threading.Barrier(2)

    def _slow_compiler(system_prompt, user_prompt):
        calls.append(user_prompt)
        time.sleep(0.3)  # long enough for the other worker to reach the lock
        return json.loads(json.dumps(_compiled_payload()))

    books = []

    def _worker():
        start.wait()
        books.append(build_question_book(spec, context="ctx", llm_json=_slow_compiler, compiler_label="stub"))

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert len(books) == 2
    # The second worker waited on the lock and then read the cached book.
    assert len(calls) == 1
    assert books[0].to_dict() == books[1].to_dict()
    assert len(_book_files()) == 1


def test_use_cache_false_never_touches_the_cache():
    spec = _hierarchical_spec()
    compiler = _CompilerStub(_compiled_payload())
    build_question_book(spec, context="ctx", llm_json=compiler, use_cache=False)
    build_question_book(spec, context="ctx", llm_json=compiler, use_cache=False)
    assert len(compiler.calls) == 2
    assert _book_files() == []


def test_unreadable_cache_is_ignored_and_rewritten():
    spec = _hierarchical_spec()
    path = cache_dir() / f"{task_hash(spec, 'ctx')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    compiler = _CompilerStub(_compiled_payload())
    book = build_question_book(spec, context="ctx", llm_json=compiler, compiler_label="stub")
    assert len(compiler.calls) == 1
    assert book.compiled_by == "stub"
    assert json.loads(path.read_text())["compiled_by"] == "stub"


def test_question_book_round_trips_through_dict():
    spec = _hierarchical_spec()
    book = build_question_book(spec, context="ctx", llm_json=_CompilerStub(_compiled_payload()), compiler_label="stub")
    clone = QuestionBook.from_dict(json.loads(json.dumps(book.to_dict())))
    assert clone.to_dict() == book.to_dict()
    assert clone.nodes["scores"].scales["hamd_total"].maximum == 52.0


# ---------------------------------------------------------------------------
# State: sections
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def direct_output():
    reload_settings()
    spec = _binary_spec()
    with contextlib.redirect_stdout(io.StringIO()):
        participant = DataLoader().load(SUBJ_001)
        out = build_direct_executor_output(
            participant,
            target_condition="MDD",
            control_condition="CONTROL",
            prediction_task_spec=spec,
        )
    return out


def test_build_sections_from_a_direct_executor_output(direct_output):
    sections, total = build_sections(executor_output=direct_output, task_context="Adult MDD cohort.", count=_cl100k)
    keys = [s.key for s in sections]
    assert keys == ["task_context", "clinical_record", "measurements", "deviation_profile"]
    # The direct route has no tool syntheses and no chunk evidence.
    assert "phenotype_synthesis" not in keys and "chunk_evidence" not in keys

    assert total == len(direct_output["coverage_ledger"]["all_features"]) > 0
    by_key = {s.key: s for s in sections}
    assert by_key["task_context"].variants[0] == "Adult MDD cohort."
    record = by_key["clinical_record"]
    assert record.variants[0] == str(direct_output["predictor_input"]["non_numerical_data_raw"]).strip()
    assert record.variant_labels == ["full", "first half", "first quarter"]
    size = _cl100k(record.variants[0])
    assert _cl100k(record.variants[1]) <= max(400, size // 2)
    assert _cl100k(record.variants[2]) <= max(300, size // 4)
    # The direct route's compact record and the decision state render leaves alike.
    assert by_key["measurements"].variants[0] == direct_output["direct_record"]["measurements"]

    measurements = by_key["measurements"]
    assert measurements.variant_labels[0] == "all leaves"
    assert measurements.feature_counts[0] == total
    assert all(a > b for a, b in zip(measurements.feature_counts, measurements.feature_counts[1:]))
    assert len(measurements.variants) == len(measurements.variant_labels) == len(measurements.feature_counts)
    for text, kept in zip(measurements.variants, measurements.feature_counts):
        assert sum(1 for line in text.splitlines() if line.startswith("- ")) == kept
    assert [s.priority for s in sections] == sorted(s.priority for s in sections)


def test_build_sections_without_task_context(direct_output):
    sections, _total = build_sections(executor_output=direct_output, task_context="   ", count=_cl100k)
    assert "task_context" not in [s.key for s in sections]


def test_state_uses_the_shared_record_renderers():
    assert state_module.measurement_rows is record_rendering.measurement_rows
    assert state_module.render_measurements is record_rendering.render_measurements
    assert state_module.render_deviation_profile is record_rendering.render_deviation_profile


def test_measurement_rows_keep_value_unit_and_z():
    multimodal = {
        "COGNITION": [
            {"feature_id": "ravlt", "field_name": "RAVLT total", "value": 38, "unit": "words", "z_score": -1.2, "domain": "COGNITION", "path_in_hierarchy": ["memory"]},
            {"feature_id": "tmt", "field_name": "TMT-B", "value": 1.5, "z_score": 1.5, "domain": "COGNITION", "path_in_hierarchy": ["executive"]},
            {"feature_id": "smoke", "field_name": "Smoking status", "value": "former", "z_score": None, "domain": "COGNITION", "path_in_hierarchy": []},
            {"feature_id": "cer", "field_name": "Cer(d18:1/16:0) High", "value": "High", "z_score": 2.4, "domain": "COGNITION", "path_in_hierarchy": ["lipids"]},
            {"feature_id": "sex", "field_name": "Sex: Female", "value": "Female", "z_score": None, "domain": "COGNITION", "path_in_hierarchy": []},
            {"feature_id": "phq", "field_name": "PHQ-9 total score", "value": 9, "z_score": 1.1, "domain": "COGNITION", "path_in_hierarchy": ["mood"]},
        ]
    }
    rows = measurement_rows(multimodal)
    assert rows[0] == ("COGNITION > memory", "RAVLT total = 38 words (z -1.20)", -1.2)
    # The loader falls back to the z-score as value; it is not printed twice.
    assert rows[1] == ("COGNITION > executive", "TMT-B (z +1.50)", 1.5)
    assert rows[2] == ("COGNITION", "Smoking status = former", None)
    # Only a label that ENDS with the value after ":" or "=" states it; "High"
    # merely appearing at the end of a lipid name does not.
    assert rows[3] == ("COGNITION > lipids", "Cer(d18:1/16:0) High = High (z +2.40)", 2.4)
    assert rows[4] == ("COGNITION", "Sex: Female", None)
    # "9" is inside "PHQ-9", but the label does not state the value: it is kept.
    assert rows[5] == ("COGNITION > mood", "PHQ-9 total score = 9 (z +1.10)", 1.1)

    text, kept = render_measurements(rows, min_abs_z=1.5)
    assert kept == 4  # |z| 1.5 and 2.4, plus the two categorical leaves
    assert "RAVLT" not in text and "Smoking status = former" in text and "PHQ-9" not in text


@pytest.mark.parametrize(
    "label, value, states",
    [
        ("PHQ-9 total score", 9, False),
        ("Age: 58.2 years", "58.2 years", True),
        ("Sex = Female", "Female", True),
        ("Sex=Female", "Female", True),
        ("Female", "Female", True),
        ("Handedness: Right", "Left", False),
        ("Cer(d18:1/16:0) High", "High", False),
        ("Smoking status", "former", False),
        ("Score: 12", "2", False),
        ("anything", "", True),
    ],
)
def test_label_states_value_is_anchored_at_the_end(label, value, states):
    assert record_rendering.label_states_value(label, value) is states


def test_leaf_lines_append_their_qualifiers():
    feat = {
        "field_name": "fasting_glucose",
        "value": "92 mg/dL",
        "z_score": -0.2,
        "qualifiers": {"ref_range": "70-99", "percentile": 42},
    }
    line, z = record_rendering.leaf_line(feat)
    assert line == "fasting_glucose = 92 mg/dL (z -0.20) [ref range 70-99; percentile 42]"
    assert z == -0.2
    # No qualifiers, no brackets.
    assert record_rendering.leaf_line({"field_name": "HDL", "value": 56, "z_score": 0.2, "qualifiers": {}})[0] == "HDL = 56 (z +0.20)"


def _deviation_tree():
    return {
        "root": {
            "node_name": "root",
            "z_score": 0.4,
            "children": [
                {
                    "node_name": "COGNITION",
                    "z_score": 1.1,
                    "children": [
                        {
                            "node_name": "memory",
                            "z_score": -1.3,
                            "children": [
                                {"node_name": "verbal", "z_score": -1.6, "children": [{"node_name": "leaf", "z_score": -2.0}]},
                                {"node_name": "visual", "z_score": None, "children": []},
                            ],
                        }
                    ],
                },
                {"node_name": "SLEEP", "z_score": 0.7, "children": []},
            ],
        }
    }


def test_render_deviation_profile_lists_every_scored_node():
    full = render_deviation_profile(_deviation_tree())
    # The root is not named; unscored nodes are skipped; childless scored nodes
    # are listed when nothing marks them as measurement leaves.
    assert full.splitlines() == [
        "COGNITION: +1.10",
        "COGNITION > memory: -1.30",
        "COGNITION > memory > verbal: -1.60",
        "COGNITION > memory > verbal > leaf: -2.00",
        "SLEEP: +0.70",
    ]
    assert render_deviation_profile(_deviation_tree(), max_depth=2).splitlines() == [
        "COGNITION: +1.10",
        "COGNITION > memory: -1.30",
        "SLEEP: +0.70",
    ]
    assert render_deviation_profile({}) == ""
    assert render_deviation_profile(None) == ""


def test_render_deviation_profile_skips_leaves_listed_in_the_measurements():
    # "COGNITION > memory > verbal" holds feature leaves in the measurements, so
    # its childless children duplicate them; the group itself stays.
    lines = render_deviation_profile(_deviation_tree(), leaf_groups={"COGNITION > memory > verbal"}).splitlines()
    assert lines == ["COGNITION: +1.10", "COGNITION > memory: -1.30", "COGNITION > memory > verbal: -1.60", "SLEEP: +0.70"]
    # A childless node that is itself a measurement group is not a leaf duplicate.
    groups = {"COGNITION > memory > verbal", "COGNITION > memory > verbal > leaf"}
    assert "COGNITION > memory > verbal > leaf: -2.00" in render_deviation_profile(_deviation_tree(), leaf_groups=groups)


def test_render_deviation_profile_prints_mean_absolute_scores_as_magnitudes():
    tree = {
        "root": {
            "node_name": "ROOT",
            "children": [
                {
                    "node_name": "BRAIN",
                    "z_score": 1.386,
                    "score_kind": "mean_abs",
                    "children": [
                        {"node_name": "Morphologics", "z_score": 0.9, "score_kind": "mean_abs", "children": []},
                        {"node_name": "Connectivity", "z_score": -0.5, "score_kind": "signed", "children": []},
                    ],
                }
            ],
        }
    }
    assert render_deviation_profile(tree).splitlines() == [
        "BRAIN: mean |z| 1.39",
        "BRAIN > Morphologics: mean |z| 0.90",
        "BRAIN > Connectivity: -0.50",
    ]


def test_render_deviation_profile_falls_back_to_numeric_domain_summaries():
    summaries = {
        "domain_summaries": {
            "BRAIN": {"mean_abs_score": 1.234, "n_leaves": 4},
            "COGNITION": {"mean_abs_score": None, "n_leaves": 2},
            "GENOMICS": {"mean_abs_score": "high"},
            "SLEEP": "not a summary",
        }
    }
    assert render_deviation_profile(summaries) == "BRAIN: mean |z| 1.23"
    assert render_deviation_profile({"domain_summaries": {"COGNITION": {"mean_abs_score": None}}}) == ""


def test_render_deviation_profile_reads_the_parsed_model(tmp_path):
    path = tmp_path / "hierarchical_deviation_map.json"
    path.write_text(
        json.dumps(
            {
                "BRAIN": {
                    "_stats": {"mean_abs_score": 1.386, "n_leaves": 3},
                    "Morphologics": {
                        "_stats": {"mean_abs_score": 0.9},
                        "left_hippocampus_volume": {"z_score": -1.2},
                    },
                },
                "COGNITION": {"memory": {"z_score": 0.4}},
            }
        )
    )
    with contextlib.redirect_stdout(io.StringIO()):
        tree = DataLoader()._load_hierarchical_deviation(path)
    brain = tree.root.children[0]
    assert (brain.node_name, brain.score_kind, brain.z_score) == ("BRAIN", "mean_abs", 1.386)
    leaf = brain.children[0].children[0]
    assert (leaf.node_name, leaf.score_kind) == ("left_hippocampus_volume", "signed")
    assert render_deviation_profile(tree).splitlines() == [
        "BRAIN: mean |z| 1.39",
        "BRAIN > Morphologics: mean |z| 0.90",
        "BRAIN > Morphologics > left_hippocampus_volume: -1.20",
        "COGNITION > memory: +0.40",
    ]
    # The hippocampus leaf is also a measurement leaf under "BRAIN > Morphologics".
    assert render_deviation_profile(tree, leaf_groups={"BRAIN > Morphologics"}).splitlines() == [
        "BRAIN: mean |z| 1.39",
        "BRAIN > Morphologics: mean |z| 0.90",
        "COGNITION > memory: +0.40",
    ]


def test_build_sections_prefers_the_parsed_deviation_tree():
    output = _synthetic_output()
    output["deviation_tree"] = {
        "root": {"node_name": "ROOT", "children": [{"node_name": "COGNITION", "z_score": 1.25, "score_kind": "mean_abs", "children": []}]}
    }
    sections, _total = build_sections(executor_output=output, task_context="ctx", count=_chars)
    deviation = next(s for s in sections if s.key == "deviation_profile")
    assert deviation.variants[0] == "COGNITION: mean |z| 1.25"
    # Without it, the serialized map in the Predictor input is used.
    del output["deviation_tree"]
    sections, _total = build_sections(executor_output=output, task_context="ctx", count=_chars)
    deviation = next(s for s in sections if s.key == "deviation_profile")
    assert deviation.variants[0].splitlines()[0] == "COGNITION: +1.20"


# ---------------------------------------------------------------------------
# State: packing (a character counter keeps the arithmetic exact)
# ---------------------------------------------------------------------------
def _chars(text: str) -> int:
    return len(text or "")


def _feature(name, z, value=None):
    return {
        "feature_id": name,
        "field_name": name,
        "value": z if value is None else value,
        "z_score": z,
        "domain": "COGNITION",
        "path_in_hierarchy": ["battery"],
    }


Z_VALUES = [0.1, 0.3, 0.6, 0.8, 1.2, 1.4, 1.6, 2.2, -2.5]
EXPECTED_COUNTS = [10, 8, 6, 4, 3]  # all, |z| >= 0.5, 1, 1.5, 2 (the categorical leaf always stays)


def _synthetic_output(*, notes="Participant reports low mood and poor sleep. " * 40, extra_strong=0):
    features = [_feature(f"test_{i}", z) for i, z in enumerate(Z_VALUES)]
    features.append({"feature_id": "smoking", "field_name": "Smoking status", "value": "former", "z_score": None, "domain": "COGNITION", "path_in_hierarchy": ["lifestyle"]})
    features += [_feature(f"strong_marker_{i:03d}_with_a_long_descriptive_label", 2.5 + i / 100.0) for i in range(extra_strong)]
    deviation = {
        "root": {
            "node_name": "root",
            "children": [
                {
                    "node_name": "COGNITION",
                    "z_score": 1.2,
                    "children": [
                        {
                            "node_name": "battery",
                            "z_score": 1.4,
                            "children": [{"node_name": "subtest", "z_score": 1.9, "children": [{"node_name": "x", "z_score": 2.0}]}],
                        }
                    ],
                }
            ],
        }
    }
    return {
        "predictor_input": {
            "non_numerical_data_raw": notes,
            "multimodal_unprocessed_raw": {"COGNITION": features},
            "hierarchical_deviation_raw": deviation,
        },
        "step_outputs": {
            "1": {"tool_name": "PhenotypeRepresentation", "summary": "Depressive phenotype with cognitive slowing. " * 30},
            "2": {"tool_name": "FeatureSynthesizer", "findings": ["memory deficit", "sleep disruption"] * 15},
        },
        "chunk_evidence": [
            {
                "chunk_index": i,
                "chunk_total": 3,
                "summary": "Chunk summary with several findings. " * 20,
                "for_case": ["low mood", "anhedonia", "insomnia", "fatigue", "guilt"],
                "for_control": ["intact attention"],
                "key_findings": [{"domain": "COGNITION", "finding": f"finding {j}", "direction": "ABNORMAL_LOW", "z_score": -1.5, "extra": "x" * 50} for j in range(6)],
            }
            for i in (1, 2, 3)
        ],
    }


def _sections(output=None, context="Study context: adults with depressive symptoms. " * 5):
    return build_sections(executor_output=output or _synthetic_output(), task_context=context, count=_chars)


def _size(sections, levels, dropped=()):
    state = {
        s.key: s.variants[levels.get(s.key, 0)]
        for s in sections
        if s.key not in dropped and s.variants[levels.get(s.key, 0)].strip()
    }
    return len(json.dumps(state, ensure_ascii=False))


def _rendering(packed: PackedState):
    return {row["section"]: row["rendering"] for row in packed.sections}


def test_synthetic_sections_cover_every_kind():
    sections, total = _sections()
    assert [s.key for s in sections] == [
        "task_context",
        "clinical_record",
        "phenotype_synthesis",
        "chunk_evidence",
        "measurements",
        "deviation_profile",
    ]
    assert total == 10
    measurements = next(s for s in sections if s.key == "measurements")
    assert measurements.feature_counts == EXPECTED_COUNTS
    deviation = next(s for s in sections if s.key == "deviation_profile")
    assert len(deviation.variants[1]) < len(deviation.variants[0])


def test_pack_state_keeps_everything_under_budget():
    sections, total = _sections()
    packed = pack_state(sections, budget_tokens=10**7, count=_chars, features_total=total)
    assert set(packed.state) == {s.key for s in sections}
    assert all(packed.state[s.key] == s.variants[0] for s in sections)
    assert packed.truncated is False
    assert packed.features_included == packed.features_total == 10
    assert packed.feature_coverage == 1.0
    assert packed.tokens == len(json.dumps(packed.state, ensure_ascii=False))
    assert set(_rendering(packed).values()) <= {"full", "all leaves", "all groups"}
    report = packed.report()
    assert report["feature_coverage"] == 1.0
    assert report["budget_tokens"] == 10**7
    assert report["features_total"] == 10
    measurement_row = next(r for r in report["sections"] if r["section"] == "measurements")
    assert measurement_row["features"] == 10


def test_pack_state_compacts_the_least_important_section_first():
    sections, total = _sections()
    budget = _size(sections, {}) - 1
    packed = pack_state(sections, budget_tokens=budget, count=_chars, features_total=total)
    rendering = _rendering(packed)
    assert rendering["deviation_profile"] == "top two levels"
    assert rendering["measurements"] == "all leaves"
    assert rendering["chunk_evidence"] == "full"
    assert packed.tokens <= budget
    assert packed.feature_coverage == 1.0


def test_packing_ladder_is_the_documented_order():
    assert PACKING_LADDER == (
        ("deviation_profile", "next"),
        ("phenotype_synthesis", "next"),
        ("chunk_evidence", "next"),
        ("deviation_profile", "drop"),
        ("task_context", "next"),
        ("measurements", "next"),
        ("measurements", "next"),
        ("measurements", "next"),
        ("measurements", "next"),
        ("chunk_evidence", "next"),
        ("phenotype_synthesis", "next"),
        ("clinical_record", "next"),
        ("clinical_record", "next"),
        ("chunk_evidence", "drop"),
        ("phenotype_synthesis", "drop"),
    )


PHENOTYPE_QUARTER = "first quarter (at most 3000 tokens)"


def test_synthetic_section_renderings():
    sections, _total = _sections()
    by_key = {s.key: s for s in sections}
    sizes = {s.key: [len(v) for v in s.variants] for s in sections}
    # 1799 characters of notes: halves and quarters are cut at max(400, 899) and max(300, 449).
    assert by_key["clinical_record"].variant_labels == ["full", "first half", "first quarter"]
    assert sizes["clinical_record"] == [1799, 898, 448]
    record = by_key["clinical_record"]
    assert record.variants[1].endswith(TRUNCATION_MARK) and record.variants[2].endswith(TRUNCATION_MARK)
    # 2013 characters of tool syntheses: first half at max(1500, 1006), first
    # quarter at min(3000, max(750, 503)).
    assert by_key["phenotype_synthesis"].variant_labels == ["full", "first half", PHENOTYPE_QUARTER]
    assert sizes["phenotype_synthesis"] == [2013, 1500, 750]
    assert by_key["chunk_evidence"].variant_labels == ["full", "compact rows", "compact rows, first half"]
    assert by_key["deviation_profile"].variant_labels == ["all groups", "top two levels"]
    # A 239 character study context is not shortened by a 1500 token cut, so
    # that rendering would not shrink and is left out.
    assert by_key["task_context"].variant_labels == ["full"]
    assert sizes["task_context"] == [239]


def test_every_kept_rendering_is_strictly_smaller_than_the_one_before():
    outputs = [
        _synthetic_output(),
        _synthetic_output(notes="Short note.", extra_strong=80),
        _synthetic_output(notes="Visit note: low mood, poor sleep. " * 400),
    ]
    for output in outputs:
        sections, _total = _sections(output)
        for s in sections:
            sizes = [len(v) for v in s.variants]
            assert all(a > b for a, b in zip(sizes, sizes[1:])), s.key
            assert len(s.variant_labels) == len(s.variants)
            if s.feature_counts:
                assert len(s.feature_counts) == len(s.variants)


def test_keep_shrinking_filters_labels_and_feature_counts_alongside():
    section = StateSection(
        key="measurements",
        priority=4,
        variants=["aaaa", "aaaa", "aa", "aaa", "a"],
        variant_labels=["l0", "l1", "l2", "l3", "l4"],
        feature_counts=[5, 4, 3, 2, 1],
    )
    state_module._keep_shrinking(section, _chars)
    # "aaaa" again does not shrink, nor does "aaa" after "aa".
    assert section.variants == ["aaaa", "aa", "a"]
    assert section.variant_labels == ["l0", "l2", "l4"]
    assert section.feature_counts == [5, 3, 1]


def test_a_short_record_keeps_only_its_full_rendering_and_is_never_stepped():
    output = _synthetic_output(notes="Short note.", extra_strong=80)
    sections, total = build_sections(executor_output=output, task_context="ctx", count=_chars)
    by_key = {s.key: s for s in sections}
    assert by_key["clinical_record"].variant_labels == ["full"]
    assert by_key["task_context"].variant_labels == ["full"]
    # The packer walks the whole ladder; the record stays "full" until cut.
    most_compact = {s.key: len(s.variants) - 1 for s in sections}
    core_only = _size(sections, most_compact, dropped=("deviation_profile", "chunk_evidence", "phenotype_synthesis"))
    packed = pack_state(sections, budget_tokens=core_only, count=_chars, features_total=total)
    assert _rendering(packed)["clinical_record"] == "full"
    assert packed.state["clinical_record"] == "Short note."
    assert packed.truncated is False


# Each case: the ladder step the packer must stop at, the rendering levels and
# dropped sections at that step, and the expected rendering labels. Step 5
# (task_context next) has no smaller rendering to step to on this record.
LADDER_CASES = [
    (
        1,
        {"deviation_profile": 1},
        (),
        {"deviation_profile": "top two levels", "phenotype_synthesis": "full", "chunk_evidence": "full", "measurements": "all leaves"},
    ),
    (
        2,
        {"deviation_profile": 1, "phenotype_synthesis": 1},
        (),
        {"deviation_profile": "top two levels", "phenotype_synthesis": "first half", "chunk_evidence": "full"},
    ),
    (
        3,
        {"deviation_profile": 1, "phenotype_synthesis": 1, "chunk_evidence": 1},
        (),
        {"chunk_evidence": "compact rows", "task_context": "full", "measurements": "all leaves"},
    ),
    (
        4,
        {"phenotype_synthesis": 1, "chunk_evidence": 1},
        ("deviation_profile",),
        {"deviation_profile": "dropped", "task_context": "full", "measurements": "all leaves"},
    ),
    (
        10,
        {"phenotype_synthesis": 1, "chunk_evidence": 2, "measurements": 4},
        ("deviation_profile",),
        {
            "chunk_evidence": "compact rows, first half",
            "phenotype_synthesis": "first half",
            "measurements": "|z| >= 2 and categorical",
            "clinical_record": "full",
            "task_context": "full",
        },
    ),
    (
        11,
        {"phenotype_synthesis": 2, "chunk_evidence": 2, "measurements": 4},
        ("deviation_profile",),
        {"phenotype_synthesis": PHENOTYPE_QUARTER, "clinical_record": "full", "chunk_evidence": "compact rows, first half"},
    ),
    (
        12,
        {"phenotype_synthesis": 2, "chunk_evidence": 2, "measurements": 4, "clinical_record": 1},
        ("deviation_profile",),
        {"phenotype_synthesis": PHENOTYPE_QUARTER, "clinical_record": "first half", "chunk_evidence": "compact rows, first half"},
    ),
    (
        13,
        {"phenotype_synthesis": 2, "chunk_evidence": 2, "measurements": 4, "clinical_record": 2},
        ("deviation_profile",),
        {"clinical_record": "first quarter", "chunk_evidence": "compact rows, first half"},
    ),
]


@pytest.mark.parametrize("step, levels, dropped, expected", LADDER_CASES, ids=[f"step{c[0]}" for c in LADDER_CASES])
def test_pack_state_walks_the_ladder_and_stops_once_it_fits(step, levels, dropped, expected):
    sections, total = _sections()
    budget = _size(sections, levels, dropped)
    packed = pack_state(sections, budget_tokens=budget, count=_chars, features_total=total)
    rendering = _rendering(packed)
    for key, label in expected.items():
        assert rendering[key] == label, key
    assert set(packed.state) == {s.key for s in sections} - set(dropped)
    assert packed.tokens == budget
    assert packed.truncated is False


@pytest.mark.parametrize("step, label", [(1, "|z| >= 0.5 and categorical"), (2, "|z| >= 1 and categorical"), (3, "|z| >= 1.5 and categorical"), (4, "|z| >= 2 and categorical")])
def test_pack_state_steps_measurements_down_by_deviation(step, label):
    sections, total = _sections()
    # Before the measurements, the ladder compacts the group aggregates (then
    # drops them), the tool syntheses and the chunk rows (the study context has
    # no smaller rendering here).
    budget = _size(
        sections,
        {"phenotype_synthesis": 1, "chunk_evidence": 1, "measurements": step},
        dropped=("deviation_profile",),
    )
    packed = pack_state(sections, budget_tokens=budget, count=_chars, features_total=total)
    rendering = _rendering(packed)
    assert rendering["measurements"] == label
    assert rendering["deviation_profile"] == "dropped"
    assert rendering["phenotype_synthesis"] == "first half"
    assert rendering["chunk_evidence"] == "compact rows"
    assert rendering["task_context"] == "full"
    # The clinical record is compacted only after the measurements.
    assert rendering["clinical_record"] == "full"
    assert packed.truncated is False
    assert packed.tokens == budget
    assert packed.features_included == EXPECTED_COUNTS[step]
    assert packed.feature_coverage == pytest.approx(EXPECTED_COUNTS[step] / 10.0)
    assert packed.report()["feature_coverage"] == round(EXPECTED_COUNTS[step] / 10.0, 4)
    row = next(r for r in packed.sections if r["section"] == "measurements")
    assert row["features"] == EXPECTED_COUNTS[step]
    kept_lines = [line for line in packed.state["measurements"].splitlines() if line.startswith("- ")]
    assert len(kept_lines) == EXPECTED_COUNTS[step]


def test_pack_state_drops_optional_sections_before_the_core_ones():
    sections, total = _sections()
    most_compact = {s.key: len(s.variants) - 1 for s in sections}
    # Ladder step 14: chunk evidence dropped, tool syntheses still in.
    budget = _size(sections, most_compact, dropped=("deviation_profile", "chunk_evidence"))
    packed = pack_state(sections, budget_tokens=budget, count=_chars, features_total=total)
    rendering = _rendering(packed)
    assert rendering["deviation_profile"] == "dropped"
    assert rendering["chunk_evidence"] == "dropped"
    assert rendering["phenotype_synthesis"] == PHENOTYPE_QUARTER
    assert rendering["clinical_record"] == "first quarter"
    assert rendering["task_context"] == "full"
    assert rendering["measurements"] == "|z| >= 2 and categorical"
    assert set(packed.state) == {"task_context", "clinical_record", "phenotype_synthesis", "measurements"}
    assert packed.truncated is False
    assert packed.tokens == budget
    assert packed.features_included == EXPECTED_COUNTS[-1]

    # Ladder step 15: only the study context, the record and the measurements remain.
    core_only = _size(sections, most_compact, dropped=("deviation_profile", "chunk_evidence", "phenotype_synthesis"))
    packed = pack_state(sections, budget_tokens=core_only, count=_chars, features_total=total)
    assert set(packed.state) == {"task_context", "clinical_record", "measurements"}
    assert _rendering(packed)["phenotype_synthesis"] == "dropped"
    assert packed.truncated is False
    assert packed.tokens == core_only


def _ladder_sizes(sections):
    """State size after each ladder step that changed something."""
    by_key = {s.key: s for s in sections}
    levels = {s.key: 0 for s in sections}
    dropped = set()
    sizes = [_size(sections, levels)]
    for key, action in PACKING_LADDER:
        if key not in by_key or key in dropped:
            continue
        if action == "drop":
            dropped.add(key)
        elif levels[key] < len(by_key[key].variants) - 1:
            levels[key] += 1
        else:
            continue
        sizes.append(_size(sections, levels, tuple(dropped)))
    return sizes


def test_no_ladder_step_grows_the_state():
    sections, _total = _sections()
    sizes = _ladder_sizes(sections)
    # Every step that changes the state shrinks it. On this record: 10299,
    # then step by step down to 864 characters of study context, record and
    # measurements.
    assert all(a > b for a, b in zip(sizes, sizes[1:]))
    assert sizes[0] == 10299 and sizes[-1] == 864
    # Concretely: a budget of 4394 characters is met once the record is halved
    # (step 12, 3642 characters), with the chunk evidence kept.
    budget = _size(
        sections,
        {"phenotype_synthesis": 1, "chunk_evidence": 2, "measurements": 4, "clinical_record": 1},
        dropped=("deviation_profile",),
    )
    assert budget == 4394
    packed = pack_state(sections, budget_tokens=budget, count=_chars, features_total=10)
    assert "chunk_evidence" in packed.state
    assert packed.tokens == 3642
    rendering = _rendering(packed)
    assert rendering["phenotype_synthesis"] == PHENOTYPE_QUARTER
    assert rendering["clinical_record"] == "first half"


def test_pack_state_truncates_the_largest_core_section_as_a_last_resort():
    notes = "Visit note: low mood, poor sleep, reduced appetite and concentration. " * 120
    sections, total = _sections(_synthetic_output(notes=notes))
    most_compact = {s.key: len(s.variants) - 1 for s in sections}
    core_only = _size(sections, most_compact, dropped=("deviation_profile", "chunk_evidence", "phenotype_synthesis"))
    budget = core_only - 1000
    packed = pack_state(sections, budget_tokens=budget, count=_chars, features_total=total)
    assert packed.truncated is True
    assert set(packed.state) == {"task_context", "clinical_record", "measurements"}
    record = next(s for s in sections if s.key == "clinical_record")
    # Cut below even its most compact rendering, and marked as cut.
    assert len(packed.state["clinical_record"]) < len(record.variants[-1])
    assert packed.state["clinical_record"].endswith(TRUNCATION_MARK)
    assert packed.state["measurements"] == next(s for s in sections if s.key == "measurements").variants[-1]
    assert packed.tokens <= budget
    # Measurements were not cut, so the coverage is that of their rendering.
    assert packed.features_included == EXPECTED_COUNTS[-1]
    assert packed.report()["truncated"] is True
    rendering = _rendering(packed)
    assert rendering["clinical_record"] == "first quarter, truncated"
    assert rendering["measurements"] == "|z| >= 2 and categorical"
    assert rendering["task_context"] == "full"
    # The truncated record is cut, as a section, to what the budget leaves it.
    record_row = next(r for r in packed.sections if r["section"] == "clinical_record")
    assert record_row["tokens"] == len(packed.state["clinical_record"])


def test_truncated_measurements_report_the_leaves_that_survived():
    output = _synthetic_output(notes="Short note.", extra_strong=80)
    sections, total = build_sections(executor_output=output, task_context="ctx", count=_chars)
    assert total == 90
    most_compact = {s.key: len(s.variants) - 1 for s in sections}
    core_only = _size(sections, most_compact, dropped=("deviation_profile", "chunk_evidence", "phenotype_synthesis"))
    budget = core_only // 2
    packed = pack_state(sections, budget_tokens=budget, count=_chars, features_total=total)
    assert packed.truncated is True
    assert packed.state["measurements"].endswith(TRUNCATION_MARK)
    survivors = sum(1 for line in packed.state["measurements"].splitlines() if line.startswith("- "))
    assert 0 < survivors < 83
    assert packed.features_included == survivors
    assert packed.feature_coverage == pytest.approx(survivors / 90.0)


def test_truncated_measurements_section_row_matches_the_survivors():
    output = _synthetic_output(notes="Short note.", extra_strong=80)
    sections, total = build_sections(executor_output=output, task_context="ctx", count=_chars)
    most_compact = {s.key: len(s.variants) - 1 for s in sections}
    core_only = _size(sections, most_compact, dropped=("deviation_profile", "chunk_evidence", "phenotype_synthesis"))
    packed = pack_state(sections, budget_tokens=core_only // 2, count=_chars, features_total=total)
    assert packed.truncated is True
    row = next(r for r in packed.sections if r["section"] == "measurements")
    survivors = sum(1 for line in packed.state["measurements"].splitlines() if line.startswith("- "))
    assert row["features"] == packed.features_included == survivors
    assert row["rendering"] == "|z| >= 2 and categorical, truncated"
    assert packed.tokens <= core_only // 2


def test_last_resort_truncation_repeats_until_the_state_fits():
    # A budget far below the most compact core: the largest section is cut, then
    # the next largest, until the whole state fits.
    output = _synthetic_output(notes="Visit note: low mood and poor sleep. " * 60, extra_strong=40)
    sections, total = build_sections(executor_output=output, task_context="Study context. " * 40, count=_chars)
    most_compact = {s.key: len(s.variants) - 1 for s in sections}
    core_only = _size(sections, most_compact, dropped=("deviation_profile", "chunk_evidence", "phenotype_synthesis"))
    budget = core_only // 3
    packed = pack_state(sections, budget_tokens=budget, count=_chars, features_total=total)
    assert packed.truncated is True
    assert packed.tokens <= budget
    assert packed.tokens == len(json.dumps(packed.state, ensure_ascii=False))
    cut = [row["section"] for row in packed.sections if row["rendering"].endswith(", truncated")]
    assert len(cut) >= 2
    for key in cut:
        assert packed.state[key].endswith(TRUNCATION_MARK)


def test_pack_state_with_real_tokens_on_a_pseudo_participant(direct_output):
    sections, total = build_sections(executor_output=direct_output, task_context="Adult MDD cohort.", count=_cl100k)
    full = pack_state(sections, budget_tokens=26_000, count=_cl100k, features_total=total)
    assert full.feature_coverage == 1.0 and not full.truncated
    tight = pack_state(sections, budget_tokens=full.tokens // 2, count=_cl100k, features_total=total)
    assert tight.tokens <= full.tokens // 2
    assert tight.features_included < total
    assert "clinical_record" in tight.state and "measurements" in tight.state


def test_feature_coverage_without_features_is_complete():
    packed = PackedState(state={}, tokens=1, budget=10, sections=[], features_total=0, features_included=0, truncated=False)
    assert packed.feature_coverage == 1.0
