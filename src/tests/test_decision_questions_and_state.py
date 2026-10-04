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
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.config.settings import reload_settings
from src.full_stack.backend.data.models.prediction_task import (
    PredictionMode,
    PredictionTaskNode,
    PredictionTaskSpec,
)
from src.full_stack.backend.decision.questions import (
    COMPILER_SYSTEM_PROMPT,
    QuestionBook,
    QuestionSet,
    add_fine_regression_questions,
    asked_summary,
    build_question_book,
    cache_dir,
    deterministic_book,
    round_one_questions,
    task_hash,
)
from src.full_stack.backend.decision.state import (
    PackedState,
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
def test_binary_node_gets_one_noul_and_the_choice_in_both_orders():
    spec = _binary_spec(class_definitions={"MDD": "Meets DSM-5 criteria.", "CONTROL": "No diagnosis."})
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)

    nouls = _of_kind(qs, "binary_noul", "dx")
    choices = _of_kind(qs, "choice", "dx")
    assert len(nouls) == 1
    noul = nouls[0]
    assert noul.payload["type"] == "noul"
    assert noul.order == ["MDD", "CONTROL"]
    assert set(noul.payload["criteria"]) == {"true", "false"}
    assert noul.payload["criteria"]["true"] == "MDD: Meets DSM-5 criteria."
    assert noul.payload["criteria"]["false"] == "CONTROL: No diagnosis."

    assert len(choices) == 2
    assert [q.order for q in choices] == [["MDD", "CONTROL"], ["CONTROL", "MDD"]]
    for q in choices:
        assert q.payload["type"] == "choice"
        # The criteria map is presented in the question's order.
        assert list(q.payload["criteria"]) == q.order
        assert q.payload["criteria"]["MDD"] == "Meets DSM-5 criteria."
    assert len(qs) == 1 + 2 + 1  # noul, two choices, evidence sufficiency


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
    assert asc.payload["criteria"][0].startswith("0 to 6 points")
    assert desc.payload["criteria"][0].startswith("54 to 60 points")
    assert "ranges from 0 to 60 points" in asc.payload["instructions"]
    assert [b.center for b in asc.bins] == [b.center for b in desc.bins]


def test_exact_integer_output_gets_one_level_per_value():
    spec = _regression_spec({"min": 0, "max": 6, "integer": True})
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    asc, desc = _of_kind(qs, "score_coarse")
    assert len(asc.payload["criteria"]) == 7
    assert all(b.exact for b in asc.bins)
    assert desc.payload["criteria"] == list(reversed(asc.payload["criteria"]))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: decision/questions.py _scale_instruction formats the range with width=span/10, so an integer "
        "0 to 6 scale is announced as 'ranges from 0.00 to 6.00' while its levels read 0, 1, ..., 6"
    ),
)
def test_integer_scale_instruction_announces_integer_bounds():
    spec = _regression_spec({"min": 0, "max": 6, "integer": True})
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    asc = _of_kind(qs, "score_coarse")[0]
    assert "ranges from 0 to 6" in asc.payload["instructions"]


def test_regression_without_a_scale_uses_the_default_standardized_scale():
    spec = _regression_spec()
    book = deterministic_book(spec)
    assert book.nodes["madrs"].scales["madrs_total"].source == "default"
    qs = round_one_questions(spec, book, choice_orders=3, score_levels=10)
    asc = _of_kind(qs, "score_coarse")[0]
    assert len(asc.payload["criteria"]) == 10
    assert "standardized units" in asc.payload["instructions"]


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
    assert len(_of_kind(qs, "binary_noul", "response")) == 1
    assert len(_of_kind(qs, "choice", "response")) == 2
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
    round_two = QuestionSet(prefix="r")
    fine = scale.fine_grid(12.0, 30.0, 10)
    add_fine_regression_questions(round_two, scores_node, "madrs_total", scale, fine, 12.0, 30.0)
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
    assert "most likely lies between 12 and 30 points" in fine_qs[0].payload["instructions"]
    assert len(fine_qs[0].payload["criteria"]) == 10


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
                "question": "",
                "label_definitions": {"responder": "At least 50 percent MADRS reduction."},
            },
        }
    }


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
    # An empty compiled question keeps the default wording.
    assert response.question.startswith("Which group does this participant belong to")
    assert response.label_definitions["responder"] == "At least 50 percent MADRS reduction."
    assert response.label_definitions["non_responder"] == "The participant belongs to the 'non_responder' group."


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


def test_compiler_failure_falls_back_to_the_deterministic_book():
    spec = _hierarchical_spec()
    book = build_question_book(spec, context="ctx", llm_json=_CompilerStub(error=RuntimeError("provider down")))
    assert book.compiled_by == "deterministic"
    assert any("Compilation failed (RuntimeError)" in note for note in book.notes)
    assert book.nodes["severity"].label_definitions["mild"] == "The participant belongs to the 'mild' group."
    assert book.nodes["scores"].scales["madrs_total"].source == "task_spec"
    assert _book_files() == []


def test_compiler_returning_garbage_keeps_defaults():
    spec = _hierarchical_spec()
    book = build_question_book(spec, context="ctx", llm_json=_CompilerStub({"unexpected": True}), compiler_label="stub")
    assert book.compiled_by == "stub"
    assert book.nodes["scores"].scales["hamd_total"].source == "default"
    assert any("No usable compiled scale for 'scores.hamd_total'" in note for note in book.notes)


def test_unusable_compiled_scale_keeps_the_default_scale():
    spec = _hierarchical_spec()
    payload = _compiled_payload()
    payload["nodes"]["scores"]["scales"]["hamd_total"] = {"min": 10, "max": 10}
    book = build_question_book(spec, context="ctx", llm_json=_CompilerStub(payload), compiler_label="stub")
    assert book.nodes["scores"].scales["hamd_total"].source == "default"
    assert any("No usable compiled scale for 'scores.hamd_total'" in note for note in book.notes)


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
    assert by_key["clinical_record"].variants[0] == str(direct_output["predictor_input"]["non_numerical_data_raw"]).strip()

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


def test_measurement_rows_keep_value_unit_and_z():
    multimodal = {
        "COGNITION": [
            {"feature_id": "ravlt", "field_name": "RAVLT total", "value": 38, "unit": "words", "z_score": -1.2, "domain": "COGNITION", "path_in_hierarchy": ["memory"]},
            {"feature_id": "tmt", "field_name": "TMT-B", "value": 1.5, "z_score": 1.5, "domain": "COGNITION", "path_in_hierarchy": ["executive"]},
            {"feature_id": "smoke", "field_name": "Smoking status", "value": "former", "z_score": None, "domain": "COGNITION", "path_in_hierarchy": []},
            {"feature_id": "cer", "field_name": "Cer(d18:1/16:0) High", "value": "High", "z_score": 2.4, "domain": "COGNITION", "path_in_hierarchy": ["lipids"]},
        ]
    }
    rows = measurement_rows(multimodal)
    assert rows[0] == ("COGNITION > memory", "RAVLT total = 38 words (z -1.20)", -1.2)
    # The loader falls back to the z-score as value; it is not printed twice.
    assert rows[1] == ("COGNITION > executive", "TMT-B (z +1.50)", 1.5)
    assert rows[2] == ("COGNITION", "Smoking status = former", None)
    # A value the label already spells out is not repeated.
    assert rows[3] == ("COGNITION > lipids", "Cer(d18:1/16:0) High (z +2.40)", 2.4)

    text, kept = render_measurements(rows, min_abs_z=1.5)
    assert kept == 3  # |z| 1.5 and 2.4, plus the categorical leaf
    assert "RAVLT" not in text and "Smoking status = former" in text


def test_render_deviation_profile_lists_internal_groups_only():
    deviation = {
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
                                {"node_name": "verbal", "z_score": -1.6, "children": [{"node_name": "leaf", "z_score": -2.0}]}
                            ],
                        }
                    ],
                }
            ],
        }
    }
    full = render_deviation_profile(deviation)
    assert full.splitlines() == [
        "COGNITION: +1.10",
        "COGNITION > memory: -1.30",
        "COGNITION > memory > verbal: -1.60",
    ]
    assert render_deviation_profile(deviation, max_depth=2).splitlines() == full.splitlines()[:2]
    assert render_deviation_profile({}) == ""


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


@pytest.mark.parametrize("step, label", [(1, "|z| >= 0.5 and categorical"), (2, "|z| >= 1 and categorical"), (3, "|z| >= 1.5 and categorical"), (4, "|z| >= 2 and categorical")])
def test_pack_state_steps_measurements_down_by_deviation(step, label):
    sections, total = _sections()
    budget = _size(sections, {"deviation_profile": 1, "measurements": step})
    packed = pack_state(sections, budget_tokens=budget, count=_chars, features_total=total)
    rendering = _rendering(packed)
    assert rendering["measurements"] == label
    # Measurements are compacted before the tool syntheses and chunk evidence.
    assert rendering["chunk_evidence"] == "full"
    assert rendering["phenotype_synthesis"] == "full"
    assert rendering["clinical_record"] == "full"
    assert rendering["task_context"] == "full"
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
    # Room for the core sections and the tool syntheses, at their most compact.
    budget = _size(sections, most_compact, dropped=("deviation_profile", "chunk_evidence"))
    packed = pack_state(sections, budget_tokens=budget, count=_chars, features_total=total)
    rendering = _rendering(packed)
    assert rendering["deviation_profile"] == "dropped"
    assert rendering["chunk_evidence"] == "dropped"
    assert rendering["phenotype_synthesis"] == "first 3000 tokens"
    assert rendering["clinical_record"] == "first half"
    assert rendering["task_context"] == "first 1500 tokens"
    assert rendering["measurements"] == "|z| >= 2 and categorical"
    assert set(packed.state) == {"task_context", "clinical_record", "phenotype_synthesis", "measurements"}
    assert packed.truncated is False
    assert packed.features_included == EXPECTED_COUNTS[-1]

    core_only = _size(sections, most_compact, dropped=("deviation_profile", "chunk_evidence", "phenotype_synthesis"))
    packed = pack_state(sections, budget_tokens=core_only, count=_chars, features_total=total)
    assert set(packed.state) == {"task_context", "clinical_record", "measurements"}
    assert _rendering(packed)["phenotype_synthesis"] == "dropped"
    assert packed.truncated is False


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: decision/state.py pack_state recounts features_included after the last-resort truncation "
        "but leaves the measurements section row at the pre-truncation count (row['features'] 83 vs 40 kept)"
    ),
)
def test_truncated_measurements_section_row_matches_the_survivors():
    output = _synthetic_output(notes="Short note.", extra_strong=80)
    sections, total = build_sections(executor_output=output, task_context="ctx", count=_chars)
    most_compact = {s.key: len(s.variants) - 1 for s in sections}
    core_only = _size(sections, most_compact, dropped=("deviation_profile", "chunk_evidence", "phenotype_synthesis"))
    packed = pack_state(sections, budget_tokens=core_only // 2, count=_chars, features_total=total)
    assert packed.truncated is True
    row = next(r for r in packed.sections if r["section"] == "measurements")
    assert row["features"] == packed.features_included


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
