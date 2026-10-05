"""
Structured decision models, offline end to end.

- aggregation of answers into node predictions (order ensembles, regression
  densities, stability);
- the DecisionPredictor against a fake decision model that holds fixed beliefs
  and answers them consistently in every presentation order;
- the decision critic;
- the HTTP transport against a fake session.

No socket may open; the decision model, the companion LLM and the HTTP session
are all fakes.
"""

import contextlib
import copy
import io
import json
import math
import re
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.agents.critic import Critic
from src.full_stack.backend.config.settings import reload_settings
from src.full_stack.backend.data.models.prediction_result import (
    BinaryClassification,
    ClassificationPrediction,
    ConfidenceLevel,
    NodePrediction,
    PredictionResult,
    RegressionPrediction,
    Verdict,
)
from src.full_stack.backend.data.models.prediction_task import (
    PredictionMode,
    PredictionTaskNode,
    PredictionTaskSpec,
)
from src.full_stack.backend.agents.decision.aggregate import (
    aggregate_classification,
    answer_usable,
    noul_value,
    aggregate_regression_output,
    build_node_prediction,
    coarse_ensemble,
    confidence_level,
    refinement_window,
    score_distribution,
)
from src.full_stack.backend.agents.decision.client import (
    DecisionClient,
    DecisionRequestError,
    DecisionResponse,
    DecisionUsage,
    _openrouter_decisions_url,
)
from src.full_stack.backend.agents.decision.predictor import DecisionPredictor
from src.full_stack.backend.agents.decision.quality import evaluate_decision_prediction, route_can_help
from src.full_stack.backend.agents.decision.questions import (
    QuestionSet,
    add_classification_questions,
    add_coarse_regression_questions,
    add_fine_regression_questions,
    cache_dir,
    deterministic_book,
    round_one_questions,
    task_hash,
)
from src.full_stack.backend.agents.decision.scales import OutputScale, combine_density, density_summary
from src.full_stack.backend.utils.core.data_loader import DataLoader
from src.full_stack.backend.utils.core.input_routing import build_direct_executor_output
from src.full_stack.backend.utils.token_packer import count_tokens

REPO = Path(__file__).resolve().parents[2]
SUBJ_001 = REPO / "src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO"
JEV = "typesafe/jev-1.13"
CONTEXT = "Adult outpatients with depressive symptoms; MADRS rated by a clinician at baseline."


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    def _blocked(*_args, **_kwargs):
        raise RuntimeError("network access is not allowed in these tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(requests.Session, "request", _blocked)
    monkeypatch.setattr(DecisionClient, "_sleep", staticmethod(lambda attempt, retry_after: None))
    monkeypatch.setenv("COMPASS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path / "compass_home"))
    monkeypatch.delenv("COMPASS_DECISION_ENDPOINT", raising=False)


@pytest.fixture
def settings():
    s = reload_settings()
    s.models.predictor_model = JEV
    # Dummy keys: the real ones must never reach a fake session either.
    s.openrouter_api_key = "test-openrouter-key"
    s.typesafe_api_key = "test-typesafe-key"
    yield s
    reload_settings()


def _tokens(state) -> int:
    return count_tokens(json.dumps(state, ensure_ascii=False), model_hint="cl100k_base")


# ---------------------------------------------------------------------------
# Task specifications
# ---------------------------------------------------------------------------
def _mixed_spec() -> PredictionTaskSpec:
    return PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="severity",
            display_name="Depression severity",
            mode=PredictionMode.MULTICLASS_CLASSIFICATION,
            class_labels=["none", "mild", "severe"],
            children=[
                PredictionTaskNode(
                    node_id="madrs",
                    display_name="MADRS total",
                    mode=PredictionMode.UNIVARIATE_REGRESSION,
                    regression_outputs=["madrs_total"],
                    output_scales={"madrs_total": {"min": 0, "max": 60, "integer": True, "unit": "points"}},
                ),
                PredictionTaskNode(
                    node_id="items",
                    display_name="MADRS items",
                    mode=PredictionMode.MULTIVARIATE_REGRESSION,
                    regression_outputs=["item_sadness", "item_sleep"],
                    output_scales={
                        "item_sadness": {"min": 0, "max": 6, "integer": True},
                        "item_sleep": {"min": 0, "max": 6, "integer": True},
                    },
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


def _binary_spec() -> PredictionTaskSpec:
    return PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="dx",
            display_name="Major depressive disorder",
            mode=PredictionMode.BINARY_CLASSIFICATION,
            class_labels=["MDD", "CONTROL"],
        )
    )


@pytest.fixture(scope="module")
def direct_output():
    reload_settings()
    with contextlib.redirect_stdout(io.StringIO()):
        participant = DataLoader().load(SUBJ_001)
        out = build_direct_executor_output(
            participant,
            target_condition="Depression severity",
            control_condition="",
            prediction_task_spec=_mixed_spec(),
        )
    return out


# ---------------------------------------------------------------------------
# A fake decision model
# ---------------------------------------------------------------------------
_LEVEL = re.compile(r"^\s*(-?\d+(?:\.\d+)?)(?:\s+to\s+(-?\d+(?:\.\d+)?))?")
LABEL_WEIGHTS = {
    "none": 0.2,
    "mild": 0.5,
    "severe": 0.3,
    "responder": 0.65,
    "non_responder": 0.35,
    "MDD": 0.7,
    "CONTROL": 0.3,
}
TARGETS = {"madrs_total": 22.0, "item_sadness": 4.0, "item_sleep": 2.0}
SPREADS = {"madrs_total": 4.0, "item_sadness": 0.6, "item_sleep": 0.6}
UNASSESSED_NOTE = "Not every presentation order was answered, so stability could not be assessed."


class OracleClient:
    """
    Holds fixed beliefs and answers them the same way in every presentation
    order: Choice and Noul from label weights, Score from a normal curve around
    a target value read off each level's label.
    """

    def __init__(self, *, label_weights=None, targets=None, sufficiency=0.8, failures=None, cost_usd=0.0000504, input_tokens=1200):
        self.label_weights = dict(label_weights or LABEL_WEIGHTS)
        self.targets = dict(targets or TARGETS)
        self.sufficiency = sufficiency
        self.failures = list(failures or [])
        self.cost_usd = cost_usd
        self.input_tokens = input_tokens
        self.calls = []

    def ask(self, *, model, state, questions):
        self.calls.append({"model": model, "state": copy.deepcopy(state), "questions": copy.deepcopy(questions)})
        if self.failures:
            raise self.failures.pop(0)
        answers = {qid: self._answer(payload) for qid, payload in questions.items()}
        return DecisionResponse(
            model=model,
            answers=answers,
            usage=DecisionUsage(input_tokens=self.input_tokens, output_tokens=0, cost_usd=self.cost_usd),
            latency_ms=4,
            request_id=f"fake-{len(self.calls)}",
        )

    def _answer(self, payload):
        criteria = payload["criteria"]
        if payload["type"] == "noul":
            if payload["instructions"].startswith("Does the participant record contain enough"):
                return {"noul": self.sufficiency}
            first = criteria["true"].split(":", 1)[0]
            second = criteria["false"].split(":", 1)[0]
            w1, w2 = self.label_weights[first], self.label_weights[second]
            return {"noul": w1 / (w1 + w2)}
        if payload["type"] == "choice":
            weights = {label: self.label_weights[label] for label in criteria}
            total = sum(weights.values())
            probs = {label: w / total for label, w in weights.items()}
            return {"choice": max(probs, key=probs.get), "probabilities": probs, "confidence": 0.7}
        output = next(name for name in self.targets if f"({name})" in payload["instructions"])
        target, spread = self.targets[output], SPREADS.get(output, 1.0)
        weights = []
        for text in criteria:
            match = _LEVEL.match(text)
            lo = float(match.group(1))
            hi = float(match.group(2)) if match.group(2) else lo
            weights.append(math.exp(-0.5 * (((lo + hi) / 2.0 - target) / spread) ** 2))
        total = sum(weights) or 1.0
        probs = {str(i): w / total for i, w in enumerate(weights)}
        return {"score": max(range(len(weights)), key=weights.__getitem__), "probabilities": probs, "confidence": 0.6}


class RecordingTokenManager:
    def __init__(self):
        self.records = []

    def record_usage(self, **kwargs):
        self.records.append(kwargs)


def _predictor(oracle=None, *, token_manager=None, context=CONTEXT) -> DecisionPredictor:
    predictor = DecisionPredictor(llm_client=SimpleNamespace(), token_manager=token_manager)
    predictor.set_runtime_instruction(context)
    predictor.question_book = lambda task_spec: deterministic_book(task_spec, predictor.task_context())
    predictor.client = oracle or OracleClient()
    return predictor


def _run(predictor, executor_output, spec, *, target="Depression severity", control=""):
    with contextlib.redirect_stdout(io.StringIO()):
        return predictor.execute(dict(executor_output), target, control, prediction_task_spec=spec)


# ---------------------------------------------------------------------------
# Aggregation: classification
# ---------------------------------------------------------------------------
def _node_questions(node: PredictionTaskNode, *, choice_orders=3, levels=10) -> QuestionSet:
    book = deterministic_book(PredictionTaskSpec(root=node)).nodes[node.node_id]
    qs = QuestionSet()
    if node.mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
        add_classification_questions(qs, node, book, choice_orders=choice_orders)
    else:
        add_coarse_regression_questions(qs, node, book, levels=levels)
    return qs


def _multiclass_node():
    return PredictionTaskNode(
        node_id="severity",
        display_name="Severity",
        mode=PredictionMode.MULTICLASS_CLASSIFICATION,
        class_labels=["A", "B", "C"],
    )


def _binary_node(labels=("MDD", "CONTROL")):
    return PredictionTaskNode(
        node_id="dx",
        display_name="Diagnosis",
        mode=PredictionMode.BINARY_CLASSIFICATION,
        class_labels=list(labels),
    )


def test_classification_orders_are_averaged_and_the_largest_pairwise_tv_is_the_instability():
    node = _multiclass_node()
    qs = _node_questions(node)
    q1, q2, q3 = qs.items
    answers = {
        q1.qid: {"probabilities": {"A": 0.6, "B": 0.3, "C": 0.1}, "confidence": 0.8, "choice": "A"},
        q2.qid: {"probabilities": {"A": 0.4, "B": 0.4, "C": 0.2}, "confidence": 0.6},
        q3.qid: {"probabilities": {"A": 0.5, "B": 0.2, "C": 0.3}, "confidence": 0.7},
    }
    agg = aggregate_classification(node, qs.items, answers)
    assert agg.probabilities == pytest.approx({"A": 0.5, "B": 0.3, "C": 0.2})
    assert agg.predicted_label == "A"
    # Each order is 0.1 total variation from the mean, but every pair of orders
    # is 0.2 apart: TV(1,2) = (0.2 + 0.1 + 0.1) / 2, TV(1,3) = (0.1 + 0.1 + 0.2) / 2,
    # TV(2,3) = (0.1 + 0.2 + 0.1) / 2. The largest pairwise distance is reported.
    assert agg.stability == pytest.approx(0.2)
    assert agg.confidence == pytest.approx(0.7)
    assert (agg.asked, agg.answered, agg.unassessed) == (3, 3, False)
    assert [v["variant"] for v in agg.variants] == ["order1", "order2", "order3"]
    assert agg.variants[0]["choice"] == "A"


def test_pairwise_tv_reports_the_worst_pair_not_the_average():
    node = _multiclass_node()
    qs = _node_questions(node)
    q1, q2, q3 = qs.items
    answers = {
        q1.qid: {"probabilities": {"A": 0.7, "B": 0.2, "C": 0.1}},
        q2.qid: {"probabilities": {"A": 0.6, "B": 0.3, "C": 0.1}},
        q3.qid: {"probabilities": {"A": 0.3, "B": 0.3, "C": 0.4}},
    }
    agg = aggregate_classification(node, qs.items, answers)
    # TV(1,2) = (0.1 + 0.1 + 0) / 2 = 0.1; TV(1,3) = (0.4 + 0.1 + 0.3) / 2 = 0.4;
    # TV(2,3) = (0.3 + 0 + 0.3) / 2 = 0.3. Mean of the pairs would be 0.267.
    assert agg.stability == pytest.approx(0.4)
    assert agg.probabilities == pytest.approx({"A": 1.6 / 3, "B": 0.8 / 3, "C": 0.6 / 3})
    # Identical answers in every order are perfectly stable.
    same = {q.qid: {"probabilities": {"A": 0.2, "B": 0.5, "C": 0.3}} for q in qs.items}
    assert aggregate_classification(node, qs.items, same).stability == pytest.approx(0.0)


def test_classification_answers_are_normalized_and_one_presentation_is_unstable():
    node = _multiclass_node()
    qs = _node_questions(node)
    answers = {qs.items[1].qid: {"probabilities": {"A": 2.0, "B": 1.0, "C": 1.0, "unknown": 9.0}}}
    agg = aggregate_classification(node, qs.items, answers)
    # Keys that match no label are ignored; the rest is normalized.
    assert agg.probabilities == pytest.approx({"A": 0.5, "B": 0.25, "C": 0.25})
    assert len(agg.variants) == 1
    # One answered presentation out of three cannot show stability.
    assert (agg.asked, agg.answered) == (3, 1)
    assert agg.unassessed is True
    assert agg.stability == 1.0
    # Without a reported confidence the margin of the winning label stands in.
    assert agg.confidence == pytest.approx(0.0)


def test_classification_labels_match_case_and_space_insensitively():
    node = _multiclass_node()
    qs = _node_questions(node, choice_orders=2)
    answers = {
        qs.items[0].qid: {"probabilities": {" a ": 0.5, "b": 0.3, "C": 0.2}},
        # An exact key wins over a key that only matches after folding.
        qs.items[1].qid: {"probabilities": {"A": 0.6, "a": 0.0, "B ": 0.2, "c": 0.2}},
    }
    agg = aggregate_classification(node, qs.items, answers)
    assert agg.variants[0]["probabilities"] == pytest.approx({"A": 0.5, "B": 0.3, "C": 0.2})
    assert agg.variants[1]["probabilities"] == pytest.approx({"A": 0.6, "B": 0.2, "C": 0.2})
    assert agg.probabilities == pytest.approx({"A": 0.55, "B": 0.25, "C": 0.2})
    assert agg.stability == pytest.approx(0.1)


def _abstained(agg, labels):
    return (
        agg.probabilities == pytest.approx({label: 1.0 / len(labels) for label in labels})
        and agg.predicted_label == labels[0]
        and agg.variants == []
        and (agg.stability, agg.confidence, agg.answered, agg.unassessed) == (1.0, 0.0, 0, True)
    )


def test_choice_answers_matching_no_label_are_missing_not_uniform():
    node = _multiclass_node()
    qs = _node_questions(node, choice_orders=2)
    answers = {
        qs.items[0].qid: {"probabilities": {"A": 0.8, "B": 0.1, "C": 0.1}},
        qs.items[1].qid: {"probabilities": {"alpha": 0.5, "beta": 0.5}},
    }
    agg = aggregate_classification(node, qs.items, answers)
    # The unmatched answer is not averaged in as a uniform distribution.
    assert agg.probabilities == pytest.approx({"A": 0.8, "B": 0.1, "C": 0.1})
    assert (agg.asked, agg.answered, agg.unassessed) == (2, 1, True)
    assert agg.stability == 1.0
    # All-zero probabilities on the labels are missing as well: nothing usable,
    # so the node abstains.
    zero = {qs.items[0].qid: {"probabilities": {"A": 0.0, "B": 0.0, "C": 0.0}}}
    abstained = aggregate_classification(node, qs.items, zero)
    assert _abstained(abstained, ["A", "B", "C"])
    assert abstained.asked == 2


def test_classification_tie_goes_to_the_first_label_and_nothing_usable_abstains():
    node = _multiclass_node()
    qs = _node_questions(node, choice_orders=1)
    tie = {qs.items[0].qid: {"probabilities": {"A": 1.0, "B": 1.0, "C": 1.0}}}
    single = aggregate_classification(node, qs.items, tie)
    assert single.predicted_label == "A"
    # One presentation asked and answered: nothing to compare, nothing unstable.
    assert (single.asked, single.answered, single.stability, single.unassessed) == (1, 1, 0.0, False)
    for missing in (
        {qs.items[0].qid: {"probabilities": {}}},
        {qs.items[0].qid: {"probabilities": {"D": 1.0}}},
        {qs.items[0].qid: {"probabilities": {"A": float("nan"), "B": float("inf")}}},
        {qs.items[0].qid: "not an answer"},
        {},
    ):
        # No exception: an abstaining aggregate the critic rejects.
        assert _abstained(aggregate_classification(node, qs.items, missing), ["A", "B", "C"])


def test_choice_answers_may_be_level_ordered_lists_and_skip_non_finite_values():
    node = _multiclass_node()
    qs = _node_questions(node, choice_orders=2)
    answers = {
        # A list is read as "0", "1", "2": keys that match no label, so missing.
        qs.items[0].qid: {"probabilities": [0.5, 0.3, 0.2]},
        # Non-finite values are dropped; the rest is normalized.
        qs.items[1].qid: {"probabilities": {"A": 0.6, "B": float("nan"), "C": 0.2}},
    }
    agg = aggregate_classification(node, qs.items, answers)
    assert agg.probabilities == pytest.approx({"A": 0.75, "B": 0.0, "C": 0.25})
    assert (agg.asked, agg.answered, agg.unassessed) == (2, 1, True)


def test_binary_nouls_form_one_presentation_beside_the_choice_orders():
    node = _binary_node()
    qs = _node_questions(node)
    noul1, noul2, order1, order2 = qs.items
    assert [q.kind for q in qs.items] == ["binary_noul", "binary_noul", "choice", "choice"]
    assert noul1.order == ["MDD", "CONTROL"] and noul2.order == ["CONTROL", "MDD"]
    assert order2.order == ["CONTROL", "MDD"]
    answers = {
        noul1.qid: {"noul": 0.8},  # P(MDD) = 0.8
        noul2.qid: {"noul": 0.3},  # P(CONTROL) = 0.3, so P(MDD) = 0.7
        order1.qid: {"probabilities": {"MDD": 0.7, "CONTROL": 0.3}, "confidence": 0.5},
        order2.qid: {"probabilities": {"CONTROL": 0.4, "MDD": 0.6}, "confidence": 0.5},
    }
    agg = aggregate_classification(node, qs.items, answers)
    assert agg.probabilities == pytest.approx({"MDD": (0.8 + 0.7 + 0.7 + 0.6) / 4, "CONTROL": (0.2 + 0.3 + 0.3 + 0.4) / 4})
    assert agg.probabilities["MDD"] == pytest.approx(0.7)
    assert agg.variants[0]["probabilities"] == pytest.approx({"MDD": 0.8, "CONTROL": 0.2})
    assert agg.variants[1]["probabilities"] == pytest.approx({"CONTROL": 0.3, "MDD": 0.7})
    assert [v.get("orientation") for v in agg.variants[:4]] == ["first", "second", "first", "second"]
    # Presentations: order1 0.7, order2 0.6 and the Noul pair (0.8 + 0.7) / 2 = 0.75.
    # On two labels TV is |P1(MDD) - P2(MDD)|: the widest pair is 0.6 against 0.75.
    assert agg.stability == pytest.approx(0.15)
    # The pair's lean toward "yes": (0.8 + 0.3 - 1) / 2.
    assert agg.variants[4] == {"variant": "noul_pair", "yes_bias": pytest.approx(0.05)}
    # A Noul's confidence is its margin |2p - 1|: 0.6 and 0.4, with 0.5 and 0.5.
    assert agg.confidence == pytest.approx((0.6 + 0.4 + 0.5 + 0.5) / 4)
    assert agg.predicted_label == "MDD"
    assert (agg.asked, agg.answered, agg.unassessed) == (4, 4, False)


@pytest.mark.parametrize("bias", [0.0, 0.05, 0.15, -0.1])
def test_two_nouls_cancel_a_constant_yes_bias(bias):
    node = _binary_node()
    noul1, noul2, order1, order2 = _node_questions(node).items
    belief = 0.6  # the model's real P(MDD)
    answers = {
        # A model that leans towards "yes" by `bias` on every Noul.
        noul1.qid: {"noul": belief + bias},
        noul2.qid: {"noul": (1.0 - belief) + bias},
        order1.qid: {"probabilities": {"MDD": belief, "CONTROL": 1.0 - belief}},
        order2.qid: {"probabilities": {"CONTROL": 1.0 - belief, "MDD": belief}},
    }
    full = aggregate_classification(node, [noul1, noul2, order1, order2], answers)
    # P(MDD) = mean(p1, 1 - p2, choices) = mean(0.6 + b, 0.6 - b, 0.6, 0.6) = 0.6 for any b.
    assert full.probabilities["MDD"] == pytest.approx(belief)
    # The pair (0.6) agrees with both Choice orders: the bias is not instability ...
    assert full.stability == pytest.approx(0.0)
    assert full.unassessed is False
    # ... it is reported on its own: ((0.6 + b) + (0.4 + b) - 1) / 2 = b.
    assert full.variants[-1]["variant"] == "noul_pair"
    assert full.variants[-1]["yes_bias"] == pytest.approx(bias)


def test_binary_unassessed_follows_the_orientations_answered():
    node = _binary_node()
    noul1, noul2, order1, order2 = _node_questions(node).items
    # One Noul only (on CONTROL, "second" orientation): unassessed.
    one = aggregate_classification(node, [noul1, noul2, order1, order2], {noul2.qid: {"noul": 0.25}})
    assert one.probabilities == pytest.approx({"MDD": 0.75, "CONTROL": 0.25})
    assert (one.asked, one.answered, one.stability, one.unassessed) == (4, 1, 1.0, True)
    # Two answers that both put MDD first (noul_1 and order1): still one orientation.
    same_side = aggregate_classification(
        node,
        [noul1, noul2, order1, order2],
        {noul1.qid: {"noul": 0.7}, order1.qid: {"probabilities": {"MDD": 0.6, "CONTROL": 0.4}}},
    )
    assert (same_side.answered, same_side.unassessed, same_side.stability) == (2, True, 1.0)
    # order1 (MDD first) and noul_2 (CONTROL first) cover both orientations: compared.
    both = aggregate_classification(
        node,
        [noul1, noul2, order1, order2],
        {order1.qid: {"probabilities": {"MDD": 0.7, "CONTROL": 0.3}}, noul2.qid: {"noul": 0.4}},
    )
    assert (both.answered, both.unassessed) == (2, False)
    # 0.7 against 1 - 0.4 = 0.6.
    assert both.stability == pytest.approx(0.1)
    assert both.probabilities["MDD"] == pytest.approx(0.65)
    # Only one orientation asked at all (a single Noul): nothing to compare, not unassessed.
    lone = aggregate_classification(node, [noul1], {noul1.qid: {"noul": 0.7}})
    assert (lone.asked, lone.unassessed, lone.stability) == (1, False, 0.0)


def test_a_binary_node_answered_only_by_its_nouls_cannot_show_stability():
    node = _binary_node()
    noul1, noul2, order1, order2 = _node_questions(node).items
    # The Nouls disagree strongly (yes_bias 0.3), and no Choice order came back.
    agg = aggregate_classification(node, [noul1, noul2, order1, order2], {noul1.qid: {"noul": 0.9}, noul2.qid: {"noul": 0.7}})
    assert agg.answered == 2
    assert agg.unassessed is True


def test_binary_noul_follows_label_order_and_is_clamped():
    node = _binary_node(labels=("CONTROL", "MDD"))
    qs = _node_questions(node)
    noul = qs.items[0]
    agg = aggregate_classification(node, [noul], {noul.qid: {"noul": 0.8}})
    assert agg.probabilities == pytest.approx({"CONTROL": 0.8, "MDD": 0.2})
    clamped = aggregate_classification(node, [noul], {noul.qid: {"noul": 1.4}})
    assert clamped.probabilities == pytest.approx({"CONTROL": 1.0, "MDD": 0.0})
    for bad in ("not a number", float("nan"), float("inf"), None):
        assert _abstained(aggregate_classification(node, [noul], {noul.qid: {"noul": bad}}), ["CONTROL", "MDD"])
    # The second Noul asks about the second label (here MDD).
    second = qs.items[1]
    assert second.order == ["MDD", "CONTROL"]
    agg = aggregate_classification(node, [second], {second.qid: {"noul": 0.8}})
    assert agg.probabilities == pytest.approx({"CONTROL": 0.2, "MDD": 0.8})
    floor = aggregate_classification(node, [second], {second.qid: {"noul": -0.3}})
    assert floor.probabilities == pytest.approx({"CONTROL": 1.0, "MDD": 0.0})


def test_noul_value_reads_only_finite_numbers():
    assert noul_value({"noul": 0.25}) == 0.25
    assert noul_value({"noul": "0.4"}) == 0.4
    assert noul_value({"noul": 1.7}) == 1.0 and noul_value({"noul": -1}) == 0.0
    for bad in ({"noul": float("nan")}, {"noul": float("-inf")}, {"noul": "yes"}, {}, None, [0.5]):
        assert noul_value(bad) is None


# ---------------------------------------------------------------------------
# Aggregation: regression
# ---------------------------------------------------------------------------
def _madrs_node(scale=None):
    return PredictionTaskNode(
        node_id="madrs",
        display_name="MADRS total",
        mode=PredictionMode.UNIVARIATE_REGRESSION,
        regression_outputs=["madrs_total"],
        output_scales={"madrs_total": scale or {"min": 0, "max": 60, "integer": True, "unit": "points"}},
    )


def _madrs_scale():
    return deterministic_book(PredictionTaskSpec(root=_madrs_node())).nodes["madrs"].scales["madrs_total"]


def test_score_distribution_maps_the_descending_order_back():
    asc, desc = _node_questions(_madrs_node()).items
    assert asc.variant == "ascending" and desc.variant == "descending"
    up = score_distribution(asc, {"probabilities": {"0": 0.7, "1": 0.3}})
    assert up[0] == pytest.approx(0.7) and up[1] == pytest.approx(0.3) and sum(up[2:]) == 0.0
    down = score_distribution(desc, {"probabilities": {"0": 0.7, "1": 0.3}})
    assert down[9] == pytest.approx(0.7) and down[8] == pytest.approx(0.3) and sum(down[:8]) == 0.0


def test_score_distribution_parses_keys_and_falls_back():
    _asc, desc = _node_questions(_madrs_node()).items
    parsed = score_distribution(desc, {"probabilities": {"2.0": 1.0, "level": 5.0, "15": 3.0, "3": -1.0}})
    assert parsed[7] == pytest.approx(1.0) and sum(parsed) == pytest.approx(1.0)
    # Only the weighted score: the presented level 2 of a descending Score is bin 7.
    by_score = score_distribution(desc, {"score": 2})
    assert by_score[7] == 1.0 and sum(by_score) == 1.0
    # Nothing usable is None, never a uniform distribution.
    for unusable in ({}, {"probabilities": {}}, {"probabilities": {"0": float("nan")}}, {"score": float("nan")}, {"score": "high"}):
        assert score_distribution(desc, unusable) is None


def test_score_levels_may_be_a_list_or_criterion_text():
    asc, desc = _node_questions(_madrs_node()).items
    # A list is level-ordered: presented levels 0 and 1 of a descending Score are bins 9 and 8.
    listed = score_distribution(desc, {"probabilities": [0.75, 0.25] + [0.0] * 8})
    assert listed[9] == pytest.approx(0.75) and listed[8] == pytest.approx(0.25)
    # Keys may be the criterion text itself, matched case and space insensitively.
    text = asc.payload["criteria"][3]
    assert text.startswith("19 to 24 points; ")
    by_text = score_distribution(asc, {"probabilities": {"  " + text.upper() + " ": 1.0}})
    assert by_text[3] == pytest.approx(1.0)
    # Non-finite values are skipped, the rest kept.
    mixed = score_distribution(asc, {"probabilities": {"1": float("inf"), "2": 0.5, "3": 0.5}})
    assert mixed[1] == 0.0 and mixed[2] == pytest.approx(0.5) and mixed[3] == pytest.approx(0.5)


def test_answer_usable_matches_what_the_aggregation_can_use():
    spec = _mixed_spec()
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    by_kind = {}
    for q in qs.items:
        by_kind.setdefault(q.kind, q)
    choice, score, noul, suff = by_kind["choice"], by_kind["score_coarse"], by_kind["binary_noul"], by_kind["sufficiency"]
    assert answer_usable(choice, {"probabilities": {"mild": 1.0}})
    assert not answer_usable(choice, {"probabilities": {"moderate": 1.0}})
    assert not answer_usable(choice, {"probabilities": {}})
    assert answer_usable(score, {"probabilities": {"3": 1.0}})
    assert answer_usable(score, {"score": 4})
    assert not answer_usable(score, {})
    assert not answer_usable(score, {"probabilities": {"0": float("nan")}})
    assert answer_usable(noul, {"noul": 0.3}) and answer_usable(suff, {"noul": 1})
    assert not answer_usable(noul, {"noul": "maybe"}) and not answer_usable(suff, {})
    for q in (choice, score, noul, suff):
        assert not answer_usable(q, None) and not answer_usable(q, "text") and not answer_usable(q, [])


def test_regression_aggregate_combines_orders_and_refinement():
    node = _madrs_node()
    scale = _madrs_scale()
    asc, desc = _node_questions(node).items
    # Coarse integer levels: 0 to 6, 7 to 12, 13 to 18, 19 to 24, 25 to 30, ...
    # Ascending puts all mass on level 2 (13 to 18, center 15.5); descending
    # presents level 5 as bin 9 - 5 = 4 (25 to 30, center 27.5).
    answers = {asc.qid: {"probabilities": {"2": 1.0}}, desc.qid: {"probabilities": {"5": 1.0}}}
    bins, coarse, runs, _conf = coarse_ensemble([asc, desc], answers)
    assert coarse[2] == pytest.approx(0.5) and coarse[4] == pytest.approx(0.5)
    assert [r["mean"] for r in runs] == pytest.approx([15.5, 27.5])
    window = refinement_window(scale, bins, coarse)
    assert window == (2, 5)

    fine_qs = QuestionSet(prefix="r")
    lo, hi = bins[2].lo, bins[4].hi
    assert (lo, hi) == (12.5, 30.5)
    fine_bins = scale.fine_grid(lo, hi, 10)
    add_fine_regression_questions(fine_qs, node, "madrs_total", scale, fine_bins, lo, hi)
    fine_asc, fine_desc = fine_qs.items
    # Both orders pick fine level 4 (21 to 22): presented 4 ascending, 5 descending.
    answers[fine_asc.qid] = {"probabilities": {"4": 1.0}, "confidence": 0.9}
    answers[fine_desc.qid] = {"probabilities": {"5": 1.0}, "confidence": 0.7}

    agg = aggregate_regression_output(scale, [asc, desc], fine_qs.items, answers, window)
    expected = density_summary(
        combine_density(bins, coarse, window=window, fine_bins=fine_bins, fine_probs=[0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0])
    )
    # Fine level 4 is the integers 21 and 22, the interval [20.5, 22.5]: the
    # estimate is its centre and its sd that of a uniform of width 2.
    assert (fine_bins[4].lo, fine_bins[4].hi) == (20.5, 22.5)
    assert agg.value == pytest.approx(21.5)
    assert agg.summary["sd"] == pytest.approx(2.0 / math.sqrt(12.0))
    assert agg.value == pytest.approx(expected["mean"])
    assert agg.summary == pytest.approx(expected)
    assert agg.value == agg.summary["mean"]
    # Instability in reference SDs: no reference SD, so a sixth of the range
    # (60 / 6 = 10). Coarse gap 27.5 - 15.5 = 12 gives 1.2 (not clipped to 1);
    # the refined orders agree (gap 0); the largest gap counts.
    assert scale.reference_sd is None
    assert agg.stability == pytest.approx(12.0 / 10.0)
    assert agg.window == (12.5, 30.5)
    assert agg.confidence == pytest.approx(0.8)
    assert agg.detail["refined"] is True
    assert agg.detail["window"] == [12.5, 30.5]
    assert len(agg.detail["fine_levels"]) == 10
    assert [r["variant"] for r in agg.detail["coarse_runs"]] == ["ascending", "descending"]
    assert [r["mean"] for r in agg.detail["fine_runs"]] == pytest.approx([21.5, 21.5])


def test_regression_aggregate_without_refinement_uses_the_coarse_density():
    asc, desc = _node_questions(_madrs_node()).items
    answers = {asc.qid: {"probabilities": {"2": 1.0}}, desc.qid: {"probabilities": {"5": 1.0}}}
    agg = aggregate_regression_output(_madrs_scale(), [asc, desc], [], answers, None)
    assert agg.value == pytest.approx(0.5 * 15.5 + 0.5 * 27.5)
    assert agg.stability == pytest.approx(1.2)
    assert agg.fine is None and agg.window is None
    assert agg.detail["refined"] is False
    assert agg.confidence == 0.5
    assert agg.unassessed is False


def test_regression_without_usable_answers_abstains_over_the_whole_range():
    asc, desc = _node_questions(_madrs_node()).items
    for answers in ({}, {asc.qid: {}, desc.qid: {"probabilities": {"0": float("nan")}}}):
        agg = aggregate_regression_output(_madrs_scale(), [asc, desc], [], answers, None)
        # Uniform over the ten coarse levels: the mean of their centres,
        # (3 + 9.5 + 15.5 + ... + 57.5) / 10 = 304.5 / 10.
        assert agg.coarse == pytest.approx([0.1] * 10)
        assert agg.value == pytest.approx(30.45)
        assert (agg.stability, agg.confidence, agg.unassessed) == (1.0, 0.0, True)
        assert agg.detail["abstained"] is True and agg.detail["refined"] is False
        assert agg.fine is None and agg.window is None


def test_flat_coarse_answers_are_not_refined():
    scale = _madrs_scale()
    bins = scale.coarse_grid(10)
    # Ten levels, a window of three: refine only when it holds more than 3/10 + 0.05.
    assert refinement_window(scale, bins, [0.1] * 10) is None
    below = [0.34 / 3] * 3 + [0.66 / 7] * 7
    assert refinement_window(scale, bins, below) is None
    above = [0.36 / 3] * 3 + [0.64 / 7] * 7
    assert refinement_window(scale, bins, above) == (0, 3)
    peaked = [0.0, 0.0, 0.2, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert refinement_window(scale, bins, peaked) == (2, 5)


def test_regression_reference_sd_has_a_floor_of_a_twentieth_of_the_range():
    node = _madrs_node()
    scale = _madrs_scale()
    # Bypass the scale's own validation to reach the aggregation floor.
    scale.reference_sd = 0.5
    asc, desc = _node_questions(node).items
    answers = {asc.qid: {"probabilities": {"2": 1.0}}, desc.qid: {"probabilities": {"5": 1.0}}}
    agg = aggregate_regression_output(scale, [asc, desc], [], answers, None)
    # Gap 12 against max(0.5, 60 / 20 = 3), not against 0.5.
    assert agg.stability == pytest.approx(12.0 / 3.0)


def test_regression_instability_uses_the_reference_sd_when_known():
    node = _madrs_node({"min": 0, "max": 60, "integer": True, "reference_mean": 20, "reference_sd": 8})
    scale = deterministic_book(PredictionTaskSpec(root=node)).nodes["madrs"].scales["madrs_total"]
    assert scale.reference_sd == 8.0
    asc, desc = _node_questions(node).items
    # Means 15.5 and 21.5 (levels 2 and 3): a gap of 6, or 0.75 reference SDs.
    answers = {asc.qid: {"probabilities": {"2": 1.0}}, desc.qid: {"probabilities": {"6": 1.0}}}
    agg = aggregate_regression_output(scale, [asc, desc], [], answers, None)
    assert [r["mean"] for r in agg.detail["coarse_runs"]] == pytest.approx([15.5, 21.5])
    assert agg.stability == pytest.approx(6.0 / 8.0)


def test_regression_instability_takes_the_larger_of_the_coarse_and_refined_gaps():
    node = _madrs_node()
    scale = _madrs_scale()
    asc, desc = _node_questions(node).items
    # The coarse orders agree on level 3 (19 to 24).
    answers = {asc.qid: {"probabilities": {"3": 1.0}}, desc.qid: {"probabilities": {"6": 1.0}}}
    bins, coarse, _runs, _conf = coarse_ensemble([asc, desc], answers)
    window = refinement_window(scale, bins, coarse)
    # A single peak at level 3: the window centred on it (levels 2 to 4) wins the tie.
    assert window == (2, 5)
    fine_qs = QuestionSet(prefix="r")
    fine_bins = scale.fine_grid(bins[2].lo, bins[4].hi, 10)
    add_fine_regression_questions(fine_qs, node, "madrs_total", scale, fine_bins, bins[2].lo, bins[4].hi)
    fine_asc, fine_desc = fine_qs.items
    # Refined: ascending says 15 to 16 (center 15.5), descending presents 2 as
    # bin 7 (27 to 28, center 27.5). A gap of 12 against a reference SD of 10.
    answers[fine_asc.qid] = {"probabilities": {"1": 1.0}}
    answers[fine_desc.qid] = {"probabilities": {"2": 1.0}}
    agg = aggregate_regression_output(scale, [asc, desc], fine_qs.items, answers, window)
    assert [r["mean"] for r in agg.detail["coarse_runs"]] == pytest.approx([21.5, 21.5])
    assert [r["mean"] for r in agg.detail["fine_runs"]] == pytest.approx([15.5, 27.5])
    assert agg.stability == pytest.approx(1.2)


def test_regression_with_a_missing_order_is_unstable():
    node = _madrs_node()
    scale = _madrs_scale()
    asc, desc = _node_questions(node).items
    only_up = {asc.qid: {"probabilities": {"3": 1.0}}}
    agg = aggregate_regression_output(scale, [asc, desc], [], only_up, None)
    assert agg.value == pytest.approx(21.5)
    assert agg.stability == 1.0
    assert agg.unassessed is True
    # A refined round with one of its two orders missing is unstable as well,
    # even when the coarse orders agree.
    both = {asc.qid: {"probabilities": {"3": 1.0}}, desc.qid: {"probabilities": {"6": 1.0}}}
    bins = asc.bins
    fine_qs = QuestionSet(prefix="r")
    fine_bins = scale.fine_grid(bins[2].lo, bins[4].hi, 10)
    add_fine_regression_questions(fine_qs, node, "madrs_total", scale, fine_bins, bins[2].lo, bins[4].hi)
    both[fine_qs.items[0].qid] = {"probabilities": {"4": 1.0}}
    refined = aggregate_regression_output(scale, [asc, desc], fine_qs.items, both, (2, 5))
    assert refined.stability == 1.0
    assert refined.unassessed is True
    # A single asked order is not penalised.
    single = aggregate_regression_output(scale, [asc], [], only_up, None)
    assert single.stability == 0.0
    assert single.unassessed is False
    # Both orders answered: assessed, whatever the gap.
    assert aggregate_regression_output(scale, [asc, desc], [], both, None).unassessed is False


def test_consistent_orders_are_perfectly_stable_and_exact_scales_skip_refinement():
    node = _madrs_node({"min": 0, "max": 6, "integer": True})
    scale = deterministic_book(PredictionTaskSpec(root=node)).nodes["madrs"].scales["madrs_total"]
    asc, desc = _node_questions(node).items
    answers = {
        asc.qid: {"probabilities": {"3": 0.5, "4": 0.5}},
        desc.qid: {"probabilities": {"3": 0.5, "2": 0.5}},  # presented 3 and 2 are values 3 and 4
    }
    bins, coarse, _runs, _conf = coarse_ensemble([asc, desc], answers)
    assert refinement_window(scale, bins, coarse) is None
    agg = aggregate_regression_output(scale, [asc, desc], [], answers, None)
    assert agg.value == pytest.approx(3.5)
    assert agg.stability == pytest.approx(0.0)


def test_confidence_levels_and_node_prediction_flags():
    assert confidence_level(0.75) == ConfidenceLevel.HIGH
    assert confidence_level(0.45) == ConfidenceLevel.MEDIUM
    assert confidence_level(0.44) == ConfidenceLevel.LOW

    node = _multiclass_node()
    qs = _node_questions(node, choice_orders=2)
    answers = {
        qs.items[0].qid: {"probabilities": {"A": 0.9, "B": 0.05, "C": 0.05}, "confidence": 0.3},
        qs.items[1].qid: {"probabilities": {"A": 0.1, "B": 0.8, "C": 0.1}, "confidence": 0.3},
    }
    agg = aggregate_classification(node, qs.items, answers)
    # TV = (0.8 + 0.75 + 0.05) / 2 = 0.8.
    assert agg.stability == pytest.approx(0.8)
    pred = build_node_prediction(node, path="severity", classification=agg, regression={}, children=[], sufficiency=0.3)
    assert pred.classification is not None and pred.regression is None
    assert pred.confidence_level == ConfidenceLevel.LOW
    details = pred.decision_details
    assert details["source"] == "decision_model"
    assert details["instability"] == pytest.approx(0.8)
    assert details["instability_max_tv"] == pytest.approx(0.8)
    assert details["instability_threshold"] == 0.25
    assert details["presentations"] == {"asked": 2, "answered": 2}
    assert "instability_reference_sd" not in details and "stability_tv" not in details
    assert details["evidence_sufficiency"] == 0.3
    assert any("evidence insufficient" in f for f in pred.uncertainty_factors)
    assert "Answer moved between presentation orders (instability 0.80)." in pred.uncertainty_factors

    # The flag follows the node's own threshold, inclusive.
    lenient = build_node_prediction(
        node, path="severity", classification=agg, regression={}, children=[], sufficiency=None, stability_threshold=0.8
    )
    assert lenient.decision_details["instability_threshold"] == 0.8
    assert not any("presentation orders" in f for f in lenient.uncertainty_factors)
    assert "evidence_sufficiency" not in lenient.decision_details


def test_regression_node_prediction_reports_instability_in_reference_sds():
    node = _madrs_node()
    asc, desc = _node_questions(node).items
    answers = {asc.qid: {"probabilities": {"3": 1.0}, "confidence": 0.6}, desc.qid: {"probabilities": {"5": 1.0}, "confidence": 0.6}}
    # Means 21.5 and 27.5: a gap of 6 over a reference SD of 10 is 0.6.
    agg = aggregate_regression_output(_madrs_scale(), [asc, desc], [], answers, None)
    assert agg.stability == pytest.approx(0.6)
    pred = build_node_prediction(node, path="madrs", classification=None, regression={"madrs_total": agg}, children=[], sufficiency=None)
    details = pred.decision_details
    assert details["instability"] == pytest.approx(0.6)
    assert details["instability_reference_sd"] == pytest.approx(0.6)
    assert details["instability_threshold"] == 0.5
    assert "instability_max_tv" not in details and "stability_range_fraction" not in details
    assert "Answer moved between presentation orders (instability 0.60)." in pred.uncertainty_factors
    assert pred.regression.values["madrs_total"] == pytest.approx(24.5)
    relaxed = build_node_prediction(
        node,
        path="madrs",
        classification=None,
        regression={"madrs_total": agg},
        children=[],
        sufficiency=None,
        regression_stability_threshold=0.6,
    )
    assert relaxed.decision_details["instability_threshold"] == 0.6
    assert not any("presentation orders" in f for f in relaxed.uncertainty_factors)


def test_unassessed_nodes_get_their_own_uncertainty_note():
    node = _multiclass_node()
    qs = _node_questions(node)
    # One of three orders answered, and that one far from the others is unknowable.
    agg = aggregate_classification(node, qs.items, {qs.items[0].qid: {"probabilities": {"A": 0.9, "B": 0.05, "C": 0.05}}})
    pred = build_node_prediction(node, path="severity", classification=agg, regression={}, children=[], sufficiency=None)
    assert pred.decision_details["unassessed"] is True
    assert pred.decision_details["instability"] == 1.0
    # The note says why, instead of reporting an instability that was never measured.
    assert pred.uncertainty_factors == [UNASSESSED_NOTE]

    # A regression node is unassessed when any of its outputs is.
    items = PredictionTaskNode(
        node_id="items",
        display_name="Items",
        mode=PredictionMode.MULTIVARIATE_REGRESSION,
        regression_outputs=["a", "b"],
        output_scales={"a": {"min": 0, "max": 6, "integer": True}, "b": {"min": 0, "max": 6, "integer": True}},
    )
    scales = deterministic_book(PredictionTaskSpec(root=items)).nodes["items"].scales
    item_qs = _node_questions(items).items
    a_asc, a_desc, b_asc, b_desc = item_qs
    answers = {
        a_asc.qid: {"probabilities": {"2": 1.0}},
        a_desc.qid: {"probabilities": {"4": 1.0}},  # presented 4 is value 2: no gap
        b_asc.qid: {"probabilities": {"5": 1.0}},  # b's descending order is missing
    }
    regression = {
        "a": aggregate_regression_output(scales["a"], [a_asc, a_desc], [], answers, None),
        "b": aggregate_regression_output(scales["b"], [b_asc, b_desc], [], answers, None),
    }
    assert (regression["a"].unassessed, regression["a"].stability) == (False, 0.0)
    assert (regression["b"].unassessed, regression["b"].stability) == (True, 1.0)
    reg_pred = build_node_prediction(items, path="items", classification=None, regression=regression, children=[], sufficiency=None)
    assert reg_pred.decision_details["unassessed"] is True
    assert reg_pred.decision_details["instability"] == 1.0
    assert reg_pred.uncertainty_factors == [UNASSESSED_NOTE]
    assert reg_pred.regression.values == pytest.approx({"a": 2.0, "b": 5.0})


def test_a_regression_node_takes_the_instability_of_its_worst_output():
    items = PredictionTaskNode(
        node_id="items",
        display_name="Items",
        mode=PredictionMode.MULTIVARIATE_REGRESSION,
        regression_outputs=["a", "b"],
        output_scales={"a": {"min": 0, "max": 6, "integer": True}, "b": {"min": 0, "max": 6, "integer": True}},
    )
    scales = deterministic_book(PredictionTaskSpec(root=items)).nodes["items"].scales
    a_asc, a_desc, b_asc, b_desc = _node_questions(items).items
    answers = {
        # a: ascending 3.6, descending presented 3 is value 3: a gap of 0.6 (reference SD 1).
        a_asc.qid: {"probabilities": {"3": 0.4, "4": 0.6}},
        a_desc.qid: {"probabilities": {"3": 1.0}},
        b_asc.qid: {"probabilities": {"1": 1.0}},
        b_desc.qid: {"probabilities": {"5": 1.0}},
    }
    regression = {
        out: aggregate_regression_output(scales[out], qs, [], answers, None)
        for out, qs in (("a", [a_asc, a_desc]), ("b", [b_asc, b_desc]))
    }
    assert regression["a"].stability == pytest.approx(0.6)
    assert regression["b"].stability == pytest.approx(0.0)
    pred = build_node_prediction(items, path="items", classification=None, regression=regression, children=[], sufficiency=None)
    # Max, not mean (0.3): above the 0.5 threshold, so the node is flagged.
    assert pred.decision_details["instability"] == pytest.approx(0.6)
    assert pred.decision_details["instability_reference_sd"] == pytest.approx(0.6)
    assert pred.uncertainty_factors == ["Answer moved between presentation orders (instability 0.60)."]


# ---------------------------------------------------------------------------
# DecisionPredictor end to end
# ---------------------------------------------------------------------------
def test_decision_predictor_end_to_end_on_a_hierarchical_mixed_task(settings, direct_output):
    spec = _mixed_spec()
    oracle = OracleClient()
    tokens = RecordingTokenManager()
    predictor = _predictor(oracle, token_manager=tokens)
    result = _run(predictor, direct_output, spec)

    # A valid PredictionResult that survives a round trip.
    assert isinstance(result, PredictionResult)
    PredictionResult.model_validate(result.model_dump())
    assert result.predictor_kind == "decision"
    assert result.predictor_model == JEV
    assert result.input_route == "direct"
    assert result.participant_id == direct_output["participant_id"]
    assert result.domains_processed == direct_output["domains_processed"]
    # A multiclass root has no binary aliases.
    assert result.binary_classification is None and result.probability_score is None

    nodes = {n.node_id: n for n in result.root_prediction.walk()}
    assert set(nodes) == {"severity", "madrs", "items", "response"}
    assert [n.node_id for n in result.flat_predictions] == ["severity", "madrs", "items", "response"]

    severity = nodes["severity"].classification
    assert severity.predicted_label == "mild"
    assert severity.probabilities == pytest.approx({"none": 0.2, "mild": 0.5, "severe": 0.3})
    response = nodes["response"].classification
    assert response.predicted_label == "responder"
    assert response.probabilities == pytest.approx({"responder": 0.65, "non_responder": 0.35})

    madrs = nodes["madrs"].regression
    value = madrs.values["madrs_total"]
    summary = madrs.uncertainty["madrs_total"]
    assert set(summary) == {"mean", "sd", "q05", "q25", "median", "q75", "q95"}
    assert value == summary["mean"]
    assert abs(value - TARGETS["madrs_total"]) < 1.5
    assert 0.0 < summary["sd"] < 8.0
    assert summary["q05"] <= summary["q25"] <= summary["median"] <= summary["q75"] <= summary["q95"]
    madrs_detail = nodes["madrs"].decision_details["outputs"]["madrs_total"]
    assert madrs_detail["refined"] is True
    # Target 22: levels 13 to 18, 19 to 24 and 25 to 30 carry the most mass.
    assert madrs_detail["window"] == [12.5, 30.5]

    items = nodes["items"].regression
    assert items.values["item_sadness"] == pytest.approx(4.0, abs=0.01)
    assert items.values["item_sleep"] == pytest.approx(2.0, abs=0.01)
    for out in ("item_sadness", "item_sleep"):
        detail = nodes["items"].decision_details["outputs"][out]
        assert detail["refined"] is False
        assert len(detail["coarse_levels"]) == 7

    # Sufficiency lives on the root only.
    assert nodes["severity"].decision_details["evidence_sufficiency"] == pytest.approx(0.8)
    assert "evidence_sufficiency" not in nodes["madrs"].decision_details

    # Two requests: round one, then a refinement round for the one non-exact output.
    assert len(oracle.calls) == 2
    first, second = oracle.calls
    assert first["model"] == second["model"] == JEV
    # 3 severity choices, 2 madrs scores, 2 x 2 item scores, 2 response Nouls and
    # 2 response choices, and the evidence sufficiency Noul.
    assert len(first["questions"]) == 3 + 2 + 4 + 4 + 1
    assert all(qid.startswith("q") for qid in first["questions"])
    assert sorted(second["questions"]) == ["r001", "r002"]
    assert all("(madrs_total)" in q["instructions"] for q in second["questions"].values())
    assert all("somewhere from 13 to 30 points" in q["instructions"] for q in second["questions"].values())
    assert {"task_context", "clinical_record", "measurements", "deviation_profile"} <= set(first["state"])
    assert first["state"]["task_context"] == CONTEXT
    assert second["state"] == first["state"]

    report = result.decision_report
    for key in (
        "model",
        "model_label",
        "route",
        "question_book",
        "question_book_hash",
        "questions",
        "state",
        "requests",
        "cost_usd",
        "input_tokens",
        "quality",
        "settings",
    ):
        assert key in report
    assert report["model"] == JEV and report["route"] == "direct"
    assert set(report["question_book"]["nodes"]) == {"severity", "madrs", "items", "response"}
    assert report["question_book_hash"] == deterministic_book(spec, CONTEXT).content_hash()
    asked = report["questions"]
    assert len(asked) == 16
    fine = [q for q in asked if q["kind"] == "score_fine"]
    assert [(q["node_id"], q["output"]) for q in fine] == [("madrs", "madrs_total"), ("madrs", "madrs_total")]
    assert [q["variant"] for q in asked if q["node_id"] == "response"] == ["noul_1", "noul_2", "order1", "order2"]
    # Every question was answered on the first request, so there was no re-ask.
    assert [r["round"] for r in report["requests"]] == ["coarse", "refine"]
    assert report["input_tokens"] == 2 * 1200 == result.total_tokens_used
    assert report["cost_usd"] == pytest.approx(2 * 0.0000504)
    assert report["state"]["features_total"] == len(direct_output["coverage_ledger"]["all_features"])
    assert report["state"]["feature_coverage"] == 1.0
    quality = report["quality"]
    assert set(quality) == {
        "evidence_sufficiency",
        "mean_instability",
        "max_instability",
        "default_scale_outputs",
        "placeholder_items",
        "mean_confidence",
        "feature_coverage",
        "per_node",
    }
    assert quality["evidence_sufficiency"] == pytest.approx(0.8)
    assert quality["max_instability"] < 1e-9
    assert quality["default_scale_outputs"] == []
    # The deterministic book (no companion LLM here) keeps generic wording.
    assert quality["placeholder_items"] == [
        "severity: question",
        "severity.none: definition",
        "severity.mild: definition",
        "severity.severe: definition",
        "madrs: question",
        "items: question",
        "response: question",
        "response.responder: definition",
        "response.non_responder: definition",
    ]
    assert quality["feature_coverage"] == 1.0
    per_node = quality["per_node"]
    assert set(per_node) == {"severity", "madrs", "items", "response"}
    for node_id, kind, threshold in (
        ("severity", "classification", 0.25),
        ("response", "classification", 0.25),
        ("madrs", "regression", 0.5),
        ("items", "regression", 0.5),
    ):
        stats = per_node[node_id]
        assert set(stats) == {"kind", "instability", "threshold", "confidence", "unassessed", "instability_ratio"}
        assert (stats["kind"], stats["threshold"], stats["unassessed"]) == (kind, threshold, False)
        assert stats["instability"] < 1e-9 and stats["instability_ratio"] < 1e-9
    assert report["settings"] == {
        "choice_orders": 3,
        "stability_threshold": 0.25,
        "regression_stability_threshold": 0.5,
        "score_levels": 10,
        "regression_refine": True,
        "tokenizer_ratio": 1.2,
    }
    assert nodes["response"].decision_details["presentations"] == {"asked": 4, "answered": 4}
    assert nodes["severity"].decision_details["presentations"] == {"asked": 3, "answered": 3}
    assert nodes["madrs"].decision_details["instability_threshold"] == 0.5
    assert all(n.decision_details["unassessed"] is False for n in nodes.values())

    assert result.key_findings
    assert all(f.z_score is not None for f in result.key_findings)
    assert result.clinical_summary.startswith("Structured decision by TypeSafe Jev 1.13 on the direct evidence route")
    assert [r["step_tool"] for r in tokens.records] == [f"DecisionModel:{JEV}:coarse", f"DecisionModel:{JEV}:refine"]

    # The decision critic accepts a stable, fully covered answer.
    evaluation = evaluate_decision_prediction(result, spec)
    assert evaluation.verdict == Verdict.SATISFACTORY


def test_binary_root_derives_the_binary_aliases(settings, direct_output):
    spec = _binary_spec()
    oracle = OracleClient()
    result = _run(_predictor(oracle), direct_output, spec, target="MDD", control="CONTROL")
    assert result.binary_classification == BinaryClassification.CASE
    assert result.probability_score == pytest.approx(0.7)
    assert result.confidence_level == result.root_prediction.confidence_level
    # Classification only: one request (a Noul per label, two Choice orders, sufficiency).
    assert len(oracle.calls) == 1
    assert len(oracle.calls[0]["questions"]) == 5
    noul_texts = [q["instructions"] for q in oracle.calls[0]["questions"].values() if q["type"] == "noul"]
    assert sum("Does this participant belong to the 'MDD' group?" in t for t in noul_texts) == 1
    assert sum("Does this participant belong to the 'CONTROL' group?" in t for t in noul_texts) == 1

    flipped = OracleClient(label_weights={"MDD": 0.3, "CONTROL": 0.7})
    control = _run(_predictor(flipped), direct_output, spec, target="MDD", control="CONTROL")
    assert control.binary_classification == BinaryClassification.CONTROL
    assert control.probability_score == pytest.approx(0.3)


class OrderSensitiveClient(OracleClient):
    """The oracle, except that the descending item_sadness Score lands on 3.4 instead of 4."""

    def _answer(self, payload):
        if payload["type"] == "score" and "(item_sadness)" in payload["instructions"] and payload["criteria"][0].startswith("6;"):
            # Descending order: presented level 3 is value 3, presented level 2 is value 4.
            return {"score": 3, "probabilities": {"3": 0.6, "2": 0.4}, "confidence": 0.6}
        return super()._answer(payload)


def test_one_unstable_output_makes_its_node_unstable(settings, direct_output):
    spec = _mixed_spec()
    result = _run(_predictor(OrderSensitiveClient()), direct_output, spec)
    nodes = {n.node_id: n for n in result.root_prediction.walk()}
    runs = nodes["items"].decision_details["outputs"]["item_sadness"]["coarse_runs"]
    assert [r["mean"] for r in runs] == pytest.approx([4.0, 3.4], abs=1e-4)
    per_node = result.decision_report["quality"]["per_node"]["items"]
    # item_sadness moved 0.6 against a reference SD of 1 (a sixth of 0 to 6);
    # item_sleep did not move. The node takes its worst output.
    assert per_node["instability"] == pytest.approx(0.6, abs=1e-4)
    assert per_node["instability_ratio"] == pytest.approx(1.2, abs=1e-3)
    assert result.decision_report["quality"]["max_instability"] == pytest.approx(1.2, abs=1e-3)
    assert evaluate_decision_prediction(result, spec).verdict == Verdict.UNSATISFACTORY


def test_node_details_report_the_same_instability_the_critic_reads(settings, direct_output):
    result = _run(_predictor(OrderSensitiveClient()), direct_output, _mixed_spec())
    items = next(n for n in result.root_prediction.walk() if n.node_id == "items")
    per_node = result.decision_report["quality"]["per_node"]["items"]
    # The node is as unstable as its worst output, in its details and for the critic.
    assert per_node["instability"] == pytest.approx(0.6, abs=1e-4)
    assert items.decision_details["instability"] == pytest.approx(per_node["instability"])
    assert items.decision_details["instability_reference_sd"] == pytest.approx(per_node["instability"])
    assert items.decision_details["unassessed"] is False
    assert any(f.startswith("Answer moved between presentation orders (instability 0.60") for f in items.uncertainty_factors)


def test_a_missing_level_order_is_unstable_whatever_the_regression_threshold(settings, direct_output):
    settings.decision.regression_stability_threshold = 1.5
    spec = PredictionTaskSpec(root=_madrs_node())
    # q001 is the ascending coarse Score, q002 the descending one (never answered).
    client = ForgetfulClient(omit={"q002"}, always=True)
    result = _run(_predictor(client), direct_output, spec)
    details = result.root_prediction.decision_details
    assert details["instability"] == 1.0
    assert details["unassessed"] is True
    assert result.root_prediction.uncertainty_factors == [UNASSESSED_NOTE]
    stats = result.decision_report["quality"]["per_node"]["madrs"]
    assert stats["unassessed"] is True
    # 1.0 / 1.5 would pass; an unassessed node is reported at twice its threshold.
    assert stats["instability_ratio"] == 2.0
    assert result.decision_report["quality"]["max_instability"] == 2.0
    evaluation = evaluate_decision_prediction(result, spec)
    assert evaluation.checklist.decision_stable is False
    assert evaluation.verdict == Verdict.UNSATISFACTORY


def test_refinement_can_be_switched_off(settings, direct_output):
    settings.decision.regression_refine = False
    oracle = OracleClient()
    result = _run(_predictor(oracle), direct_output, _mixed_spec())
    assert len(oracle.calls) == 1
    nodes = {n.node_id: n for n in result.root_prediction.walk()}
    assert nodes["madrs"].decision_details["outputs"]["madrs_total"]["refined"] is False
    assert result.decision_report["settings"]["regression_refine"] is False
    assert not [q for q in result.decision_report["questions"] if q["kind"] == "score_fine"]


def test_choice_orders_setting_is_respected(settings, direct_output):
    settings.decision.choice_orders = 2
    oracle = OracleClient()
    result = _run(_predictor(oracle), direct_output, _mixed_spec())
    severity_choices = [q for q in result.decision_report["questions"] if q["node_id"] == "severity" and q["kind"] == "choice"]
    assert len(severity_choices) == 2


def test_measure_direct_reports_the_full_state_against_the_state_budget(settings, direct_output):
    spec = _mixed_spec()
    predictor = _predictor()
    measurement = predictor.measure_direct(dict(direct_output), prediction_task_spec=spec, target_condition="x", control_condition="")
    assert measurement["predictor_kind"] == "decision"
    assert measurement["model"] == JEV
    assert measurement["input_tokens"] > 0
    qs = round_one_questions(spec, deterministic_book(spec, CONTEXT), choice_orders=3, score_levels=10)
    assert measurement["budget_tokens"] == predictor._state_budget(qs)
    assert 1000 <= measurement["budget_tokens"] < int(32_000 / 1.2)
    # Nothing was asked to measure.
    assert predictor.client.calls == []


@pytest.mark.parametrize("spec_factory", [_mixed_spec, _binary_spec])
def test_the_dashboard_question_reserve_covers_the_engine_reserve(settings, spec_factory):
    """The cost projection keeps back DECISION_STATE_QUESTION_RESERVE tokens of the
    state for questions; the engine keeps back the longest question plus 256."""
    from src.full_stack.backend.api import cost as cost_module

    predictor = _predictor()
    spec = spec_factory()
    qs = round_one_questions(spec, deterministic_book(spec, CONTEXT), choice_orders=3, score_levels=10)
    engine_reserve = predictor.spec.state_budget_cl100k(ratio=1.2) - predictor._state_budget(qs)
    assert engine_reserve == max(predictor._question_tokens(qs)) + 256
    # Conservative: the dashboard never projects a direct route the engine would refuse.
    assert cost_module.DECISION_STATE_QUESTION_RESERVE == 800
    assert engine_reserve <= cost_module.DECISION_STATE_QUESTION_RESERVE


def test_length_error_repacks_the_state_to_a_smaller_budget(settings, direct_output):
    spec = _mixed_spec()
    error = DecisionRequestError("Decision request returned HTTP 400", status=400, body="context length exceeded")
    assert error.looks_like_length_error
    oracle = OracleClient(failures=[error])
    predictor = _predictor(oracle)
    full = predictor.measure_direct(dict(direct_output), prediction_task_spec=spec)["input_tokens"]
    budget = full + 64
    predictor._state_budget = lambda questions: budget

    result = _run(predictor, direct_output, spec)

    assert len(oracle.calls) == 3  # rejected, repacked round one, refinement
    rejected, repacked, refine = (call["state"] for call in oracle.calls)
    # The rejected state was the full record (it fit the budget of full + 64).
    assert _tokens(rejected) == full
    # The retry shrinks what was actually sent: int(min(budget, sent) * 0.8).
    assert _tokens(repacked) <= int(full * 0.8) < _tokens(rejected)
    assert refine == repacked
    report = result.decision_report
    assert report["state"]["budget_tokens"] == int(full * 0.8)
    assert [r["round"] for r in report["requests"]] == ["coarse", "refine"]
    assert report["state"]["feature_coverage"] < 1.0
    included = report["state"]["features_included"]
    # The state report rounds to 4 decimals; the quality block keeps full precision.
    assert report["quality"]["feature_coverage"] == pytest.approx(included / report["state"]["features_total"])
    assert report["state"]["feature_coverage"] == pytest.approx(report["quality"]["feature_coverage"], abs=1e-4)
    assert any(f"State packing kept {included} of" in factor for factor in result.uncertainty_factors)
    # Coverage below 0.9 makes the decision critic reject the attempt.
    assert evaluate_decision_prediction(result, spec).verdict == Verdict.UNSATISFACTORY


def test_a_length_error_that_would_shrink_the_state_below_1000_tokens_is_raised(settings, direct_output):
    spec = _binary_spec()
    error = DecisionRequestError("HTTP 400", status=400, body="maximum context length exceeded")
    oracle = OracleClient(failures=[error, error])
    predictor = _predictor(oracle)
    predictor._state_budget = lambda questions: 1100
    with pytest.raises(DecisionRequestError):
        _run(predictor, direct_output, spec, target="MDD", control="CONTROL")
    # The state sent was at most 1100 tokens; 80 percent of it is below 1000, so
    # there is no second attempt.
    assert len(oracle.calls) == 1
    assert _tokens(oracle.calls[0]["state"]) <= 1100


def test_persistent_length_errors_give_up_after_three_attempts(settings, direct_output):
    errors = [DecisionRequestError("HTTP 413", status=413, body="request too long") for _ in range(3)]
    oracle = OracleClient(failures=errors)
    with pytest.raises(DecisionRequestError):
        _run(_predictor(oracle), direct_output, _mixed_spec())
    assert len(oracle.calls) == 3


def test_other_request_errors_are_not_retried_by_the_predictor(settings, direct_output):
    error = DecisionRequestError("HTTP 401", status=401, body="invalid api key")
    assert not error.looks_like_length_error
    oracle = OracleClient(failures=[error])
    with pytest.raises(DecisionRequestError):
        _run(_predictor(oracle), direct_output, _mixed_spec())
    assert len(oracle.calls) == 1


class ForgetfulClient(OracleClient):
    """Leaves chosen questions unanswered: on the first request only, or always."""

    def __init__(self, *, omit, always=False, **kwargs):
        super().__init__(**kwargs)
        self.omit = set(omit)
        self.always = always

    def ask(self, *, model, state, questions):
        response = super().ask(model=model, state=state, questions=questions)
        if self.always or len(self.calls) == 1:
            response.answers = {qid: a for qid, a in response.answers.items() if qid not in self.omit}
        return response


def test_unanswered_questions_are_asked_again_once(settings, direct_output):
    spec = _binary_spec()
    # q001 is the Noul on MDD; q003 the first Choice order.
    client = ForgetfulClient(omit={"q001", "q003"})
    result = _run(_predictor(client), direct_output, spec, target="MDD", control="CONTROL")
    assert len(client.calls) == 2
    first, reask = client.calls
    assert len(first["questions"]) == 5
    # Only the missing questions are asked again, on the same state.
    assert sorted(reask["questions"]) == ["q001", "q003"]
    assert reask["state"] == first["state"]
    report = result.decision_report
    assert [r["round"] for r in report["requests"]] == ["coarse", "coarse.reask"]
    assert [r["questions"] for r in report["requests"]] == [5, 2]
    assert result.root_prediction.decision_details["presentations"] == {"asked": 4, "answered": 4}
    assert result.probability_score == pytest.approx(0.7)
    assert report["quality"]["max_instability"] < 1e-9


def test_questions_still_unanswered_after_the_reask_count_as_missing(settings, direct_output):
    spec = _binary_spec()
    client = ForgetfulClient(omit={"q001", "q002", "q003"}, always=True)
    result = _run(_predictor(client), direct_output, spec, target="MDD", control="CONTROL")
    # One re-ask, never more.
    assert len(client.calls) == 2
    assert sorted(client.calls[1]["questions"]) == ["q001", "q002", "q003"]
    root = result.root_prediction
    # One of four presentations answered: the node cannot show stability.
    assert root.decision_details["presentations"] == {"asked": 4, "answered": 1}
    assert root.decision_details["instability"] == 1.0
    assert root.decision_details["unassessed"] is True
    assert UNASSESSED_NOTE in root.uncertainty_factors
    assert not any(f.startswith("Answer moved") for f in root.uncertainty_factors)
    quality = result.decision_report["quality"]
    assert quality["per_node"]["dx"]["unassessed"] is True
    # max(1.0 / 0.25, 2.0): already above the floor of twice the threshold.
    assert quality["per_node"]["dx"]["instability_ratio"] == pytest.approx(4.0)
    assert quality["max_instability"] == pytest.approx(4.0)
    evaluation = evaluate_decision_prediction(result, spec)
    assert evaluation.verdict == Verdict.UNSATISFACTORY
    assert evaluation.checklist.decision_stable is False


def test_a_failed_reask_keeps_the_answers_already_given(settings, direct_output):
    spec = _binary_spec()

    class _ReaskFails(ForgetfulClient):
        def ask(self, *, model, state, questions):
            if len(self.calls) == 1:
                self.calls.append({"model": model, "state": state, "questions": questions})
                raise DecisionRequestError("HTTP 503", status=503, body="busy")
            return super().ask(model=model, state=state, questions=questions)

    client = _ReaskFails(omit={"q004"})
    result = _run(_predictor(client), direct_output, spec, target="MDD", control="CONTROL")
    assert len(client.calls) == 2
    assert [r["round"] for r in result.decision_report["requests"]] == ["coarse"]
    assert result.root_prediction.decision_details["presentations"] == {"asked": 4, "answered": 3}
    assert result.probability_score == pytest.approx(0.7)


class SlowStartClient(OracleClient):
    """Leaves the chosen questions unanswered on its first `calls` requests."""

    def __init__(self, *, omit, calls, fail_first_reask=False, **kwargs):
        super().__init__(**kwargs)
        self.omit = set(omit)
        self.omit_calls = calls
        self.fail_first_reask = fail_first_reask

    def ask(self, *, model, state, questions):
        if self.fail_first_reask and len(self.calls) == self.omit_calls:
            self.calls.append({"model": model, "state": state, "questions": questions})
            raise DecisionRequestError("HTTP 503", status=503, body="busy")
        response = super().ask(model=model, state=state, questions=questions)
        if len(self.calls) <= self.omit_calls:
            response.answers = {qid: a for qid, a in response.answers.items() if qid not in self.omit}
        return response


class UnusableAnswersClient(OracleClient):
    """Replaces the answers to chosen questions with unusable ones (on the first `calls` requests, or always)."""

    def __init__(self, *, overrides, calls=None, **kwargs):
        super().__init__(**kwargs)
        self.overrides = dict(overrides)
        self.affected_calls = calls

    def ask(self, *, model, state, questions):
        response = super().ask(model=model, state=state, questions=questions)
        if self.affected_calls is None or len(self.calls) <= self.affected_calls:
            response.answers = {
                qid: (copy.deepcopy(self.overrides[qid]) if qid in self.overrides else answer)
                for qid, answer in response.answers.items()
            }
        return response


def test_unusable_answers_are_asked_again(settings, direct_output):
    spec = PredictionTaskSpec(root=_madrs_node())
    # q001 ascending Score, q002 descending Score, q003 sufficiency. An empty
    # Score answer and a non-finite Noul are answers in form only.
    client = UnusableAnswersClient(overrides={"q002": {}, "q003": {"noul": float("nan")}}, calls=1)
    result = _run(_predictor(client), direct_output, spec)
    assert sorted(client.calls[1]["questions"]) == ["q002", "q003"]
    assert [r["round"] for r in result.decision_report["requests"]] == ["coarse", "coarse.reask", "refine"]
    details = result.root_prediction.decision_details
    assert details["unassessed"] is False
    assert [r["variant"] for r in details["outputs"]["madrs_total"]["coarse_runs"]] == ["ascending", "descending"]
    assert result.decision_report["quality"]["evidence_sufficiency"] == pytest.approx(0.8)


def test_a_sufficiency_answer_that_is_not_a_number_is_reported_as_unknown(settings, direct_output):
    spec = _binary_spec()
    client = UnusableAnswersClient(overrides={"q005": {"noul": "plenty"}})
    result = _run(_predictor(client), direct_output, spec, target="MDD", control="CONTROL")
    assert sorted(client.calls[1]["questions"]) == ["q005"]
    assert result.decision_report["quality"]["evidence_sufficiency"] is None
    assert "evidence_sufficiency" not in result.root_prediction.decision_details


def test_a_node_without_usable_answers_abstains_and_the_critic_rejects(settings, direct_output):
    spec = _binary_spec()
    unusable = {"q001": {"noul": "n/a"}, "q002": {"noul": None}, "q003": {"probabilities": {}}, "q004": {"choice": "MDD"}}
    client = UnusableAnswersClient(overrides=unusable)
    result = _run(_predictor(client), direct_output, spec, target="MDD", control="CONTROL")
    # One re-ask of the four, then the node abstains instead of failing the run.
    assert len(client.calls) == 2
    assert sorted(client.calls[1]["questions"]) == ["q001", "q002", "q003", "q004"]
    root = result.root_prediction
    assert root.classification.probabilities == pytest.approx({"MDD": 0.5, "CONTROL": 0.5})
    assert root.decision_details["presentations"] == {"asked": 4, "answered": 0}
    assert root.decision_details["unassessed"] is True
    assert root.decision_details["variants"] == []
    assert root.confidence_score == 0.0
    stats = result.decision_report["quality"]["per_node"]["dx"]
    assert stats["unassessed"] is True and stats["instability_ratio"] == pytest.approx(4.0)
    evaluation = evaluate_decision_prediction(result, spec)
    assert evaluation.verdict == Verdict.UNSATISFACTORY
    assert evaluation.checklist.decision_stable is False
    assert "Stability could not be assessed for dx (presentation orders went unanswered)." in evaluation.weaknesses
    assert route_can_help(evaluation) is True


def test_a_regression_output_without_usable_answers_abstains(settings, direct_output):
    spec = PredictionTaskSpec(root=_madrs_node())
    client = UnusableAnswersClient(overrides={"q001": {"probabilities": {}}, "q002": {"score": "high"}})
    result = _run(_predictor(client), direct_output, spec)
    # No refinement round for an output that came back empty.
    assert [r["round"] for r in result.decision_report["requests"]] == ["coarse", "coarse.reask"]
    root = result.root_prediction
    detail = root.decision_details["outputs"]["madrs_total"]
    assert detail["abstained"] is True and detail["refined"] is False
    assert root.regression.values["madrs_total"] == pytest.approx(30.45)
    assert root.decision_details["unassessed"] is True
    assert root.uncertainty_factors == [UNASSESSED_NOTE]
    assert evaluate_decision_prediction(result, spec).verdict == Verdict.UNSATISFACTORY


def _sized_questions(predictor, sizes):
    """Give every question of a round a fixed token size, by question id."""
    predictor._question_tokens = lambda qs: [sizes[q.qid] for q in qs.items]


def test_split_puts_questions_into_requests_that_leave_room_for_the_state(settings):
    predictor = _predictor()
    state_room = predictor.spec.state_budget_cl100k(ratio=1.2)
    assert state_room == 26_666
    # A tiny request limit: 2000 tokens of questions per request beside the state.
    predictor._request_limit = lambda: state_room + 2000
    spec = _binary_spec()
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    assert [q.qid for q in qs.items] == ["q001", "q002", "q003", "q004", "q005"]
    _sized_questions(predictor, {"q001": 900, "q002": 800, "q003": 700, "q004": 600, "q005": 500})
    parts = predictor._split(qs)
    # 900 + 800 = 1700; adding 700 would make 2400 > 2000, so a new request starts.
    # 700 + 600 + 500 = 1800 fits.
    assert [[q.qid for q in part.items] for part in parts] == [["q001", "q002"], ["q003", "q004", "q005"]]
    assert all(part.prefix == "q" for part in parts)
    # Budget per part: min(state room - longest question - 256, request limit - all questions).
    assert predictor._state_budget(parts[0]) == min(26_666 - 900 - 256, 28_666 - 1700)
    assert predictor._state_budget(parts[1]) == min(26_666 - 700 - 256, 28_666 - 1800)
    assert predictor._state_budget(parts[0]) == 25_510


def test_split_never_leaves_a_request_empty(settings):
    predictor = _predictor()
    predictor._request_limit = lambda: predictor.spec.state_budget_cl100k(ratio=1.2) + 2000
    spec = _binary_spec()
    qs = round_one_questions(spec, deterministic_book(spec), choice_orders=3, score_levels=10)
    # A question larger than a request on its own still goes, alone.
    _sized_questions(predictor, {"q001": 2500, "q002": 100, "q003": 100, "q004": 2100, "q005": 100})
    parts = predictor._split(qs)
    assert [[q.qid for q in part.items] for part in parts] == [["q001"], ["q002", "q003"], ["q004"], ["q005"]]
    # The usual case: everything fits one request.
    _sized_questions(predictor, {q.qid: 300 for q in qs.items})
    assert [[q.qid for q in part.items] for part in predictor._split(qs)] == [[q.qid for q in qs.items]]


def test_a_split_round_is_asked_over_several_requests_on_one_state(settings, direct_output):
    spec = _binary_spec()
    oracle = OracleClient()
    predictor = _predictor(oracle)
    state_room = predictor.spec.state_budget_cl100k(ratio=1.2)
    predictor._request_limit = lambda: state_room + 2000
    _sized_questions(predictor, {"q001": 900, "q002": 800, "q003": 700, "q004": 600, "q005": 500})

    measurement = predictor.measure_direct(dict(direct_output), prediction_task_spec=spec)
    # The direct budget is the smallest over the requests: 26666 - 900 - 256.
    assert measurement["budget_tokens"] == 25_510
    assert oracle.calls == []

    result = _run(predictor, direct_output, spec, target="MDD", control="CONTROL")
    assert [sorted(c["questions"]) for c in oracle.calls] == [["q001", "q002"], ["q003", "q004", "q005"]]
    assert oracle.calls[0]["state"] == oracle.calls[1]["state"]
    report = result.decision_report
    assert [r["round"] for r in report["requests"]] == ["coarse.1", "coarse.2"]
    assert report["state"]["budget_tokens"] == 25_510
    assert result.probability_score == pytest.approx(0.7)
    assert result.root_prediction.decision_details["presentations"] == {"asked": 4, "answered": 4}
    assert report["quality"]["evidence_sufficiency"] == pytest.approx(0.8)


def _wide_spec(outputs) -> PredictionTaskSpec:
    return PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="severity",
            display_name="Depression severity",
            mode=PredictionMode.MULTICLASS_CLASSIFICATION,
            class_labels=["none", "mild", "severe"],
            children=[
                PredictionTaskNode(
                    node_id="scales",
                    display_name="Rating scales",
                    mode=PredictionMode.MULTIVARIATE_REGRESSION,
                    regression_outputs=list(outputs),
                    output_scales={out: {"min": 0, "max": 60, "integer": True, "unit": "points"} for out in outputs},
                )
            ],
        )
    )


def test_real_question_sizes_split_under_a_tiny_request_limit(settings, direct_output):
    targets = {f"scale_{i}": 10.0 + 5.0 * i for i in range(1, 9)}
    spec = _wide_spec(targets)
    oracle = OracleClient(targets=targets)
    predictor = _predictor(oracle)
    state_room = predictor.spec.state_budget_cl100k(ratio=1.2)
    predictor._request_limit = lambda: state_room + 2000
    qs = round_one_questions(spec, deterministic_book(spec, CONTEXT), choice_orders=3, score_levels=10)
    sizes = predictor._question_tokens(qs)
    assert sum(sizes) > 2000  # too many for one request under this limit
    parts = predictor._split(qs)
    assert len(parts) >= 2
    # Order is kept and nothing is lost or repeated.
    assert [q.qid for part in parts for q in part.items] == [q.qid for q in qs.items]
    for part in parts:
        assert sum(predictor._question_tokens(part)) <= 2000 or len(part.items) == 1

    result = _run(predictor, direct_output, spec)
    rounds = [r["round"] for r in result.decision_report["requests"]]
    assert rounds[: len(parts)] == [f"coarse.{i + 1}" for i in range(len(parts))]
    refine_rounds = rounds[len(parts):]
    assert refine_rounds and all(r == "refine" or r.startswith("refine.") for r in refine_rounds)
    # Every request carried the same packed state.
    assert all(call["state"] == oracle.calls[0]["state"] for call in oracle.calls)
    nodes = {n.node_id: n for n in result.root_prediction.walk()}
    assert nodes["severity"].classification.probabilities == pytest.approx({"none": 0.2, "mild": 0.5, "severe": 0.3})
    for out, target in targets.items():
        assert abs(nodes["scales"].regression.values[out] - target) < 1.0


def test_the_reask_is_split_like_the_round(settings, direct_output):
    spec = _binary_spec()
    client = SlowStartClient(omit={"q001", "q002", "q004"}, calls=2)
    predictor = _predictor(client)
    state_room = predictor.spec.state_budget_cl100k(ratio=1.2)
    predictor._request_limit = lambda: state_room + 2000
    _sized_questions(predictor, {"q001": 900, "q002": 800, "q003": 700, "q004": 600, "q005": 500})
    result = _run(predictor, direct_output, spec, target="MDD", control="CONTROL")
    # Round: [q001, q002] and [q003, q004, q005]. Missing: q001, q002 and q004,
    # 900 + 800 + 600 = 2300 tokens, more than 2000: two re-ask requests.
    assert [sorted(c["questions"]) for c in client.calls] == [
        ["q001", "q002"],
        ["q003", "q004", "q005"],
        ["q001", "q002"],
        ["q004"],
    ]
    assert all(c["state"] == client.calls[0]["state"] for c in client.calls)
    report = result.decision_report
    assert [r["round"] for r in report["requests"]] == ["coarse.1", "coarse.2", "coarse.reask", "coarse.reask"]
    assert [r["questions"] for r in report["requests"]] == [2, 3, 2, 1]
    assert result.root_prediction.decision_details["presentations"] == {"asked": 4, "answered": 4}
    assert result.root_prediction.decision_details["unassessed"] is False
    assert result.probability_score == pytest.approx(0.7)


def test_a_failed_reask_part_does_not_stop_the_others(settings, direct_output):
    spec = _binary_spec()
    client = SlowStartClient(omit={"q001", "q002", "q004"}, calls=2, fail_first_reask=True)
    predictor = _predictor(client)
    state_room = predictor.spec.state_budget_cl100k(ratio=1.2)
    predictor._request_limit = lambda: state_room + 2000
    _sized_questions(predictor, {"q001": 900, "q002": 800, "q003": 700, "q004": 600, "q005": 500})
    result = _run(predictor, direct_output, spec, target="MDD", control="CONTROL")
    assert [sorted(c["questions"]) for c in client.calls][2:] == [["q001", "q002"], ["q004"]]
    # The failed part leaves no ledger row; the other part's answer is kept.
    assert [r["round"] for r in result.decision_report["requests"]] == ["coarse.1", "coarse.2", "coarse.reask"]
    # Answered: q003 (first Choice order) and q004 (second order); both Nouls missing.
    assert result.root_prediction.decision_details["presentations"] == {"asked": 4, "answered": 2}
    assert result.root_prediction.decision_details["unassessed"] is False


def test_a_question_set_without_room_for_a_state_is_refused(settings, direct_output):
    spec = _binary_spec()
    oracle = OracleClient()
    predictor = _predictor(oracle)
    # The longest question leaves 26666 - 25500 - 256 = 910 tokens for the state.
    _sized_questions(predictor, {"q001": 25_500, "q002": 10, "q003": 10, "q004": 10, "q005": 10})
    with pytest.raises(ValueError, match="leaves no room for a state"):
        _run(predictor, direct_output, spec, target="MDD", control="CONTROL")
    assert oracle.calls == []


def test_more_than_255_labels_fail_before_any_request(settings, direct_output):
    labels = [f"subtype_{i:03d}" for i in range(256)]
    spec = PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="subtype",
            display_name="Subtype",
            mode=PredictionMode.MULTICLASS_CLASSIFICATION,
            class_labels=labels,
        )
    )
    oracle = OracleClient(label_weights={label: 1.0 for label in labels})
    predictor = _predictor(oracle)
    with pytest.raises(ValueError, match="at most 255"):
        predictor.measure_direct(dict(direct_output), prediction_task_spec=spec)
    with pytest.raises(ValueError, match="at most 255"):
        _run(predictor, direct_output, spec, target="Subtype")
    assert oracle.calls == []


BINARY_BOOK = {
    "nodes": {
        "dx": {
            "question": "Does this participant meet criteria for MDD?",
            "label_definitions": {"MDD": "Meets DSM-5 criteria for MDD.", "CONTROL": "No psychiatric diagnosis."},
        }
    }
}


def test_question_book_cache_is_keyed_by_the_task_not_the_object(settings):
    compiled = BINARY_BOOK

    class StubLLM:
        def __init__(self):
            self.calls = []

        def call(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(content=json.dumps(compiled), prompt_tokens=10, completion_tokens=10)

    llm = StubLLM()
    predictor = DecisionPredictor(llm_client=llm)
    predictor.set_runtime_instruction(CONTEXT)
    book = predictor.question_book(_binary_spec())
    assert len(llm.calls) == 1
    # Remove the disk cache: an equal task (another object) is served from memory.
    for path in cache_dir().glob("*.json"):
        path.unlink()
    assert predictor.question_book(_binary_spec()) is book
    assert len(llm.calls) == 1
    # Another task context is another instrument.
    predictor.set_runtime_instruction("Another study.")
    other = predictor.question_book(_binary_spec())
    assert other is not book
    assert len(llm.calls) == 2
    assert other.task_hash == task_hash(_binary_spec(), "Another study.")


MIXED_BOOK = {
    "nodes": {
        "severity": {"question": "How severe is the depression?", "label_definitions": {"none": "No symptoms.", "mild": "Mild.", "severe": "Severe."}},
        "madrs": {"question": "What is the MADRS total?"},
        "items": {"question": "What are the item scores?"},
        "response": {
            "question": "Will the participant respond?",
            "label_definitions": {"responder": "50 percent reduction.", "non_responder": "Less than 50 percent."},
        },
    }
}


def test_a_wrapped_multi_node_reply_is_compiled_whole(settings):
    class StubLLM:
        def __init__(self):
            self.calls = 0

        def call(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(content=json.dumps(MIXED_BOOK), prompt_tokens=1, completion_tokens=1)

    llm = StubLLM()
    predictor = DecisionPredictor(llm_client=llm)
    predictor.set_runtime_instruction(CONTEXT)
    assert sorted(predictor._companion_json("system", "user")["nodes"]) == ["items", "madrs", "response", "severity"]
    book = predictor.question_book(_mixed_spec())
    assert book.placeholder_items() == [] and book.notes == []
    assert llm.calls == 2  # the direct call above, then one compilation


def test_question_book_is_compiled_once_by_the_companion_llm(settings):
    # The reply keyed directly by node id (also accepted; see the xfail above
    # for the wrapped form).
    compiled = {
        "nodes": {
            "severity": {"question": "How severe is the depression?", "label_definitions": {"none": "No symptoms.", "mild": "Mild.", "severe": "Severe."}},
            "madrs": {"question": "What is the MADRS total?"},
            "items": {"question": "What are the item scores?"},
            "response": {
                "question": "Will the participant respond?",
                "label_definitions": {"responder": "50 percent reduction.", "non_responder": "Less than 50 percent."},
            },
        }
    }["nodes"]

    class StubLLM:
        def __init__(self):
            self.calls = []

        def call(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(content=json.dumps(compiled), prompt_tokens=300, completion_tokens=200)

    settings.decision.compiler_model = "stub/compiler-llm"
    spec = _mixed_spec()
    llm = StubLLM()
    tokens = RecordingTokenManager()
    predictor = DecisionPredictor(llm_client=llm, token_manager=tokens)
    predictor.set_runtime_instruction(CONTEXT)
    book = predictor.question_book(spec)
    assert book.compiled_by == "stub/compiler-llm"
    assert book.notes == [] and book.placeholder_items() == []
    assert book.nodes["response"].label_definitions["non_responder"] == "Less than 50 percent."
    assert book.nodes["severity"].label_definitions["mild"] == "Mild."
    assert book.nodes["madrs"].scales["madrs_total"].source == "task_spec"
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["model"] == "stub/compiler-llm"
    assert call["temperature"] == 0.0
    assert call["response_format"] == {"type": "json_object"}
    assert CONTEXT in call["messages"][1]["content"]
    assert tokens.records[0]["step_tool"] == "QuestionBookCompiler"

    assert predictor.question_book(spec) is book
    assert len(llm.calls) == 1
    assert list(cache_dir().glob("*.json"))

    # A new predictor (the next participant) reads the frozen book from disk.
    other_llm = StubLLM()
    other = DecisionPredictor(llm_client=other_llm)
    other.set_runtime_instruction(CONTEXT)
    assert other.question_book(spec).to_dict() == book.to_dict()
    assert other_llm.calls == []


def test_companion_defaults_to_the_orchestrator_model(settings):
    settings.models.orchestrator_model = "stub/orchestrator-llm"
    seen = []

    class StubLLM:
        def call(self, **kwargs):
            seen.append(kwargs["model"])
            return SimpleNamespace(content=json.dumps({"nodes": {}}), prompt_tokens=1, completion_tokens=1)

    predictor = DecisionPredictor(llm_client=StubLLM())
    book = predictor.question_book(_mixed_spec())
    # An empty reply leaves every question generic: retried once, then returned
    # as compiled by that model with an incomplete note.
    assert seen == ["stub/orchestrator-llm", "stub/orchestrator-llm"]
    assert book.compiled_by == "stub/orchestrator-llm"
    assert book.notes[-1].startswith("Compilation incomplete (ValueError: compiled book left severity: question")


def test_an_incomplete_book_is_not_memoized(settings):
    replies = [{"nodes": {"dx": {"question": "Does this participant meet criteria for MDD?"}}}] * 2 + [BINARY_BOOK]

    class StubLLM:
        def __init__(self):
            self.calls = 0

        def call(self, **kwargs):
            reply = replies[self.calls]
            self.calls += 1
            return SimpleNamespace(content=json.dumps(reply), prompt_tokens=1, completion_tokens=1)

    llm = StubLLM()
    predictor = DecisionPredictor(llm_client=llm)
    predictor.set_runtime_instruction(CONTEXT)
    first = predictor.question_book(_binary_spec())
    # Both attempts left the class definitions generic.
    assert llm.calls == 2
    assert first.placeholder_items() == ["dx.MDD: definition", "dx.CONTROL: definition"]
    assert any(note.startswith("Compilation incomplete") for note in first.notes)
    assert list(cache_dir().glob("*.json")) == []
    # Not kept in memory either: the next call compiles again, and succeeds.
    second = predictor.question_book(_binary_spec())
    assert llm.calls == 3
    assert second is not first and second.placeholder_items() == [] and second.notes == []
    assert predictor.question_book(_binary_spec()) is second
    assert llm.calls == 3


def test_decision_predictor_refuses_an_llm_predictor_model(settings):
    settings.models.predictor_model = "openai/gpt-5"
    with pytest.raises(ValueError, match="not a structured decision model"):
        DecisionPredictor(llm_client=SimpleNamespace())


# ---------------------------------------------------------------------------
# Decision critic
# ---------------------------------------------------------------------------
def _critic_spec(*, child_required=True) -> PredictionTaskSpec:
    return PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="severity",
            display_name="Depression severity",
            mode=PredictionMode.MULTICLASS_CLASSIFICATION,
            class_labels=["none", "mild", "severe"],
            children=[
                PredictionTaskNode(
                    node_id="madrs",
                    display_name="MADRS total",
                    mode=PredictionMode.UNIVARIATE_REGRESSION,
                    regression_outputs=["madrs_total"],
                    required=child_required,
                )
            ],
        )
    )


def _decision_prediction(spec, *, route="direct", include_child=True, **quality) -> PredictionResult:
    children = []
    if include_child:
        children.append(
            NodePrediction(
                node_id="madrs",
                path="severity/madrs",
                mode=PredictionMode.UNIVARIATE_REGRESSION,
                regression=RegressionPrediction(values={"madrs_total": 21.5}),
            )
        )
    root = NodePrediction(
        node_id="severity",
        path="severity",
        mode=PredictionMode.MULTICLASS_CLASSIFICATION,
        classification=ClassificationPrediction(predicted_label="mild", probabilities={"none": 0.2, "mild": 0.5, "severe": 0.3}),
        children=children,
    )
    # Instabilities in the quality block are multiples of each node's threshold.
    q = {
        "evidence_sufficiency": 0.8,
        "mean_instability": 0.1,
        "max_instability": 0.2,
        "default_scale_outputs": [],
        "mean_confidence": 0.7,
        "feature_coverage": 1.0,
        "per_node": {
            "severity": {"kind": "classification", "instability": 0.05, "threshold": 0.25, "confidence": 0.7, "instability_ratio": 0.2},
            "madrs": {"kind": "regression", "instability": 0.0, "threshold": 0.5, "confidence": 0.7, "instability_ratio": 0.0},
        },
    }
    q.update(quality)
    return PredictionResult(
        prediction_id="decision_test",
        participant_id="P001",
        target_condition="Depression severity",
        control_condition="",
        prediction_task_spec=spec,
        root_prediction=root,
        predictor_kind="decision",
        predictor_model=JEV,
        input_route=route,
        decision_report={"route": route, "quality": q},
    )


def test_decision_critic_accepts_stable_and_covered_answers():
    spec = _critic_spec()
    evaluation = evaluate_decision_prediction(_decision_prediction(spec), spec)
    assert evaluation.verdict == Verdict.SATISFACTORY
    checklist = evaluation.checklist
    assert checklist.decision_stable and checklist.sufficient_coverage and checklist.evidence_sufficient
    assert checklist.output_schema_valid and checklist.has_required_outputs and checklist.hierarchy_consistent
    assert "decision_stable" in checklist.active_checks
    assert "evidence_sufficient" not in checklist.active_checks
    assert checklist.all_passed
    assert 0.0 <= evaluation.composite_score <= 1.0
    assert "reported only" in evaluation.reasoning
    assert evaluation.improvement_suggestions == []
    assert "Answers were stable across presentation orders (largest shift 0.20 of its threshold)." in evaluation.strengths
    # stability score = max(0, 1 - mean ratio / 2) = 1 - 0.1 / 2.
    assert evaluation.score_breakdown["stability"] == pytest.approx(0.95)
    assert evaluation.score_breakdown["schema"] == 1.0


def test_decision_critic_takes_no_stability_threshold_argument():
    spec = _critic_spec()
    with pytest.raises(TypeError):
        evaluate_decision_prediction(_decision_prediction(spec), spec, stability_threshold=0.2)


@pytest.mark.parametrize("ratio, verdict", [(0.99, Verdict.SATISFACTORY), (1.0, Verdict.SATISFACTORY), (1.0001, Verdict.UNSATISFACTORY)])
def test_decision_critic_threshold_is_inclusive_at_ratio_one(ratio, verdict):
    spec = _critic_spec()
    evaluation = evaluate_decision_prediction(_decision_prediction(spec, max_instability=ratio), spec)
    assert evaluation.verdict == verdict
    assert evaluation.checklist.decision_stable is (verdict == Verdict.SATISFACTORY)


@pytest.mark.parametrize("mean, score", [(0.0, 1.0), (0.5, 0.75), (1.0, 0.5), (2.0, 0.0), (3.5, 0.0)])
def test_decision_critic_stability_score_is_one_minus_half_the_mean_ratio(mean, score):
    spec = _critic_spec()
    evaluation = evaluate_decision_prediction(_decision_prediction(spec, mean_instability=mean, max_instability=0.5), spec)
    assert evaluation.score_breakdown["stability"] == pytest.approx(score)


def test_decision_critic_rejects_order_instability():
    spec = _critic_spec()
    prediction = _decision_prediction(
        spec,
        max_instability=1.4,
        mean_instability=0.8,
        per_node={
            "severity": {"kind": "classification", "instability": 0.35, "threshold": 0.25, "confidence": 0.6, "instability_ratio": 1.4},
            "madrs": {"kind": "regression", "instability": 0.1, "threshold": 0.5, "confidence": 0.6, "instability_ratio": 0.2},
        },
    )
    evaluation = evaluate_decision_prediction(prediction, spec)
    assert evaluation.verdict == Verdict.UNSATISFACTORY
    assert evaluation.checklist.decision_stable is False
    assert "Answers shifted with presentation order (largest shift 1.40 times its threshold) for severity." in evaluation.weaknesses
    assert [s.issue for s in evaluation.improvement_suggestions] == ["Order-sensitive decision"]
    assert "multi-agent workflow" in evaluation.concise_summary
    assert evaluation.score_breakdown["stability"] == pytest.approx(0.6)
    assert "max instability=1.40 of the threshold" in evaluation.reasoning

    orchestrated = _decision_prediction(spec, route="orchestrated", max_instability=1.4)
    assert "re-orchestrates" in evaluate_decision_prediction(orchestrated, spec).concise_summary


def test_regression_instability_is_judged_against_its_own_threshold():
    """A regression node moving 0.4 reference SDs is within its 0.5 threshold; the
    same raw number on a classification node (0.4 TV against 0.25) is not."""
    spec = _critic_spec()
    per_node = {
        "severity": {"kind": "classification", "instability": 0.1, "threshold": 0.25, "confidence": 0.7, "instability_ratio": 0.4},
        "madrs": {"kind": "regression", "instability": 0.4, "threshold": 0.5, "confidence": 0.7, "instability_ratio": 0.8},
    }
    ok = evaluate_decision_prediction(_decision_prediction(spec, max_instability=0.8, mean_instability=0.6, per_node=per_node), spec)
    assert ok.verdict == Verdict.SATISFACTORY
    per_node["severity"].update(instability=0.4, instability_ratio=1.6)
    bad = evaluate_decision_prediction(_decision_prediction(spec, max_instability=1.6, mean_instability=1.2, per_node=per_node), spec)
    assert bad.verdict == Verdict.UNSATISFACTORY
    assert any(w.endswith("for severity.") for w in bad.weaknesses)


def test_default_scale_outputs_fail_the_schema_check():
    spec = _critic_spec()
    evaluation = evaluate_decision_prediction(_decision_prediction(spec, default_scale_outputs=["madrs.madrs_total"]), spec)
    assert evaluation.verdict == Verdict.UNSATISFACTORY
    assert evaluation.checklist.output_schema_valid is False
    # The outputs themselves are present and valid; only the scale is unknown.
    assert evaluation.checklist.has_required_outputs is True
    assert evaluation.checklist.regression_values_valid is True
    assert evaluation.score_breakdown["schema"] == 0.0
    assert (
        "No measurement scale was available for madrs.madrs_total; the estimate used a generic standardized "
        "scale and is not interpretable." in evaluation.weaknesses
    )
    assert [s.issue for s in evaluation.improvement_suggestions] == ["Unknown output scale"]
    assert "schema=False" in evaluation.reasoning


def test_decision_critic_rejects_lost_coverage():
    spec = _critic_spec()
    evaluation = evaluate_decision_prediction(_decision_prediction(spec, feature_coverage=0.85), spec)
    assert evaluation.verdict == Verdict.UNSATISFACTORY
    assert evaluation.checklist.sufficient_coverage is False
    assert [s.issue for s in evaluation.improvement_suggestions] == ["Coverage lost in state packing"]
    at_gate = evaluate_decision_prediction(_decision_prediction(spec, feature_coverage=0.9), spec)
    assert at_gate.verdict == Verdict.SATISFACTORY


def test_sufficiency_is_reported_but_not_gating_at_threshold_zero():
    spec = _critic_spec()
    evaluation = evaluate_decision_prediction(_decision_prediction(spec, evidence_sufficiency=0.2), spec, sufficiency_threshold=0.0)
    assert evaluation.verdict == Verdict.SATISFACTORY
    assert evaluation.checklist.evidence_sufficient is True
    assert "evidence_sufficient" not in evaluation.checklist.active_checks
    assert any("evidence thin" in w for w in evaluation.weaknesses)
    assert evaluation.score_breakdown["evidence_sufficiency"] == pytest.approx(0.2)
    assert "reported only" in evaluation.reasoning


def test_sufficiency_gates_when_the_threshold_is_positive():
    spec = _critic_spec()
    low = evaluate_decision_prediction(_decision_prediction(spec, evidence_sufficiency=0.3), spec, sufficiency_threshold=0.5)
    assert low.verdict == Verdict.UNSATISFACTORY
    assert low.checklist.evidence_sufficient is False
    assert "evidence_sufficient" in low.checklist.active_checks
    assert [s.issue for s in low.improvement_suggestions] == ["Insufficient evidence in the state"]
    assert "gate 0.50" in low.reasoning

    high = evaluate_decision_prediction(_decision_prediction(spec, evidence_sufficiency=0.6), spec, sufficiency_threshold=0.5)
    assert high.verdict == Verdict.SATISFACTORY
    unknown = evaluate_decision_prediction(_decision_prediction(spec, evidence_sufficiency=None), spec, sufficiency_threshold=0.5)
    assert unknown.verdict == Verdict.SATISFACTORY


def test_low_confidence_is_reported_but_never_a_reason_to_reject():
    spec = _critic_spec()
    evaluation = evaluate_decision_prediction(_decision_prediction(spec, mean_confidence=0.2), spec)
    assert evaluation.verdict == Verdict.SATISFACTORY
    assert any("Low decision confidence" in w for w in evaluation.weaknesses)


def test_missing_required_node_is_a_schema_failure():
    spec = _critic_spec()
    evaluation = evaluate_decision_prediction(_decision_prediction(spec, include_child=False), spec)
    assert evaluation.verdict == Verdict.UNSATISFACTORY
    assert evaluation.checklist.has_required_outputs is False
    assert evaluation.checklist.output_schema_valid is False
    assert evaluation.checklist.hierarchy_consistent is False
    assert any("missing nodes: madrs" in w for w in evaluation.weaknesses)

    optional = _critic_spec(child_required=False)
    assert evaluate_decision_prediction(_decision_prediction(optional, include_child=False), optional).verdict == Verdict.SATISFACTORY


def test_placeholder_wording_is_reported_but_not_a_rejection():
    spec = _critic_spec()
    items = [f"node{i}: question" for i in range(8)]
    evaluation = evaluate_decision_prediction(_decision_prediction(spec, placeholder_items=items), spec)
    assert evaluation.verdict == Verdict.SATISFACTORY
    assert (
        "The question book still uses generic wording for node0: question, node1: question, node2: question, "
        "node3: question, node4: question, node5: question and 2 more; add class_definitions or check the "
        "companion LLM." in evaluation.weaknesses
    )
    few = evaluate_decision_prediction(_decision_prediction(spec, placeholder_items=["severity.mild: definition"]), spec)
    assert (
        "The question book still uses generic wording for severity.mild: definition; add class_definitions or "
        "check the companion LLM." in few.weaknesses
    )


def test_unassessed_nodes_are_named_in_the_weaknesses():
    spec = _critic_spec()
    per_node = {
        "severity": {"kind": "classification", "instability": 1.0, "threshold": 0.25, "confidence": 0.0, "unassessed": True, "instability_ratio": 4.0},
        "madrs": {"kind": "regression", "instability": 0.0, "threshold": 0.5, "confidence": 0.7, "unassessed": False, "instability_ratio": 0.0},
    }
    evaluation = evaluate_decision_prediction(_decision_prediction(spec, max_instability=4.0, mean_instability=2.0, per_node=per_node), spec)
    assert evaluation.verdict == Verdict.UNSATISFACTORY
    assert "Stability could not be assessed for severity (presentation orders went unanswered)." in evaluation.weaknesses


def test_route_can_help_only_when_a_high_priority_issue_depends_on_the_evidence():
    spec = _critic_spec()
    # Only an unknown output scale: no evidence route can fix it.
    scale_only = evaluate_decision_prediction(_decision_prediction(spec, default_scale_outputs=["madrs.madrs_total"]), spec)
    assert [s.issue for s in scale_only.improvement_suggestions] == ["Unknown output scale"]
    assert route_can_help(scale_only) is False
    # An order-sensitive answer as well: distilling the record can help.
    both = evaluate_decision_prediction(
        _decision_prediction(spec, default_scale_outputs=["madrs.madrs_total"], max_instability=1.5), spec
    )
    assert [s.issue for s in both.improvement_suggestions] == ["Unknown output scale", "Order-sensitive decision"]
    assert route_can_help(both) is True
    # Lost coverage alone.
    assert route_can_help(evaluate_decision_prediction(_decision_prediction(spec, feature_coverage=0.5), spec)) is True
    # Nothing to fix.
    accepted = evaluate_decision_prediction(_decision_prediction(spec), spec)
    assert accepted.improvement_suggestions == [] and route_can_help(accepted) is True


def test_critic_agent_dispatches_decision_predictions(settings):
    # The thresholds live in the Predictor's ratios; the critic agent passes none.
    settings.decision.stability_threshold = 0.01
    spec = _critic_spec()
    critic = Critic(llm_client=SimpleNamespace())  # any LLM call would fail
    with contextlib.redirect_stdout(io.StringIO()):
        rejected = critic.execute(
            prediction=_decision_prediction(spec, max_instability=1.25),
            executor_output={},
            data_overview={},
            prediction_task_spec=spec,
        )
        accepted = critic.execute(
            prediction=_decision_prediction(spec, max_instability=0.75),
            executor_output={},
            data_overview={},
            prediction_task_spec=spec,
        )
    assert rejected.verdict == Verdict.UNSATISFACTORY
    assert rejected.reasoning.startswith("Decision critic on the direct route")
    assert "max instability=1.25 of the threshold" in rejected.reasoning
    assert accepted.verdict == Verdict.SATISFACTORY


def test_critic_agent_gates_sufficiency_only_above_zero(settings):
    spec = _critic_spec()
    critic = Critic(llm_client=SimpleNamespace())
    thin = _decision_prediction(spec, evidence_sufficiency=0.2)
    with contextlib.redirect_stdout(io.StringIO()):
        settings.decision.sufficiency_threshold = 0.0
        assert critic.execute(prediction=thin, executor_output={}, data_overview={}, prediction_task_spec=spec).verdict == Verdict.SATISFACTORY
        settings.decision.sufficiency_threshold = 0.5
        gated = critic.execute(prediction=thin, executor_output={}, data_overview={}, prediction_task_spec=spec)
    assert gated.verdict == Verdict.UNSATISFACTORY
    assert "gate 0.50" in gated.reasoning


# ---------------------------------------------------------------------------
# HTTP transport against a fake session
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code, payload=None, text="", headers=None, not_json=False):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = headers or {}
        self.not_json = not_json

    def json(self):
        if self.not_json:
            # requests raises a ValueError subclass for a body that is not JSON.
            raise requests.exceptions.JSONDecodeError("Expecting value", self.text, 0)
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, headers=None, data=None, timeout=None):
        self.posts.append({"url": url, "headers": dict(headers or {}), "body": json.loads(data), "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok(cost=0.00005, answers=None):
    usage = {"input_tokens": 1000, "output_tokens": 0}
    if cost is not None:
        usage["cost"] = cost
    return FakeResponse(200, {"id": "gen-1", "model": JEV, "answers": answers or {"q001": {"noul": 0.7}}, "usage": usage})


QUESTION = {"q001": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes", "false": "no"}}}


def test_openrouter_decisions_url(monkeypatch):
    assert _openrouter_decisions_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/alpha/decisions"
    assert _openrouter_decisions_url("https://openrouter.ai/api/v1/") == "https://openrouter.ai/api/alpha/decisions"
    assert _openrouter_decisions_url("") == "https://openrouter.ai/api/alpha/decisions"
    monkeypatch.setenv("COMPASS_DECISION_ENDPOINT", "http://localhost:9999/decide")
    assert _openrouter_decisions_url("https://openrouter.ai/api/v1") == "http://localhost:9999/decide"


def test_client_posts_model_state_and_questions_to_openrouter(settings):
    session = FakeSession([_ok()])
    client = DecisionClient(settings=settings, session=session)
    with contextlib.redirect_stdout(io.StringIO()):
        response = client.ask(model="~typesafe/jev-latest", state={"clinical_record": "notes"}, questions=QUESTION)
    post = session.posts[0]
    assert post["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert post["body"] == {"model": "typesafe/jev-latest", "state": {"clinical_record": "notes"}, "questions": QUESTION}
    assert post["headers"]["Authorization"] == "Bearer test-openrouter-key"
    assert post["headers"]["Content-Type"] == "application/json"
    assert response.answers == {"q001": {"noul": 0.7}}
    assert response.usage.cost_usd == pytest.approx(0.00005)
    assert response.request_id == "gen-1"
    assert client.calls[0]["question_count"] == 1
    assert client.total_cost() == pytest.approx(0.00005)


def test_client_prices_input_tokens_when_the_provider_omits_cost(settings):
    client = DecisionClient(settings=settings, session=FakeSession([_ok(cost=None)]))
    with contextlib.redirect_stdout(io.StringIO()):
        response = client.ask(model=JEV, state={}, questions=QUESTION)
    assert response.usage.cost_usd == pytest.approx(1000 * 0.042 / 1e6)


def test_client_retries_transient_errors(settings):
    session = FakeSession([FakeResponse(429, text="rate limited"), requests.ConnectionError("reset"), _ok()])
    client = DecisionClient(settings=settings, session=session)
    with contextlib.redirect_stdout(io.StringIO()):
        response = client.ask(model=JEV, state={}, questions=QUESTION)
    assert len(session.posts) == 3
    assert response.answers["q001"]["noul"] == 0.7


def test_client_gives_up_after_max_retries(settings):
    settings.decision.max_retries = 2
    session = FakeSession([FakeResponse(503, text="busy"), FakeResponse(503, text="busy")])
    with pytest.raises(DecisionRequestError) as info:
        DecisionClient(settings=settings, session=session).ask(model=JEV, state={}, questions=QUESTION)
    assert info.value.status == 503
    assert len(session.posts) == 2


def test_client_raises_length_errors_without_retrying(settings):
    session = FakeSession([FakeResponse(400, text='{"error": "state exceeds the maximum context length"}')])
    with pytest.raises(DecisionRequestError) as info:
        DecisionClient(settings=settings, session=session).ask(model=JEV, state={}, questions=QUESTION)
    assert info.value.status == 400
    assert info.value.looks_like_length_error
    assert len(session.posts) == 1


def test_client_zero_retries_means_one_attempt(settings):
    settings.decision.max_retries = 0
    session = FakeSession([FakeResponse(503, text="busy"), _ok()])
    with pytest.raises(DecisionRequestError) as info:
        DecisionClient(settings=settings, session=session).ask(model=JEV, state={}, questions=QUESTION)
    assert info.value.status == 503
    assert len(session.posts) == 1


def test_client_unset_retries_default_to_three_attempts(settings):
    settings.decision.max_retries = None
    session = FakeSession([FakeResponse(503, text="busy")] * 3)
    with pytest.raises(DecisionRequestError):
        DecisionClient(settings=settings, session=session).ask(model=JEV, state={}, questions=QUESTION)
    assert len(session.posts) == 3


def test_client_retries_a_200_response_that_is_not_json(settings):
    session = FakeSession([FakeResponse(200, text="<html>gateway hiccup</html>", not_json=True), _ok()])
    with contextlib.redirect_stdout(io.StringIO()):
        response = DecisionClient(settings=settings, session=session).ask(model=JEV, state={}, questions=QUESTION)
    assert len(session.posts) == 2
    assert response.answers == {"q001": {"noul": 0.7}}


def test_client_gives_up_on_bodies_that_are_never_json(settings):
    settings.decision.max_retries = 2
    session = FakeSession([FakeResponse(200, text="<html>oops</html>", not_json=True) for _ in range(2)])
    with pytest.raises(DecisionRequestError, match="not JSON") as info:
        DecisionClient(settings=settings, session=session).ask(model=JEV, state={}, questions=QUESTION)
    assert info.value.status == 200
    assert "<html>oops</html>" in info.value.body
    assert not info.value.looks_like_length_error
    assert len(session.posts) == 2


def test_client_rejects_bad_requests_and_responses(settings):
    with pytest.raises(ValueError):
        DecisionClient(settings=settings, session=FakeSession([])).ask(model=JEV, state={}, questions={})
    bad = FakeSession([FakeResponse(200, {"usage": {}})])
    with pytest.raises(DecisionRequestError, match="no answers"):
        DecisionClient(settings=settings, session=bad).ask(model=JEV, state={}, questions=QUESTION)
    # A JSON body that is not an object has no answers map either (and is not retried).
    listed = FakeSession([FakeResponse(200, [{"q001": {"noul": 0.7}}]), _ok()])
    with pytest.raises(DecisionRequestError, match="no answers"):
        DecisionClient(settings=settings, session=listed).ask(model=JEV, state={}, questions=QUESTION)
    assert len(listed.posts) == 1


def test_client_requires_a_key_before_any_request(settings):
    settings.openrouter_api_key = ""
    session = FakeSession([_ok()])
    with pytest.raises(DecisionRequestError, match="OPENROUTER_API_KEY"):
        DecisionClient(settings=settings, session=session).ask(model=JEV, state={}, questions=QUESTION)
    assert session.posts == []


def test_client_typesafe_native_route(settings):
    settings.decision.provider = "typesafe"
    session = FakeSession([_ok()])
    with contextlib.redirect_stdout(io.StringIO()):
        DecisionClient(settings=settings, session=session).ask(model=JEV, state={}, questions=QUESTION)
    post = session.posts[0]
    assert post["url"] == "https://api.typesafe.ai/v1/systemone"
    assert post["body"]["model"] == "jev-1.13"
    assert post["headers"]["Authorization"] == "Bearer test-typesafe-key"
    assert "X-Title" not in post["headers"]

    settings.typesafe_api_key = ""
    with pytest.raises(DecisionRequestError, match="TYPESAFE_API_KEY"):
        DecisionClient(settings=settings, session=FakeSession([])).ask(model=JEV, state={}, questions=QUESTION)


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (400, "context length exceeded", True),
        (413, "payload too long", True),
        (400, "This model's maximum context length is 32000 tokens", True),
        (400, "prompt does not fit the context window", True),
        (413, "too many tokens in state", True),
        (422, "token limit reached", True),
        (400, "state exceeds the limit of the model", True),
        (400, "request exceeds the maximum size", True),
        (400, "CONTEXT LENGTH EXCEEDED", True),
        # Generic words alone are no longer a length error.
        (422, "state exceeds 32000 tokens", False),
        (400, "invalid token in the questions map", False),
        (400, "field length must be positive", False),
        (400, "maximum of 3 orders exceeded", False),
        (400, "unknown question type", False),
        (500, "context length exceeded", False),
        (None, "context length exceeded", False),
    ],
)
def test_length_error_detection(status, body, expected):
    assert DecisionRequestError("rejected", status=status, body=body).looks_like_length_error is expected
