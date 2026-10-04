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
from src.full_stack.backend.decision.aggregate import (
    aggregate_classification,
    aggregate_regression_output,
    build_node_prediction,
    coarse_ensemble,
    confidence_level,
    refinement_window,
    score_distribution,
)
from src.full_stack.backend.decision.client import (
    DecisionClient,
    DecisionRequestError,
    DecisionResponse,
    DecisionUsage,
    _openrouter_decisions_url,
)
from src.full_stack.backend.decision.predictor import DecisionPredictor
from src.full_stack.backend.decision.quality import evaluate_decision_prediction
from src.full_stack.backend.decision.questions import (
    QuestionSet,
    add_classification_questions,
    add_coarse_regression_questions,
    add_fine_regression_questions,
    cache_dir,
    deterministic_book,
    round_one_questions,
)
from src.full_stack.backend.decision.scales import OutputScale, combine_density, density_summary
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


def test_classification_orders_are_averaged_and_tv_is_the_stability():
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
    # Each order is 0.1 total variation away from the mean.
    assert agg.stability == pytest.approx(0.1)
    assert agg.confidence == pytest.approx(0.7)
    assert [v["variant"] for v in agg.variants] == ["order1", "order2", "order3"]
    assert agg.variants[0]["choice"] == "A"


def test_classification_answers_are_normalized_and_missing_orders_skipped():
    node = _multiclass_node()
    qs = _node_questions(node)
    answers = {qs.items[1].qid: {"probabilities": {"A": 2.0, "B": 1.0, "C": 1.0, "unknown": 9.0}}}
    agg = aggregate_classification(node, qs.items, answers)
    assert agg.probabilities == pytest.approx({"A": 0.5, "B": 0.25, "C": 0.25})
    assert len(agg.variants) == 1
    assert agg.stability == pytest.approx(0.0)
    # Without a reported confidence the margin of the winning label stands in.
    assert agg.confidence == pytest.approx(0.0)


def test_classification_tie_goes_to_the_first_label_and_no_answers_raise():
    node = _multiclass_node()
    qs = _node_questions(node, choice_orders=1)
    tie = {qs.items[0].qid: {"probabilities": {"A": 1.0, "B": 1.0, "C": 1.0}}}
    assert aggregate_classification(node, qs.items, tie).predicted_label == "A"
    empty = {qs.items[0].qid: {"probabilities": {}}}
    assert aggregate_classification(node, qs.items, empty).probabilities == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    with pytest.raises(ValueError):
        aggregate_classification(node, qs.items, {})


def test_binary_noul_maps_onto_the_first_label():
    node = _binary_node()
    qs = _node_questions(node)
    noul, order1, order2 = qs.items
    assert noul.kind == "binary_noul" and order2.order == ["CONTROL", "MDD"]
    answers = {
        noul.qid: {"noul": 0.8},
        order1.qid: {"probabilities": {"MDD": 0.7, "CONTROL": 0.3}, "confidence": 0.5},
        order2.qid: {"probabilities": {"CONTROL": 0.4, "MDD": 0.6}, "confidence": 0.5},
    }
    agg = aggregate_classification(node, qs.items, answers)
    assert agg.probabilities == pytest.approx({"MDD": 0.7, "CONTROL": 0.3})
    assert agg.variants[0]["probabilities"] == pytest.approx({"MDD": 0.8, "CONTROL": 0.2})
    assert agg.stability == pytest.approx((0.1 + 0.0 + 0.1) / 3)
    # The Noul's confidence is its margin |2p - 1|.
    assert agg.confidence == pytest.approx((0.6 + 0.5 + 0.5) / 3)
    assert agg.predicted_label == "MDD"


def test_binary_noul_follows_label_order_and_is_clamped():
    node = _binary_node(labels=("CONTROL", "MDD"))
    qs = _node_questions(node)
    noul = qs.items[0]
    agg = aggregate_classification(node, [noul], {noul.qid: {"noul": 0.8}})
    assert agg.probabilities == pytest.approx({"CONTROL": 0.8, "MDD": 0.2})
    clamped = aggregate_classification(node, [noul], {noul.qid: {"noul": 1.4}})
    assert clamped.probabilities == pytest.approx({"CONTROL": 1.0, "MDD": 0.0})
    with pytest.raises(ValueError):
        aggregate_classification(node, [noul], {noul.qid: {"noul": "not a number"}})


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
    assert score_distribution(desc, {}) == pytest.approx([0.1] * 10)


def test_regression_aggregate_combines_orders_and_refinement():
    node = _madrs_node()
    scale = _madrs_scale()
    asc, desc = _node_questions(node).items
    # Ascending puts all mass on bin 2 (12 to 18), descending on bin 4 (24 to 30).
    answers = {asc.qid: {"probabilities": {"2": 1.0}}, desc.qid: {"probabilities": {"5": 1.0}}}
    bins, coarse, runs, _conf = coarse_ensemble([asc, desc], answers)
    assert coarse[2] == pytest.approx(0.5) and coarse[4] == pytest.approx(0.5)
    assert [r["mean"] for r in runs] == pytest.approx([15.0, 27.0])
    window = refinement_window(scale, bins, coarse)
    assert window == (2, 5)

    fine_qs = QuestionSet(prefix="r")
    fine_bins = scale.fine_grid(bins[2].lo, bins[4].hi, 10)
    add_fine_regression_questions(fine_qs, node, "madrs_total", scale, fine_bins, bins[2].lo, bins[4].hi)
    fine_asc, fine_desc = fine_qs.items
    answers[fine_asc.qid] = {"probabilities": {"4": 1.0}, "confidence": 0.9}
    answers[fine_desc.qid] = {"probabilities": {"5": 1.0}, "confidence": 0.7}

    agg = aggregate_regression_output(scale, [asc, desc], fine_qs.items, answers, window)
    expected = density_summary(
        combine_density(bins, coarse, window=window, fine_bins=fine_bins, fine_probs=[0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0])
    )
    # Fine level 4 is 19.2 to 21.0, so the estimate is its centre.
    assert agg.value == pytest.approx(20.1)
    assert agg.value == pytest.approx(expected["mean"])
    assert agg.summary == pytest.approx(expected)
    assert agg.value == agg.summary["mean"]
    assert agg.stability == pytest.approx(abs(15.0 - 27.0) / 60.0)
    assert agg.window == (12.0, 30.0)
    assert agg.confidence == pytest.approx(0.8)
    assert agg.detail["refined"] is True
    assert agg.detail["window"] == [12.0, 30.0]
    assert len(agg.detail["fine_levels"]) == 10
    assert [r["variant"] for r in agg.detail["coarse_runs"]] == ["ascending", "descending"]


def test_regression_aggregate_without_refinement_uses_the_coarse_density():
    asc, desc = _node_questions(_madrs_node()).items
    answers = {asc.qid: {"probabilities": {"2": 1.0}}, desc.qid: {"probabilities": {"5": 1.0}}}
    agg = aggregate_regression_output(_madrs_scale(), [asc, desc], [], answers, None)
    assert agg.value == pytest.approx(0.5 * 15.0 + 0.5 * 27.0)
    assert agg.fine is None and agg.window is None
    assert agg.detail["refined"] is False
    assert agg.confidence == 0.5
    with pytest.raises(ValueError):
        aggregate_regression_output(_madrs_scale(), [asc, desc], [], {}, None)


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
    assert agg.stability > 0.2
    pred = build_node_prediction(node, path="severity", classification=agg, regression={}, children=[], sufficiency=0.3)
    assert pred.classification is not None and pred.regression is None
    assert pred.confidence_level == ConfidenceLevel.LOW
    assert pred.decision_details["source"] == "decision_model"
    assert pred.decision_details["stability_tv"] == pytest.approx(agg.stability)
    assert pred.decision_details["evidence_sufficiency"] == 0.3
    assert any("evidence insufficient" in f for f in pred.uncertainty_factors)
    assert any("presentation orders" in f for f in pred.uncertainty_factors)


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
    assert madrs_detail["window"] == [12.0, 30.0]

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
    assert len(first["questions"]) == 13
    assert all(qid.startswith("q") for qid in first["questions"])
    assert sorted(second["questions"]) == ["r001", "r002"]
    assert all("(madrs_total)" in q["instructions"] for q in second["questions"].values())
    assert {"task_context", "clinical_record", "measurements", "deviation_profile"} <= set(first["state"])
    assert first["state"]["task_context"] == CONTEXT
    assert second["state"] == first["state"]

    report = result.decision_report
    for key in ("model", "model_label", "route", "question_book", "questions", "state", "requests", "cost_usd", "input_tokens", "quality", "settings"):
        assert key in report
    assert report["model"] == JEV and report["route"] == "direct"
    assert set(report["question_book"]["nodes"]) == {"severity", "madrs", "items", "response"}
    asked = report["questions"]
    assert len(asked) == 15
    fine = [q for q in asked if q["kind"] == "score_fine"]
    assert [(q["node_id"], q["output"]) for q in fine] == [("madrs", "madrs_total"), ("madrs", "madrs_total")]
    assert [r["round"] for r in report["requests"]] == ["coarse", "refine"]
    assert report["input_tokens"] == 2 * 1200 == result.total_tokens_used
    assert report["cost_usd"] == pytest.approx(2 * 0.0000504)
    assert report["state"]["features_total"] == len(direct_output["coverage_ledger"]["all_features"])
    assert report["state"]["feature_coverage"] == 1.0
    quality = report["quality"]
    assert set(quality) == {"evidence_sufficiency", "mean_instability", "max_instability", "mean_confidence", "feature_coverage", "per_node"}
    assert quality["evidence_sufficiency"] == pytest.approx(0.8)
    assert quality["max_instability"] < 1e-9
    assert quality["feature_coverage"] == 1.0
    assert set(quality["per_node"]) == {"severity", "madrs", "items", "response"}
    assert report["settings"] == {"choice_orders": 3, "score_levels": 10, "regression_refine": True, "tokenizer_ratio": 1.2}

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
    # Classification only: one request (Noul, two Choice orders, sufficiency).
    assert len(oracle.calls) == 1
    assert len(oracle.calls[0]["questions"]) == 4

    flipped = OracleClient(label_weights={"MDD": 0.3, "CONTROL": 0.7})
    control = _run(_predictor(flipped), direct_output, spec, target="MDD", control="CONTROL")
    assert control.binary_classification == BinaryClassification.CONTROL
    assert control.probability_score == pytest.approx(0.3)


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
    assert _tokens(rejected) <= budget
    assert _tokens(repacked) <= int(budget * 0.8) < _tokens(rejected)
    assert refine == repacked
    report = result.decision_report
    assert report["state"]["budget_tokens"] == int(budget * 0.8)
    assert [r["round"] for r in report["requests"]] == ["coarse", "refine"]
    assert report["state"]["feature_coverage"] < 1.0
    included = report["state"]["features_included"]
    # The state report rounds to 4 decimals; the quality block keeps full precision.
    assert report["quality"]["feature_coverage"] == pytest.approx(included / report["state"]["features_total"])
    assert report["state"]["feature_coverage"] == pytest.approx(report["quality"]["feature_coverage"], abs=1e-4)
    assert any(f"State packing kept {included} of" in factor for factor in result.uncertainty_factors)
    # Coverage below 0.9 makes the decision critic reject the attempt.
    assert evaluate_decision_prediction(result, spec).verdict == Verdict.UNSATISFACTORY


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


def test_question_book_is_compiled_once_by_the_companion_llm(settings):
    compiled = {
        "nodes": {
            "severity": {"question": "How severe is the depression?", "label_definitions": {"none": "No symptoms.", "mild": "Mild.", "severe": "Severe."}},
            "madrs": {"question": "What is the MADRS total?"},
            "items": {"question": "What are the item scores?"},
            "response": {"question": "Will the participant respond?", "label_definitions": {"responder": "50 percent reduction."}},
        }
    }

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
    assert seen == ["stub/orchestrator-llm"]
    assert book.compiled_by == "stub/orchestrator-llm"


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
    q = {
        "evidence_sufficiency": 0.8,
        "mean_instability": 0.02,
        "max_instability": 0.05,
        "mean_confidence": 0.7,
        "feature_coverage": 1.0,
        "per_node": {"severity": {"stability": 0.05, "confidence": 0.7}, "madrs": {"stability": 0.0, "confidence": 0.7}},
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


def test_decision_critic_threshold_is_inclusive():
    spec = _critic_spec()
    evaluation = evaluate_decision_prediction(_decision_prediction(spec, max_instability=0.20), spec, stability_threshold=0.20)
    assert evaluation.verdict == Verdict.SATISFACTORY


def test_decision_critic_rejects_order_instability():
    spec = _critic_spec()
    prediction = _decision_prediction(
        spec,
        max_instability=0.35,
        mean_instability=0.2,
        per_node={"severity": {"stability": 0.35, "confidence": 0.6}, "madrs": {"stability": 0.05, "confidence": 0.6}},
    )
    evaluation = evaluate_decision_prediction(prediction, spec)
    assert evaluation.verdict == Verdict.UNSATISFACTORY
    assert evaluation.checklist.decision_stable is False
    assert any("for severity" in w and "madrs" not in w for w in evaluation.weaknesses)
    assert [s.issue for s in evaluation.improvement_suggestions] == ["Order-sensitive decision"]
    assert "multi-agent workflow" in evaluation.concise_summary

    orchestrated = _decision_prediction(spec, route="orchestrated", max_instability=0.35)
    assert "re-orchestrates" in evaluate_decision_prediction(orchestrated, spec).concise_summary


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


def test_critic_agent_dispatches_decision_predictions(settings):
    settings.decision.stability_threshold = 0.04
    spec = _critic_spec()
    critic = Critic(llm_client=SimpleNamespace())  # any LLM call would fail
    with contextlib.redirect_stdout(io.StringIO()):
        rejected = critic.execute(
            prediction=_decision_prediction(spec, max_instability=0.05),
            executor_output={},
            data_overview={},
            prediction_task_spec=spec,
        )
        accepted = critic.execute(
            prediction=_decision_prediction(spec, max_instability=0.03),
            executor_output={},
            data_overview={},
            prediction_task_spec=spec,
        )
    assert rejected.verdict == Verdict.UNSATISFACTORY
    assert rejected.reasoning.startswith("Decision critic on the direct route")
    assert "threshold 0.04" in rejected.reasoning
    assert accepted.verdict == Verdict.SATISFACTORY


# ---------------------------------------------------------------------------
# HTTP transport against a fake session
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = headers or {}

    def json(self):
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


def test_client_rejects_bad_requests_and_responses(settings):
    with pytest.raises(ValueError):
        DecisionClient(settings=settings, session=FakeSession([])).ask(model=JEV, state={}, questions={})
    bad = FakeSession([FakeResponse(200, {"usage": {}})])
    with pytest.raises(DecisionRequestError, match="no answers"):
        DecisionClient(settings=settings, session=bad).ask(model=JEV, state={}, questions=QUESTION)


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
        (422, "state exceeds 32000 tokens", True),
        (400, "unknown question type", False),
        (500, "context length exceeded", False),
        (None, "context length exceeded", False),
    ],
)
def test_length_error_detection(status, body, expected):
    assert DecisionRequestError("rejected", status=status, body=body).looks_like_length_error is expected
