import sys
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.agents.critic import Critic
from src.full_stack.backend.data.models.prediction_result import (
    BinaryClassification,
    ClassificationPrediction,
    ConfidenceLevel,
    KeyFinding,
    NodePrediction,
    PredictionResult,
    RegressionPrediction,
    Verdict,
)
from src.full_stack.backend.data.models.prediction_task import (
    PredictionMode,
    PredictionTaskNode,
    PredictionTaskSpec,
    build_binary_task_spec,
)


def _build_univariate_prediction() -> tuple[PredictionResult, PredictionTaskSpec]:
    spec = PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="root",
            display_name="IQ",
            mode=PredictionMode.UNIVARIATE_REGRESSION,
            regression_outputs=["total_iq"],
        )
    )
    root_prediction = NodePrediction(
        node_id="root",
        path="root",
        mode=PredictionMode.UNIVARIATE_REGRESSION,
        regression=RegressionPrediction(values={"total_iq": 91.0}),
        confidence_level=ConfidenceLevel.HIGH,
        confidence_score=0.82,
    )
    prediction = PredictionResult(
        prediction_id="pred_uni",
        participant_id="SUBJ_001",
        target_condition="IQ",
        control_condition="NON_TARGET_COMPARATOR",
        created_at=datetime.now(),
        prediction_task_spec=spec,
        root_prediction=root_prediction,
        confidence_level=ConfidenceLevel.HIGH,
        key_findings=[
            KeyFinding(
                domain="COGNITION",
                finding="Global IQ pattern is internally consistent.",
                direction="NORMAL",
                z_score=None,
                relevance_to_prediction="Supports stable univariate estimate.",
            )
        ],
        reasoning_chain=["Findings support total_iq estimate."],
        supporting_evidence={"for_target": ["coherent signal"], "against_target": []},
        uncertainty_factors=[],
        clinical_summary="Predicted total IQ is 91 with moderate confidence.",
        domains_processed=["COGNITION"],
        total_tokens_used=500,
        iteration=1,
    )
    return prediction, spec


def _build_binary_prediction() -> tuple[PredictionResult, PredictionTaskSpec]:
    spec = build_binary_task_spec(target_label="CASE", control_label="CONTROL")
    root_prediction = NodePrediction(
        node_id="root",
        path="root",
        mode=PredictionMode.BINARY_CLASSIFICATION,
        classification=ClassificationPrediction(
            predicted_label="CONTROL",
            probabilities={"CASE": 0.25, "CONTROL": 0.75},
        ),
        confidence_level=ConfidenceLevel.HIGH,
        confidence_score=0.75,
    )
    prediction = PredictionResult(
        prediction_id="pred_binary",
        participant_id="SUBJ_BIN",
        target_condition="CASE",
        control_condition="CONTROL",
        created_at=datetime.now(),
        prediction_task_spec=spec,
        root_prediction=root_prediction,
        binary_classification=BinaryClassification.CONTROL,
        probability_score=0.25,
        confidence_level=ConfidenceLevel.HIGH,
        key_findings=[
            KeyFinding(
                domain="BRAIN_MRI",
                finding="Pattern better matches comparator profile.",
                direction="NORMAL",
                z_score=None,
                relevance_to_prediction="Supports control-side classification.",
            )
        ],
        reasoning_chain=["Integrated findings support CONTROL over CASE."],
        supporting_evidence={"for_target": [], "against_target": ["Comparator fit is stronger."]},
        uncertainty_factors=[],
        clinical_summary="Binary classification favors CONTROL with adequate evidence.",
        domains_processed=["BRAIN_MRI"],
        total_tokens_used=450,
        iteration=1,
    )
    return prediction, spec


class _StubResponse:
    def __init__(self, content: str):
        self.content = content
        self.prompt_tokens = 0
        self.completion_tokens = 0


class _StubCriticClient:
    def call(self, **_kwargs):
        payload = {
            "verdict": "SATISFACTORY",
            "confidence_in_verdict": 0.9,
            "composite_score": 0.9,
            "concise_summary": "Binary output is consistent and evidence-grounded.",
            "score_breakdown": {
                "logic": 0.9,
                "evidence": 0.9,
                "completeness": 0.9,
                "relevance": 0.9,
            },
            "checklist": {
                "has_binary_outcome": True,
                "valid_probability": True,
                "sufficient_coverage": True,
                "evidence_based_reasoning": True,
                "clinically_relevant": True,
                "logically_coherent": True,
                "critical_domains_processed": True,
            },
            "improvement_suggestions": [],
            "reasoning": "Checks passed.",
        }
        return _StubResponse(json.dumps(payload))


class _MalformedSummaryClient:
    def call(self, **_kwargs):
        return _StubResponse('{"concise_summary":"truncated')


def test_univariate_checklist_is_task_specific():
    prediction, spec = _build_univariate_prediction()
    critic = Critic(llm_client=SimpleNamespace())

    evaluation = critic.execute(
        prediction=prediction,
        executor_output={"domains_processed": ["COGNITION"]},
        data_overview={"domain_coverage": {"COGNITION": {"present_leaves": 4}}},
        prediction_task_spec=spec,
    )

    assert evaluation.verdict == Verdict.SATISFACTORY
    assert "Estimated value: total_iq=91.000" in evaluation.concise_summary
    assert "regression_values_valid" in evaluation.checklist.active_checks
    assert "classification_probabilities_valid" not in evaluation.checklist.active_checks
    assert "hierarchy_consistent" not in evaluation.checklist.active_checks
    assert evaluation.checklist.total_count == 8
    assert evaluation.checklist.pass_count == 8


def test_malformed_optional_summary_falls_back_without_failing_evaluation():
    prediction, spec = _build_univariate_prediction()
    critic = Critic(llm_client=_MalformedSummaryClient())

    evaluation = critic.execute(
        prediction=prediction,
        executor_output={"domains_processed": ["COGNITION"]},
        data_overview={"domain_coverage": {"COGNITION": {"present_leaves": 4}}},
        prediction_task_spec=spec,
    )

    assert evaluation.verdict == Verdict.SATISFACTORY
    assert "Estimated value: total_iq=91.000" in evaluation.concise_summary


def test_binary_execute_uses_binary_relevant_active_checks_only():
    prediction, spec = _build_binary_prediction()
    critic = Critic(llm_client=_StubCriticClient())

    evaluation = critic.execute(
        prediction=prediction,
        executor_output={"domains_processed": ["BRAIN_MRI"]},
        data_overview={"domain_coverage": {"BRAIN_MRI": {"present_leaves": 4}}},
        prediction_task_spec=spec,
    )

    active = evaluation.checklist.active_checks
    assert "has_required_outputs" in active
    assert "output_schema_valid" in active
    assert "classification_probabilities_valid" in active
    assert "regression_values_valid" not in active
    assert "hierarchy_consistent" not in active
    assert evaluation.checklist.total_count == 8
    assert evaluation.checklist.pass_count == 8
    assert isinstance(evaluation.concise_summary, str)
    assert len(evaluation.concise_summary.strip()) > 20


def test_hierarchy_check_included_when_structure_present():
    spec = PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="root",
            display_name="Cognition",
            mode=PredictionMode.UNIVARIATE_REGRESSION,
            regression_outputs=["total_iq"],
            children=[
                PredictionTaskNode(
                    node_id="subfacet_profile",
                    display_name="Subfacets",
                    mode=PredictionMode.MULTIVARIATE_REGRESSION,
                    regression_outputs=["verbal", "working_memory"],
                )
            ],
        )
    )
    root_prediction = NodePrediction(
        node_id="root",
        path="root",
        mode=PredictionMode.UNIVARIATE_REGRESSION,
        regression=RegressionPrediction(values={"total_iq": 101.0}),
        confidence_level=ConfidenceLevel.MEDIUM,
        confidence_score=0.7,
        children=[
            NodePrediction(
                node_id="subfacet_profile",
                path="root.subfacet_profile",
                mode=PredictionMode.MULTIVARIATE_REGRESSION,
                regression=RegressionPrediction(values={"verbal": 104.0, "working_memory": 97.0}),
                confidence_level=ConfidenceLevel.MEDIUM,
                confidence_score=0.68,
            )
        ],
    )
    prediction = PredictionResult(
        prediction_id="pred_hier",
        participant_id="SUBJ_002",
        target_condition="Cognition",
        control_condition="NON_TARGET_COMPARATOR",
        created_at=datetime.now(),
        prediction_task_spec=spec,
        root_prediction=root_prediction,
        confidence_level=ConfidenceLevel.MEDIUM,
        key_findings=[
            KeyFinding(
                domain="COGNITION",
                finding="Root and subfacet outputs are coherent.",
                direction="NORMAL",
                z_score=None,
                relevance_to_prediction="Supports hierarchical consistency.",
            )
        ],
        reasoning_chain=["Hierarchy has all required nodes."],
        supporting_evidence={"for_target": ["full node coverage"], "against_target": []},
        uncertainty_factors=[],
        clinical_summary="Hierarchical cognition outputs are internally consistent.",
        domains_processed=["COGNITION"],
        total_tokens_used=700,
        iteration=1,
    )

    critic = Critic(llm_client=SimpleNamespace())
    evaluation = critic.execute(
        prediction=prediction,
        executor_output={"domains_processed": ["COGNITION"]},
        data_overview={"domain_coverage": {"COGNITION": {"present_leaves": 6}}},
        prediction_task_spec=spec,
    )

    assert "hierarchy_consistent" in evaluation.checklist.active_checks
    assert evaluation.checklist.hierarchy_consistent is True


def test_generalized_fallback_prediction_forces_unsatisfactory_verdict():
    prediction, spec = _build_univariate_prediction()
    prediction.prediction_id = "fallback_SUBJ001_deadbeef"
    prediction.uncertainty_factors = ["predictor_fallback_error:invalid_json"]
    prediction.clinical_summary = "Predictor response unavailable; deterministic fallback applied."

    critic = Critic(llm_client=SimpleNamespace())
    evaluation = critic.execute(
        prediction=prediction,
        executor_output={"domains_processed": ["COGNITION"]},
        data_overview={"domain_coverage": {"COGNITION": {"present_leaves": 4}}},
        prediction_task_spec=spec,
    )

    assert evaluation.verdict == Verdict.UNSATISFACTORY
    assert "fallback" in evaluation.reasoning.lower()


def test_univariate_template_zero_without_justification_is_unsatisfactory():
    prediction, spec = _build_univariate_prediction()
    if prediction.root_prediction and prediction.root_prediction.regression:
        prediction.root_prediction.regression.values["total_iq"] = 0.0
    prediction.clinical_summary = "Estimated total IQ is around 91 based on integrated evidence."
    prediction.reasoning_chain = ["Integrated findings support an IQ estimate near 91."]

    critic = Critic(llm_client=SimpleNamespace())
    evaluation = critic.execute(
        prediction=prediction,
        executor_output={"domains_processed": ["COGNITION"]},
        data_overview={"domain_coverage": {"COGNITION": {"present_leaves": 4}}},
        prediction_task_spec=spec,
    )

    assert evaluation.verdict == Verdict.UNSATISFACTORY
    assert evaluation.checklist.regression_values_valid is False
