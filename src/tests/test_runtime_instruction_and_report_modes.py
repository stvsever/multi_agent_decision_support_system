import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.agents.executor import Executor
import src.full_stack.backend.agents.integrator as integrator_mod
from src.full_stack.backend.data.models.execution_plan import PlanStep, ToolName
from src.full_stack.backend.data.models.prediction_result import (
    ConfidenceLevel,
    CriticEvaluation,
    EvaluationChecklist,
    KeyFinding,
    NodePrediction,
    PredictionResult,
    RegressionPrediction,
    Verdict,
)
from src.full_stack.backend.data.models.prediction_task import PredictionMode, PredictionTaskNode, PredictionTaskSpec
from src.full_stack.backend.utils.compass_logging.patient_report import PatientReportGenerator
from src.full_stack.backend.utils.core.plan_executor import PlanExecutor


def test_plan_executor_passes_runtime_instruction_fields():
    executor = PlanExecutor()
    step = PlanStep(
        step_id=1,
        tool_name=ToolName.FEATURE_SYNTHESIZER,
        description="Feature synthesis",
        expected_output="Narrative",
    )
    context = {
        "hierarchical_deviation": {"root": {}},
        "data_overview": {"domain_coverage": {}},
        "non_numerical_data": "",
        "target_condition": "personality",
        "control_condition": "NON_TARGET_COMPARATOR",
        "prediction_task_spec": {},
        "participant_id": "SUBJ_TEST",
        "tool_runtime_instruction": "Prioritize quantitative signal quality.",
        "executor_runtime_instruction": "Keep extraction conservative.",
    }

    tool_input = executor._build_tool_input(step, context, previous_outputs={})
    assert tool_input["tool_runtime_instruction"] == "Prioritize quantitative signal quality."
    assert tool_input["executor_runtime_instruction"] == "Keep extraction conservative."


def test_executor_context_combines_global_runtime_guidance_for_tools():
    executor = Executor()
    participant = SimpleNamespace(
        participant_id="SUBJ_TEST",
        hierarchical_deviation=SimpleNamespace(
            participant_id="SUBJ_TEST",
            domain_summaries={},
            root=None,
        ),
        non_numerical_data=SimpleNamespace(raw_text=""),
        multimodal_data=SimpleNamespace(features={}),
        data_overview=SimpleNamespace(
            participant_id="SUBJ_TEST",
            domain_coverage={},
            total_tokens=0,
            available_domains=[],
        ),
    )
    context = executor._build_context(
        participant_data=participant,
        target_condition="personality",
        control_condition="",
        prediction_task_spec=None,
        agent_instructions={
            "global": "Global instruction",
            "tools": "Tools instruction",
            "executor": "Executor instruction",
        },
    )
    assert context["tool_runtime_instruction"] == "Global instruction\n\nTools instruction"
    assert context["executor_runtime_instruction"] == "Global instruction\n\nExecutor instruction"


def test_patient_report_markdown_is_mode_aware_for_regression():
    spec = PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="root",
            display_name="Personality",
            mode=PredictionMode.MULTIVARIATE_REGRESSION,
            regression_outputs=["P1", "P2", "P3"],
        )
    )
    prediction = PredictionResult(
        prediction_id="pred_report_mode",
        participant_id="SUBJ_MODE",
        target_condition="personality",
        control_condition="NON_TARGET_COMPARATOR",
        created_at=datetime.now(),
        prediction_task_spec=spec,
        root_prediction=NodePrediction(
            node_id="root",
            mode=PredictionMode.MULTIVARIATE_REGRESSION,
            regression=RegressionPrediction(values={"P1": -0.3, "P2": -1.2, "P3": -1.1}),
            confidence_level=ConfidenceLevel.MEDIUM,
            confidence_score=0.7,
        ),
        confidence_level=ConfidenceLevel.MEDIUM,
        key_findings=[
            KeyFinding(
                domain="COGNITION",
                finding="Executive and social-affective patterns converge.",
                direction="NORMAL",
                z_score=None,
                relevance_to_prediction="Supports multivariate trait estimation.",
            )
        ],
        reasoning_chain=["Integrated domain signals and estimated trait profile outputs."],
        clinical_summary="Profile suggests low-to-moderate trait dysregulation.",
        domains_processed=["COGNITION", "BRAIN_MRI"],
        total_tokens_used=1000,
        iteration=1,
    )
    evaluation = CriticEvaluation(
        evaluation_id="eval_report_mode",
        prediction_id=prediction.prediction_id,
        verdict=Verdict.SATISFACTORY,
        confidence_in_verdict=0.8,
        composite_score=0.9,
        checklist=EvaluationChecklist(
            has_required_outputs=True,
            output_schema_valid=True,
            regression_values_valid=True,
            sufficient_coverage=True,
            evidence_based_reasoning=True,
            clinically_relevant=True,
            logically_coherent=True,
            critical_domains_processed=True,
            active_checks=[
                "has_required_outputs",
                "output_schema_valid",
                "regression_values_valid",
                "sufficient_coverage",
                "evidence_based_reasoning",
                "clinically_relevant",
                "logically_coherent",
                "critical_domains_processed",
            ],
        ),
        concise_summary="Multivariate regression evaluation passed.",
    )

    report = PatientReportGenerator().generate(
        participant_id="SUBJ_MODE",
        prediction=prediction,
        evaluation=evaluation,
        execution_summary={"domains_processed": ["COGNITION", "BRAIN_MRI"], "tokens_used": 1000},
        decision_trace=[],
    )
    markdown = PatientReportGenerator().to_markdown(report)

    assert "- **Prediction Type**: multivariate_regression" in markdown
    assert "- **Primary Output**:" in markdown
    assert "- **Classification**:" not in markdown


def test_patient_report_uses_hierarchical_type_and_root_confidence():
    spec = PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="phenotype_profile",
            display_name="Phenotype Profile",
            mode=PredictionMode.MULTIVARIATE_REGRESSION,
            regression_outputs=["P1", "P2"],
            children=[
                PredictionTaskNode(
                    node_id="subtype_axis",
                    display_name="Subtype Axis",
                    mode=PredictionMode.UNIVARIATE_REGRESSION,
                    regression_outputs=["sub_score"],
                    children=[],
                )
            ],
        )
    )
    prediction = PredictionResult(
        prediction_id="pred_hier_report",
        participant_id="SUBJ_HIER",
        target_condition="Phenotype Profile",
        control_condition="",
        created_at=datetime.now(),
        prediction_task_spec=spec,
        root_prediction=NodePrediction(
            node_id="phenotype_profile",
            path="phenotype_profile",
            mode=PredictionMode.MULTIVARIATE_REGRESSION,
            regression=RegressionPrediction(values={"P1": -0.3, "P2": 0.9}),
            confidence_level=ConfidenceLevel.HIGH,
            confidence_score=0.88,
            children=[
                NodePrediction(
                    node_id="subtype_axis",
                    path="phenotype_profile.subtype_axis",
                    mode=PredictionMode.UNIVARIATE_REGRESSION,
                    regression=RegressionPrediction(values={"sub_score": 0.4}),
                    confidence_level=ConfidenceLevel.MEDIUM,
                    confidence_score=0.75,
                    children=[],
                )
            ],
        ),
        confidence_level=ConfidenceLevel.HIGH,
        key_findings=[],
        reasoning_chain=["Hierarchical synthesis complete."],
        clinical_summary="Hierarchical phenotype output produced.",
        domains_processed=["BRAIN_MRI"],
        total_tokens_used=250,
        iteration=1,
    )
    evaluation = CriticEvaluation(
        evaluation_id="eval_hier_report",
        prediction_id=prediction.prediction_id,
        verdict=Verdict.SATISFACTORY,
        confidence_in_verdict=0.9,
        composite_score=0.95,
        checklist=EvaluationChecklist(
            has_required_outputs=True,
            output_schema_valid=True,
            regression_values_valid=True,
            hierarchy_consistent=True,
            sufficient_coverage=True,
            evidence_based_reasoning=True,
            clinically_relevant=True,
            logically_coherent=True,
            critical_domains_processed=True,
            active_checks=[
                "has_required_outputs",
                "output_schema_valid",
                "regression_values_valid",
                "hierarchy_consistent",
                "sufficient_coverage",
                "evidence_based_reasoning",
                "clinically_relevant",
                "logically_coherent",
                "critical_domains_processed",
            ],
        ),
        concise_summary="Hierarchical report formatting test.",
    )

    generator = PatientReportGenerator()
    report = generator.generate(
        participant_id="SUBJ_HIER",
        prediction=prediction,
        evaluation=evaluation,
        execution_summary={"domains_processed": ["BRAIN_MRI"], "tokens_used": 250},
        decision_trace=[],
    )
    markdown = generator.to_markdown(report)

    assert report["prediction"]["prediction_type"] == "hierarchical"
    assert report["prediction"]["root_mode"] == "multivariate_regression"
    assert report["prediction"]["probability"] == 0.88
    assert report["prediction"]["root_confidence"] == 0.88
    assert "| nodes: 2" in report["prediction"]["primary_output"]
    assert "- **Prediction Type**: hierarchical" in markdown
    assert "- **Probability / Root Confidence**: 88.0%" in markdown


def test_patient_report_markdown_infers_hierarchical_type_from_legacy_payload():
    legacy_report = {
        "participant_id": "SUBJ_LEGACY",
        "generated_at": "2026-02-24T00:00:00",
        "prediction": {
            "prediction_type": "multivariate_regression",
            "classification": None,
            "primary_output": "dimension_a: 1.700, dimension_b: 1.600",
            "probability": None,
            "confidence": "HIGH",
            "target_condition": "Phenotype Profile",
            "control_condition": "",
            "prediction_task_spec": {
                "schema_version": "1.0",
                "root": {
                    "node_id": "phenotype_profile",
                    "display_name": "Phenotype Profile",
                    "mode": "multivariate_regression",
                    "regression_outputs": ["dimension_a", "dimension_b"],
                    "children": [
                        {
                            "node_id": "risk_band",
                            "display_name": "Risk Band",
                            "mode": "multiclass_classification",
                            "class_labels": ["low", "medium", "high"],
                            "children": [],
                        }
                    ],
                },
            },
            "root_prediction": {
                "node_id": "phenotype_profile",
                "mode": "multivariate_regression",
                "confidence_score": 0.88,
                "children": [
                    {
                        "node_id": "risk_band",
                        "mode": "multiclass_classification",
                        "confidence_score": 0.8,
                        "children": [],
                    }
                ],
            },
        },
        "evaluation": {"verdict": "SATISFACTORY", "checklist_passed": 10, "checklist_total": 10},
        "key_findings": [],
        "reasoning": [],
        "clinical_summary": "summary",
        "execution": {"iterations": 1, "selected_iteration": 1, "tokens_used": 0, "domains_processed": []},
    }

    markdown = PatientReportGenerator().to_markdown(legacy_report)
    assert "- **Prediction Type**: hierarchical" in markdown
    assert "| nodes: 2" in markdown
    assert "- **Probability / Root Confidence**: 88.0%" in markdown


def test_integrator_chunk_extractor_receives_runtime_instructions(monkeypatch):
    captured = []

    class _FakeAssembler:
        def __init__(self, *args, **kwargs):
            pass

        def build_sections(self, executor_output, predictor_input, coverage_ledger):
            return [
                SimpleNamespace(
                    name="chunk_source_1",
                    text="Signal block A",
                    feature_keys=[],
                )
            ]

        def build_chunks(self, sections):
            return [sections]

        def chunk_to_text(self, sections, chunk_index, chunk_total):
            return "chunk_text_payload"

    class _FakeTool:
        def execute(self, payload):
            captured.append(payload)
            return SimpleNamespace(
                success=True,
                error=None,
                output={
                    "summary": "ok",
                    "for_case": [],
                    "for_control": [],
                    "evidence_for_targets": {},
                    "evidence_against_targets": {},
                    "uncertainty_factors": [],
                    "key_findings": [],
                    "cited_feature_keys": [],
                },
            )

    monkeypatch.setattr(integrator_mod, "PredictorInputAssembler", _FakeAssembler)
    monkeypatch.setattr(integrator_mod, "get_tool", lambda _name: _FakeTool())

    integrator = integrator_mod.Integrator()
    result = integrator.extract_chunk_evidence(
        step_outputs={},
        predictor_input={
            "context_fill_report": {
                "predictor_payload_estimate": {
                    "threshold": 1,
                    "final_tokens": 100,
                }
            }
        },
        coverage_ledger={},
        data_overview={},
        hierarchical_deviation={},
        non_numerical_data="",
        target_condition="DEPRESSION",
        control_condition="HEALTHY",
        prediction_task_spec={},
        iteration=1,
        tool_runtime_instruction="Global guidance\n\nTools guidance",
        executor_runtime_instruction="Global guidance\n\nExecutor guidance",
    )

    assert result["chunking_skipped"] is False
    assert len(captured) == 1
    assert captured[0]["tool_runtime_instruction"] == "Global guidance\n\nTools guidance"
    assert captured[0]["executor_runtime_instruction"] == "Global guidance\n\nExecutor guidance"
