import sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.agents.critic import Critic
from src.full_stack.backend.data.models.prediction_result import (
    Verdict,
    PredictionResult,
    BinaryClassification,
    ConfidenceLevel,
    KeyFinding,
)
from src.full_stack.backend.tools.anomaly_narrative import AnomalyNarrativeBuilder


def test_anomaly_preanalysis_supports_nested_score_schema():
    tool = AnomalyNarrativeBuilder(llm_client=SimpleNamespace())
    deviation = {
        "BIOLOGICAL_ASSAY": {
            "proteomics": {
                "inflammation_markers": {"score": 1.6},
                "neurotrophic_factors": {"score": -2.3},
            }
        },
        "BRAIN_MRI": {
            "structural": {
                "hippocampus": {"score": -2.1},
            }
        },
    }
    analysis = tool._analyze_deviation(deviation)
    assert analysis["total_features"] == 3
    assert analysis["abnormal_features"] >= 2
    assert "BIOLOGICAL_ASSAY" in analysis["domains_summary"]
    assert "BRAIN_MRI" in analysis["domains_summary"]


def test_anomaly_process_output_rejects_false_no_data_claim():
    tool = AnomalyNarrativeBuilder(llm_client=SimpleNamespace())
    deviation = {
        "BRAIN_MRI": {
            "functional_connectivity": {
                "default_mode_network": {"score": 2.8}
            }
        }
    }
    output_data = {
        "integrated_narrative": "No multimodal data available across all domains (total features: 0)."
    }
    with pytest.raises(ValueError, match="data-inconsistent"):
        tool._process_output(output_data, {"hierarchical_deviation": deviation})


def test_critic_summary_fallback_uses_reasoning_when_missing():
    critic = Critic(llm_client=SimpleNamespace())
    evaluation_data = {
        "verdict": "SATISFACTORY",
        "composite_score": 0.92,
        "confidence_in_verdict": 0.9,
        "reasoning": "Prediction is coherent and supported by evidence.",
        "checklist": {
            "has_binary_outcome": True,
            "valid_probability": True,
            "sufficient_coverage": True,
            "evidence_based_reasoning": True,
            "clinically_relevant": True,
            "logically_coherent": True,
            "critical_domains_processed": True,
        },
    }
    evaluation = critic._parse_evaluation(evaluation_data, prediction_id="pred_x")
    assert evaluation.verdict == Verdict.SATISFACTORY
    assert evaluation.concise_summary != "No summary provided."
    assert "coherent and supported" in evaluation.concise_summary


def test_critic_binary_checklist_maps_to_active_generalized_checks():
    critic = Critic(llm_client=SimpleNamespace())
    evaluation_data = {
        "verdict": "SATISFACTORY",
        "composite_score": 0.88,
        "confidence_in_verdict": 0.84,
        "concise_summary": "Binary classification passed all required quality checks.",
        "checklist": {
            "has_binary_outcome": True,
            "valid_probability": True,
            "sufficient_coverage": True,
            "evidence_based_reasoning": True,
            "clinically_relevant": True,
            "logically_coherent": True,
            "critical_domains_processed": True,
        },
    }
    evaluation = critic._parse_evaluation(evaluation_data, prediction_id="pred_binary")
    active = evaluation.checklist.active_checks
    assert "has_required_outputs" in active
    assert "output_schema_valid" in active
    assert "classification_probabilities_valid" in active
    assert "regression_values_valid" not in active
    assert "hierarchy_consistent" not in active
    assert evaluation.checklist.output_schema_valid is True
    assert evaluation.checklist.classification_probabilities_valid is True
    assert evaluation.checklist.total_count == 8
    assert evaluation.checklist.pass_count == 8


def test_critic_execute_raises_when_llm_json_is_invalid(monkeypatch):
    critic = Critic(llm_client=SimpleNamespace())

    def _raise(*args, **kwargs):
        raise ValueError("No JSON found in LLM response")

    monkeypatch.setattr(critic, "_call_llm_raw", _raise)

    prediction = PredictionResult(
        prediction_id="pred_failsafe",
        participant_id="P1",
        target_condition="MDD",
        control_condition="brain-implicated pathology, but NOT psychiatric",
        created_at=datetime.now(),
        binary_classification=BinaryClassification.CASE,
        probability_score=0.8,
        confidence_level=ConfidenceLevel.MEDIUM,
        key_findings=[
            KeyFinding(
                domain="BIOLOGICAL_ASSAY",
                finding="BDNF low",
                direction="ABNORMAL_LOW",
                z_score=-2.1,
                relevance_to_prediction="supportive",
            )
        ],
        reasoning_chain=["reason"],
        supporting_evidence={"for_case": ["x"], "for_control": []},
        uncertainty_factors=[],
        clinical_summary="summary",
        domains_processed=["BIOLOGICAL_ASSAY"],
        total_tokens_used=1000,
        iteration=1,
    )

    with pytest.raises(ValueError, match="No JSON found"):
        critic.execute(
            prediction=prediction,
            executor_output={},
            data_overview={},
            hierarchical_deviation={},
            non_numerical_data="",
            control_condition=prediction.control_condition,
        )
