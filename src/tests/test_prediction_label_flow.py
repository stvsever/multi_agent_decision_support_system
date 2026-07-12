from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from main import (
    _align_binary_task_spec_labels,
    _prediction_primary_label,
    _resolve_prediction_task_spec,
    _task_spec_to_legacy_labels,
)
from src.full_stack.backend.data.models.prediction_result import (
    ClassificationPrediction,
    ConfidenceLevel,
    NodePrediction,
    PredictionResult,
)
from src.full_stack.backend.data.models.prediction_task import build_binary_task_spec


def test_binary_task_spec_alignment_overrides_stale_case_control_payload() -> None:
    stale_payload = {
        "schema_version": "1.0",
        "root": {
            "node_id": "root",
            "display_name": "CASE",
            "mode": "binary_classification",
            "class_labels": ["CASE", "CONTROL"],
            "regression_outputs": [],
            "required": True,
            "children": [],
        },
    }
    spec = _resolve_prediction_task_spec(
        prediction_type="binary",
        target_label="DEPRESSION",
        control_label="HEALTHY",
        class_labels=[],
        regression_outputs=[],
        task_spec_payload=stale_payload,
        task_spec_json="",
        task_spec_file="",
    )
    aligned = _align_binary_task_spec_labels(
        spec,
        target_label="DEPRESSION",
        control_label="HEALTHY",
    )

    assert aligned.root.class_labels == ["DEPRESSION", "HEALTHY"]
    assert _task_spec_to_legacy_labels(aligned) == ("DEPRESSION", "HEALTHY")


def test_prediction_primary_label_prefers_runtime_root_label() -> None:
    spec = build_binary_task_spec(target_label="DEPRESSION", control_label="HEALTHY")
    root_prediction = NodePrediction(
        node_id="root",
        path="root",
        mode=spec.root.mode,
        classification=ClassificationPrediction(
            predicted_label="HEALTHY",
            probabilities={"DEPRESSION": 0.2, "HEALTHY": 0.8},
        ),
        confidence_level=ConfidenceLevel.HIGH,
        confidence_score=0.8,
    )
    prediction = PredictionResult(
        prediction_id="pred_1",
        participant_id="SUBJ_001",
        target_condition="DEPRESSION",
        control_condition="HEALTHY",
        prediction_task_spec=spec,
        root_prediction=root_prediction,
        confidence_level=ConfidenceLevel.HIGH,
    )

    assert _prediction_primary_label(prediction) == "HEALTHY"


def test_binary_alias_and_probability_follow_task_spec_labels() -> None:
    spec = build_binary_task_spec(target_label="DEPRESSION", control_label="HEALTHY")
    root_prediction = NodePrediction(
        node_id="root",
        path="root",
        mode=spec.root.mode,
        classification=ClassificationPrediction(
            predicted_label="HEALTHY",
            probabilities={"DEPRESSION": 0.2, "HEALTHY": 0.8},
        ),
        confidence_level=ConfidenceLevel.MEDIUM,
        confidence_score=0.8,
    )
    prediction = PredictionResult(
        prediction_id="pred_2",
        participant_id="SUBJ_001",
        target_condition="DEPRESSION",
        control_condition="HEALTHY",
        prediction_task_spec=spec,
        root_prediction=root_prediction,
        confidence_level=ConfidenceLevel.MEDIUM,
    )

    assert prediction.binary_classification is not None
    assert prediction.binary_classification.value == "CONTROL"
    assert prediction.probability_score == 0.2
