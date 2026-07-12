import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.agents.predictor import Predictor
from src.full_stack.backend.data.models.prediction_result import ConfidenceLevel, NodePrediction, RegressionPrediction
from src.full_stack.backend.data.models.prediction_task import PredictionMode, PredictionTaskNode


def test_regression_output_normalized_key_mapping():
    predictor = Predictor.__new__(Predictor)
    node_spec = PredictionTaskNode(
        node_id="root",
        display_name="IQ",
        mode=PredictionMode.UNIVARIATE_REGRESSION,
        regression_outputs=["total_iq"],
    )
    payload = {
        "node_id": "root",
        "regression": {
            "values": {
                "Total IQ": 91.25,
            }
        },
        "confidence_level": "HIGH",
        "confidence_score": 0.8,
    }
    node = predictor._parse_node_prediction(node_payload=payload, node_spec=node_spec, path="root")
    assert node.regression is not None
    assert node.regression.values["total_iq"] == pytest.approx(91.25)


def test_univariate_uses_single_numeric_value_when_key_differs():
    predictor = Predictor.__new__(Predictor)
    node_spec = PredictionTaskNode(
        node_id="root",
        display_name="Mortality age",
        mode=PredictionMode.UNIVARIATE_REGRESSION,
        regression_outputs=["individual will die at what age?"],
    )
    payload = {
        "node_id": "root",
        "regression": {
            "values": {
                "predicted_age_years": 84,
            }
        },
        "confidence_level": "MEDIUM",
        "confidence_score": 0.7,
    }
    node = predictor._parse_node_prediction(node_payload=payload, node_spec=node_spec, path="root")
    assert node.regression is not None
    assert node.regression.values["individual will die at what age?"] == pytest.approx(84.0)


def test_multivariate_missing_output_raises():
    predictor = Predictor.__new__(Predictor)
    node_spec = PredictionTaskNode(
        node_id="root",
        display_name="Traits",
        mode=PredictionMode.MULTIVARIATE_REGRESSION,
        regression_outputs=["trait_a", "trait_b"],
    )
    payload = {
        "node_id": "root",
        "regression": {
            "values": {
                "trait_a": 0.2,
            }
        },
    }
    with pytest.raises(ValueError, match="Missing regression output"):
        predictor._parse_node_prediction(node_payload=payload, node_spec=node_spec, path="root")


def test_univariate_zero_can_be_recovered_from_age_narrative():
    predictor = Predictor.__new__(Predictor)
    root_prediction = NodePrediction(
        node_id="root",
        path="root",
        mode=PredictionMode.UNIVARIATE_REGRESSION,
        regression=RegressionPrediction(values={"individual will die at what age?": 0.0}),
        confidence_level=ConfidenceLevel.MEDIUM,
        confidence_score=0.5,
    )
    note = predictor._repair_univariate_zero_from_narrative(
        root_prediction=root_prediction,
        prediction_data={
            "clinical_summary": "The estimated mortality age is around 82 years with moderate uncertainty.",
            "reasoning_chain": [],
        },
    )
    assert note is not None
    assert root_prediction.regression is not None
    assert root_prediction.regression.values["individual will die at what age?"] == pytest.approx(82.0)
