import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.data.models.prediction_task import (
    PredictionMode,
    PredictionTaskNode,
    PredictionTaskSpec,
    build_task_spec_from_flat_args,
)


def test_valid_mixed_hierarchical_spec():
    spec = PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="root",
            display_name="Personality",
            mode=PredictionMode.MULTIVARIATE_REGRESSION,
            regression_outputs=["openness", "conscientiousness"],
            children=[
                PredictionTaskNode(
                    node_id="subtype",
                    display_name="Subtype",
                    mode=PredictionMode.MULTICLASS_CLASSIFICATION,
                    class_labels=["a", "b", "c"],
                    children=[],
                )
            ],
        )
    )
    assert spec.root.mode == PredictionMode.MULTIVARIATE_REGRESSION
    assert len(spec.node_index()) == 2


def test_invalid_multiclass_requires_three_labels():
    with pytest.raises(ValueError):
        PredictionTaskNode(
            node_id="x",
            display_name="Subtype",
            mode=PredictionMode.MULTICLASS_CLASSIFICATION,
            class_labels=["a", "b"],
        )


def test_invalid_multivariate_requires_two_outputs():
    with pytest.raises(ValueError):
        PredictionTaskNode(
            node_id="x",
            display_name="Traits",
            mode=PredictionMode.MULTIVARIATE_REGRESSION,
            regression_outputs=["trait_a"],
        )


def test_build_flat_multiclass_spec():
    spec = build_task_spec_from_flat_args(
        prediction_type="multiclass",
        target_label="Subtype",
        class_labels=["A", "B", "C"],
    )
    assert spec.root.mode == PredictionMode.MULTICLASS_CLASSIFICATION
    assert spec.root.class_labels == ["A", "B", "C"]


def test_build_flat_univariate_spec_requires_exactly_one_output():
    with pytest.raises(ValueError):
        build_task_spec_from_flat_args(
            prediction_type="regression_univariate",
            target_label="Score",
            regression_outputs=["a", "b"],
        )
