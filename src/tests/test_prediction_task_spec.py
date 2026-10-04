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


def _leaf(node_id, display_name="Leaf"):
    return PredictionTaskNode(
        node_id=node_id,
        display_name=display_name,
        mode=PredictionMode.UNIVARIATE_REGRESSION,
        regression_outputs=["score"],
    )


def test_duplicate_node_id_between_root_and_child_is_rejected():
    with pytest.raises(ValueError, match="node_id 'dx' appears more than once"):
        PredictionTaskSpec(
            root=PredictionTaskNode(
                node_id="dx",
                display_name="Diagnosis",
                mode=PredictionMode.BINARY_CLASSIFICATION,
                class_labels=["case", "control"],
                children=[_leaf("dx")],
            )
        )


def test_duplicate_node_id_in_different_branches_is_rejected():
    with pytest.raises(ValueError, match="node_id 'total' appears more than once"):
        PredictionTaskSpec(
            root=PredictionTaskNode(
                node_id="root",
                display_name="Severity",
                mode=PredictionMode.MULTICLASS_CLASSIFICATION,
                class_labels=["none", "mild", "severe"],
                children=[
                    PredictionTaskNode(
                        node_id="madrs",
                        display_name="MADRS",
                        mode=PredictionMode.UNIVARIATE_REGRESSION,
                        regression_outputs=["madrs_total"],
                        children=[_leaf("total", "MADRS total")],
                    ),
                    PredictionTaskNode(
                        node_id="hamd",
                        display_name="HAM-D",
                        mode=PredictionMode.UNIVARIATE_REGRESSION,
                        regression_outputs=["hamd_total"],
                        children=[_leaf("total", "HAM-D total")],
                    ),
                ],
            )
        )


def test_unique_node_ids_and_shared_display_names_are_accepted():
    spec = PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="root",
            display_name="Scores",
            mode=PredictionMode.UNIVARIATE_REGRESSION,
            regression_outputs=["score"],
            children=[_leaf("a", "Same name"), _leaf("b", "Same name")],
        )
    )
    assert list(spec.node_index()) == ["root", "a", "b"]


@pytest.mark.parametrize(
    "labels",
    [
        ["MDD", "mdd"],
        ["Major depression", "major  depression"],
        ["none", "mild", " Mild "],
    ],
)
def test_class_labels_that_differ_only_by_case_or_spacing_are_rejected(labels):
    mode = PredictionMode.BINARY_CLASSIFICATION if len(labels) == 2 else PredictionMode.MULTICLASS_CLASSIFICATION
    with pytest.raises(ValueError, match="differ by more than letter case or spacing"):
        PredictionTaskNode(node_id="dx", display_name="Diagnosis", mode=mode, class_labels=labels)


def test_distinct_class_labels_are_accepted():
    node = PredictionTaskNode(
        node_id="dx",
        display_name="Diagnosis",
        mode=PredictionMode.MULTICLASS_CLASSIFICATION,
        class_labels=["MDD", "MDD in remission", "CONTROL"],
    )
    assert node.class_labels == ["MDD", "MDD in remission", "CONTROL"]
