import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "full_stack"
        / "backend"
        / "utils"
        / "validation"
        / "with_annotated_dataset"
    ),
)

from core.io_utils import load_ground_truth
from core.annotation_contract import summarize_annotation_contract
from core.metrics import (
    compute_hierarchical_metrics,
    compute_multiclass_metrics,
    compute_regression_metrics,
)
from core.workflows import run_detailed_workflow, run_metrics_workflow


def test_multiclass_probability_diagnostics_and_topk():
    rows = [
        {
            "actual": "A",
            "predicted": "A",
            "probabilities": {"A": 0.70, "B": 0.20, "C": 0.10},
            "predicted_probability": None,
        },
        {
            "actual": "B",
            "predicted": "C",
            "probabilities": {"A": 0.05, "B": 0.40, "C": 0.55},
            "predicted_probability": None,
        },
        {
            "actual": "C",
            "predicted": "C",
            "probabilities": {"A": 0.10, "B": 0.15, "C": 0.75},
            "predicted_probability": None,
        },
    ]
    metrics = compute_multiclass_metrics(rows)
    diag = metrics.get("probability_diagnostics")
    assert isinstance(diag, dict)
    topk = diag.get("top_k_accuracy")
    assert isinstance(topk, dict)
    assert topk["top1"] is not None
    assert topk["top2"] is not None
    assert topk["top3"] is not None
    assert topk["top2"] >= topk["top1"]
    assert len(diag.get("confidence_correct") or []) > 0
    assert len(diag.get("confidence_incorrect") or []) > 0


def test_regression_residual_summaries_and_top_errors():
    rows = [
        {"eid": "01", "disorder": "G1", "actual_values": {"y": 10.0}, "predicted_values": {"y": 12.5}},
        {"eid": "02", "disorder": "G1", "actual_values": {"y": 5.0}, "predicted_values": {"y": 4.0}},
        {"eid": "03", "disorder": "G2", "actual_values": {"y": 8.0}, "predicted_values": {"y": 8.5}},
    ]
    metrics = compute_regression_metrics(rows, expected_outputs=["y"])
    per_output = metrics["per_output"]["y"]
    residual = per_output.get("residual_summary")
    assert isinstance(residual, dict)
    assert residual.get("p95_error") is not None
    assert isinstance(per_output.get("raw_residuals"), list)
    top_errors = metrics.get("largest_absolute_errors")
    assert isinstance(top_errors, list)
    assert len(top_errors) >= 1
    assert top_errors[0]["abs_error"] >= top_errors[-1]["abs_error"]


def test_hierarchical_coverage_summary_present():
    rows = [
        {
            "truth_nodes": {
                "root": {"mode": "multiclass_classification", "label": "A"},
                "node_reg": {"mode": "univariate_regression", "values": {"score": 1.0}},
            },
            "pred_nodes": {
                "root": {"mode": "multiclass_classification", "predicted_label": "A"},
                "node_reg": {"mode": "univariate_regression", "values": {"score": 1.2}},
            },
        },
        {
            "truth_nodes": {
                "root": {"mode": "multiclass_classification", "label": "A"},
                "node_reg": {"mode": "univariate_regression", "values": {"score": 0.8}},
            },
            "pred_nodes": {
                "root": {"mode": "multiclass_classification", "predicted_label": "B"},
                "node_reg": {"mode": "univariate_regression", "values": {"score": 0.7}},
            },
        },
    ]
    metrics = compute_hierarchical_metrics(rows)
    coverage = metrics.get("coverage_summary")
    assert isinstance(coverage, dict)
    assert coverage.get("macro_node_coverage_rate") is not None
    per_node = metrics.get("per_node")
    assert isinstance(per_node, dict)
    for node_id, block in per_node.items():
        assert "coverage" in block


def test_detailed_workflow_non_binary_emits_advanced_artifacts(tmp_path):
    results_dir = tmp_path / "participant_runs"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create three participant reports with one multiclass error.
    for eid, pred_label, probs in [
        ("11", "A", {"A": 0.7, "B": 0.2, "C": 0.1}),
        ("12", "C", {"A": 0.2, "B": 0.35, "C": 0.45}),
        ("13", "C", {"A": 0.1, "B": 0.2, "C": 0.7}),
    ]:
        pdir = results_dir / f"participant_ID{eid}"
        pdir.mkdir(parents=True, exist_ok=True)
        root = {
            "node_id": "root",
            "mode": "multiclass_classification",
            "classification": {"predicted_label": pred_label, "probabilities": probs},
            "confidence_score": max(probs.values()),
        }
        report = {
            "prediction": {
                "prediction_task_spec": {
                    "root": {"mode": "multiclass_classification", "class_labels": ["A", "B", "C"]}
                },
                "root_prediction": root,
                "flat_predictions": [root],
            },
            "evaluation": {"verdict": "SATISFACTORY", "composite_score": 0.85},
            "execution": {"iterations": 1, "duration_seconds": 10.0, "tokens_used": {"total": 1200}},
        }
        (pdir / f"report_{eid}.json").write_text(json.dumps(report, indent=2))

    annotations = {
        "11": {"label": "A", "disorder": "GROUP_X"},
        "12": {"label": "B", "disorder": "GROUP_X"},
        "13": {"label": "C", "disorder": "GROUP_Y"},
    }
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(annotations, indent=2))

    out_dir = tmp_path / "analysis"
    summary = run_detailed_workflow(
        results_dir=str(results_dir),
        output_dir=str(out_dir),
        prediction_type="multiclass",
        annotations_json=str(ann_path),
    )

    outputs = [Path(x).name for x in summary.get("outputs") or []]
    assert "detailed_annotation_contract_multiclass.json" in outputs
    assert "detailed_rows_multiclass.json" in outputs
    assert "detailed_multiclass_top_confusions.png" in outputs
    assert "detailed_multiclass_confidence_diagnostics.png" in outputs
    assert "detailed_multiclass_label_distribution.png" in outputs


def test_binary_ground_truth_json_format_supported(tmp_path):
    target_path = tmp_path / "binary_targets.json"
    target_path.write_text(
        json.dumps(
            {
                "01": {"label": "CASE", "disorder": "GROUP_A"},
                "02": {"label": "CONTROL", "disorder": "GROUP_A"},
            },
            indent=2,
        )
    )

    gt = load_ground_truth(str(target_path))
    assert gt["01"]["label"] == "CASE"
    assert gt["02"]["label"] == "CONTROL"
    assert gt["01"]["disorder"] == "GROUP_A"


def test_binary_ground_truth_legacy_txt_rejected(tmp_path):
    target_path = tmp_path / "binary_targets.txt"
    target_path.write_text("01|CASE (GROUP_A)\n02|CONTROL (GROUP_B)\n")

    with pytest.raises(ValueError) as exc:
        load_ground_truth(str(target_path))
    assert "must be valid JSON" in str(exc.value)


def test_binary_ground_truth_invalid_json_payload_rejected(tmp_path):
    target_path = tmp_path / "binary_targets.json"
    target_path.write_text(json.dumps({"annotations": [{"eid": "01", "disorder": "GROUP_A"}]}))

    with pytest.raises(ValueError) as exc:
        load_ground_truth(str(target_path))
    assert "no valid CASE/CONTROL rows" in str(exc.value)


def test_hierarchical_annotation_contract_requires_consistent_node_schema():
    rows = [
        {
            "eid": "01",
            "truth": {
                "disorder": "GROUP_A",
                "nodes": {
                    "root": {"mode": "multiclass_classification", "label": "profile_a"},
                    "trait_class": {"mode": "binary_classification", "label": "HIGH"},
                    "trait_reg": {"mode": "univariate_regression", "values": {"severity": 0.62}},
                    "facet_reg": {
                        "mode": "multivariate_regression",
                        "values": {"facet_1": 0.11, "facet_2": -0.23, "facet_3": 0.44},
                    },
                },
            },
        },
        {
            "eid": "02",
            "truth": {
                "disorder": "GROUP_B",
                "nodes": {
                    "root": {"mode": "multiclass_classification", "label": "profile_b"},
                    "trait_class": {"mode": "binary_classification", "label": "LOW"},
                    "trait_reg": {"mode": "univariate_regression", "values": {"severity": 0.21}},
                },
            },
        },
    ]
    contract = summarize_annotation_contract(rows=rows, prediction_type="hierarchical")
    assert contract["n_valid_rows"] == 1
    issues = contract.get("issue_counts") or {}
    assert int(issues.get("hierarchy_node_set_mismatch") or 0) == 1


def test_hierarchical_workflow_rejects_schema_mismatch(tmp_path):
    results_dir = tmp_path / "participant_runs"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build two predictions; evaluator should reject before scoring due to annotation mismatch.
    for eid, root_label, has_facet in [("01", "profile_a", True), ("02", "profile_b", False)]:
        pdir = results_dir / f"participant_ID{eid}"
        pdir.mkdir(parents=True, exist_ok=True)
        flat = [
            {
                "node_id": "root",
                "mode": "multiclass_classification",
                "classification": {
                    "predicted_label": root_label,
                    "probabilities": {"profile_a": 0.6, "profile_b": 0.4},
                },
            },
            {
                "node_id": "trait_reg",
                "mode": "univariate_regression",
                "regression": {"values": {"severity": 0.5}},
            },
        ]
        if has_facet:
            flat.append(
                {
                    "node_id": "facet_reg",
                    "mode": "multivariate_regression",
                    "regression": {"values": {"facet_1": 0.1, "facet_2": -0.1}},
                }
            )
        report = {
            "prediction": {
                "prediction_task_spec": {"root": {"mode": "multiclass_classification", "class_labels": ["profile_a", "profile_b"]}},
                "root_prediction": flat[0],
                "flat_predictions": flat,
            },
            "evaluation": {"verdict": "SATISFACTORY", "composite_score": 0.9},
            "execution": {"iterations": 1, "duration_seconds": 8.0, "tokens_used": {"total": 1400}},
        }
        (pdir / f"report_{eid}.json").write_text(json.dumps(report, indent=2))

    annotations = {
        "01": {
            "disorder": "GROUP_A",
            "nodes": {
                "root": {"mode": "multiclass_classification", "label": "profile_a"},
                "trait_reg": {"mode": "univariate_regression", "values": {"severity": 0.62}},
                "facet_reg": {"mode": "multivariate_regression", "values": {"facet_1": 0.11, "facet_2": -0.23}},
            },
        },
        "02": {
            "disorder": "GROUP_B",
            "nodes": {
                "root": {"mode": "multiclass_classification", "label": "profile_b"},
                "trait_reg": {"mode": "univariate_regression", "values": {"severity": 0.21}},
            },
        },
    }
    ann_path = tmp_path / "hier_annotations.json"
    ann_path.write_text(json.dumps(annotations, indent=2))

    out_dir = tmp_path / "analysis"
    with pytest.raises(ValueError) as exc:
        run_metrics_workflow(
            results_dir=str(results_dir),
            output_dir=str(out_dir),
            prediction_type="hierarchical",
            annotations_json=str(ann_path),
        )
    assert "Hierarchical annotation schema mismatch" in str(exc.value)
