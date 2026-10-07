"""Records whose leaves hold preprocessed values without deviation scores.

Many datasets have no reference sample (no healthy controls, no normative model),
so a record may carry native-scale values only, or deviation scores for some
leaves and native values for the rest. Such a record may omit its deviation map;
the loader then derives the hierarchy from the leaves.
"""

import json
from pathlib import Path

import pytest

from src.full_stack.backend.agents.decision.state import build_sections
from src.full_stack.backend.utils.core.data_loader import DataLoader
from src.full_stack.backend.utils.core.record_rendering import (
    describe_value_representation,
    render_data_overview,
    render_deviation_profile,
    render_direct_record,
    value_representation,
)
from src.full_stack.backend.utils.validation import validate_participant_files

NATIVE = {
    "BRAIN": {
        "Morphometry": {
            "Hippocampus": {
                "_leaves": [
                    {"feature": "Left hippocampal volume", "value": 3.12, "unit": "cm3"},
                    {"feature": "Right hippocampal volume", "value": 3.30, "unit": "cm3"},
                ]
            },
            "Cortical thickness": {
                "_leaves": [{"feature": "Mean cortical thickness", "value": "2.41 mm"}],
            },
        }
    },
    "COGNITION": {
        "_leaves": [
            {"feature": "MoCA total", "value": 24, "unit": "points", "ref_range": "26 to 30"},
            {"feature": "Handedness", "value": "Right"},
        ]
    },
}

MIXED = {
    "BRAIN": {
        "Morphometry": {
            "_leaves": [
                {"feature": "Left hippocampal volume", "value": 3.12, "unit": "cm3", "z_score": -2.0},
                {"feature": "Right hippocampal volume", "value": 3.30, "unit": "cm3", "z_score": -1.0},
            ]
        }
    },
    "COGNITION": {"_leaves": [{"feature": "MoCA total", "value": 24, "unit": "points"}]},
}


def _write_record(folder: Path, multimodal: dict, *, with_map: bool = False) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    coverage = {
        domain: {"present_leaves": 2, "total_leaves": 2, "coverage_percentage": 100.0, "missing_count": 0}
        for domain in multimodal
    }
    (folder / "data_overview.json").write_text(json.dumps({"participant_id": folder.name, "domain_coverage": coverage}))
    (folder / "multimodal_data.json").write_text(json.dumps(multimodal))
    (folder / "non_numerical_data.txt").write_text("No reference sample was available for this study.")
    if with_map:
        (folder / "hierarchical_deviation_map.json").write_text(json.dumps({"BRAIN": {"_stats": {"mean_abs_score": 1.5}}}))
    return folder


def test_the_deviation_map_is_optional_but_the_measurements_are_not(tmp_path):
    folder = _write_record(tmp_path / "native", NATIVE)
    valid, errors, files = validate_participant_files(folder)
    assert valid and errors == []
    assert "hierarchical_deviation" not in files

    (folder / "multimodal_data.json").unlink()
    valid, errors, _ = validate_participant_files(folder)
    assert not valid and any("multimodal_data.json" in e for e in errors)


def test_a_native_record_loads_with_a_derived_hierarchy_and_no_scores(tmp_path):
    data = DataLoader().load(_write_record(tmp_path / "native", NATIVE))
    rep = data.data_overview.value_representation
    assert rep["mode"] == "native"
    assert rep["leaves_with_deviation"] == 0
    assert rep["numeric_leaves_without_deviation"] == 4  # "2.41 mm" counts as numeric
    assert rep["categorical_leaves"] == 1
    assert "never read a value as a z-score" in rep["note"]

    domains = {node.node_name: node for node in data.hierarchical_deviation.root.children}
    assert set(domains) == {"BRAIN", "COGNITION"}
    assert [c.node_name for c in domains["BRAIN"].children] == ["Morphometry"]
    assert {c.node_name for c in domains["BRAIN"].children[0].children} == {"Hippocampus", "Cortical thickness"}
    assert render_deviation_profile(data.hierarchical_deviation) == ""

    # Leaf values stay on their native scale: no z is invented.
    leaves = [f for feats in data.multimodal_data.features.values() for f in feats]
    assert all(f.z_score is None for f in leaves)
    assert {f.value for f in leaves} >= {3.12, 24, "2.41 mm", "Right"}


def test_a_mixed_record_scores_groups_from_its_deviation_scored_leaves_only(tmp_path):
    data = DataLoader().load(_write_record(tmp_path / "mixed", MIXED))
    rep = data.data_overview.value_representation
    assert rep["mode"] == "mixed"
    assert rep["note"].startswith("2 of 3 numeric leaves carry a deviation score")

    brain = next(n for n in data.hierarchical_deviation.root.children if n.node_name == "BRAIN")
    cognition = next(n for n in data.hierarchical_deviation.root.children if n.node_name == "COGNITION")
    assert brain.z_score == pytest.approx(1.5) and brain.score_kind == "mean_abs"
    assert cognition.z_score is None
    assert data.hierarchical_deviation.domain_summaries["COGNITION"]["mean_abs_score"] is None


def test_a_record_with_a_deviation_map_still_uses_it(tmp_path):
    data = DataLoader().load(_write_record(tmp_path / "with_map", MIXED, with_map=True))
    brain = next(n for n in data.hierarchical_deviation.root.children if n.node_name == "BRAIN")
    assert brain.z_score == pytest.approx(1.5)


def test_the_rendering_states_values_without_inventing_deviations(tmp_path):
    data = DataLoader().load(_write_record(tmp_path / "native", NATIVE))
    multimodal = {d: [f.model_dump() for f in feats] for d, feats in data.multimodal_data.features.items()}
    record = render_direct_record(
        multimodal=multimodal,
        deviation=data.hierarchical_deviation,
        overview=data.data_overview.model_dump(),
        notes=data.non_numerical_data.raw_text,
    )
    assert "(z " not in record["measurements"]
    assert "- Left hippocampal volume = 3.12 cm3" in record["measurements"]
    assert "- MoCA total = 24 points [ref range 26 to 30]" in record["measurements"]
    assert record["deviation_profile"] == ""
    first = record["data_overview"].splitlines()[0]
    assert first.startswith("Values: No leaf carries a deviation score")


def test_the_value_representation_covers_every_case():
    assert value_representation({})["mode"] == "empty"
    assert value_representation({"A": {"_leaves": [{"feature": "Sex", "value": "Female"}]}})["mode"] == "categorical"
    assert value_representation({"A": {"_leaves": [{"feature": "x", "value": 1.0, "z_score": 0.3}]}})["mode"] == "deviation"
    assert "reference sample" in describe_value_representation("deviation")
    assert render_data_overview({"value_representation": {"note": "Stated."}}) == "Values: Stated."


def _decision_measurements(multimodal: dict) -> str:
    sections, _ = build_sections(
        executor_output={"predictor_input": {"multimodal_unprocessed_raw": multimodal}},
        task_context="",
        count=len,
    )
    return next(s for s in sections if s.key == "measurements").variants[0]


def test_the_decision_state_flags_native_values_and_leaves_deviation_records_unchanged():
    assert _decision_measurements(NATIVE).startswith("Values: No leaf carries a deviation score")
    assert _decision_measurements(MIXED).startswith("Values: 2 of 3 numeric leaves")
    deviation_only = {"A": {"_leaves": [{"feature": "x", "value": 1.0, "z_score": 0.3}]}}
    assert not _decision_measurements(deviation_only).startswith("Values:")
