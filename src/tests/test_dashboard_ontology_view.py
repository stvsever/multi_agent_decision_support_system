"""The ontology view merges the deviation map with the feature payload."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.api.ontology import band_for, build_ontology, humanise

REPO = Path(__file__).resolve().parent.parent.parent
SUBJ_001 = REPO / "src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO"


def _write(directory: Path, deviation: dict, multimodal: dict | None = None, overview: dict | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hierarchical_deviation_map.json").write_text(json.dumps(deviation))
    (directory / "multimodal_data.json").write_text(json.dumps(multimodal or {}))
    (directory / "data_overview.json").write_text(json.dumps(overview or {"participant_id": directory.name}))


def test_bands_follow_the_writer_thresholds():
    assert band_for(2.4) == "very_high"
    assert band_for(1.0) == "high"
    assert band_for(0.5) == "high_normal"
    assert band_for(0.0) == "normal"
    assert band_for(-0.6) == "low_normal"
    assert band_for(-1.2) == "low"
    assert band_for(-2.5) == "very_low"
    assert band_for(None) == "missing"


def test_generated_ids_keep_their_informative_tail():
    assert humanise("Resting_EEG__Spectral_power__Alpha_8_13_hz") == "Alpha 8 13 hz"
    assert humanise("cortical_thickness") == "Cortical thickness"
    assert humanise("") == ""


def test_real_participant_produces_a_navigable_tree():
    tree = build_ontology(SUBJ_001)
    assert tree["participant_id"] == "SUBJ_001_PSEUDO"
    assert tree["summary"]["domain_count"] == 5
    assert tree["summary"]["leaf_count"] > 0
    assert tree["summary"]["max_depth"] >= 3
    assert tree["has_deviation_map"] and tree["has_multimodal"]
    # Domains are ordered by how much they deviate, which is what the explorer shows first.
    magnitudes = [d["mean_abs_score"] for d in tree["domains"] if d["mean_abs_score"] is not None]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_features_attach_to_the_node_that_owns_them():
    tree = build_ontology(SUBJ_001)
    biological = next(d for d in tree["domains"] if d["id"] == "BIOLOGICAL_ASSAY")

    def find(node, target):
        if node["id"] == target:
            return node
        for child in node["children"]:
            hit = find(child, target)
            if hit:
                return hit
        return None

    ceramides = find(biological, "ceramides")
    assert ceramides is not None
    assert ceramides["features"]
    first = ceramides["features"][0]
    assert first["feature"].startswith("Cer(")
    assert first["band"] == band_for(first["z_score"])
    # Optional keys the hand authored sample carries must survive.
    assert "ref_range" in first


def test_stats_nodes_are_read_as_statistics_not_children(tmp_path):
    """`_stats` is metadata the writer attaches to internal nodes."""
    _write(
        tmp_path / "P",
        {
            "DOMAIN": {
                "group": {
                    "a": {"score": 1.0},
                    "b": {"score": -3.0},
                    "_stats": {"mean_abs_score": 2.0, "n_leaves": 2, "present_leaves": 2},
                },
                "_stats": {"mean_abs_score": 2.0, "n_leaves": 2, "present_leaves": 2},
            }
        },
    )
    tree = build_ontology(tmp_path / "P")
    domain = tree["domains"][0]
    assert [c["id"] for c in domain["children"]] == ["group"]
    group = domain["children"][0]
    assert group["mean_abs_score"] == 2.0
    assert group["leaf_count"] == 2
    assert {c["id"] for c in group["children"]} == {"a", "b"}


def test_null_scores_count_as_missing_not_as_zero(tmp_path):
    _write(tmp_path / "P", {"DOMAIN": {"a": {"score": None}, "b": {"score": 1.5}}})
    tree = build_ontology(tmp_path / "P")
    domain = tree["domains"][0]
    a = next(c for c in domain["children"] if c["id"] == "a")
    assert a["score"] is None
    assert a["band"] == "missing"
    assert domain["present_leaves"] == 1


def test_extremes_are_ranked_by_absolute_deviation(tmp_path):
    _write(tmp_path / "P", {"D": {"small": {"score": 0.2}, "big": {"score": -3.4}, "mid": {"score": 1.1}}})
    tree = build_ontology(tmp_path / "P")
    assert [row["label"] for row in tree["extremes"]] == ["Big", "Mid", "Small"]


def test_deep_generated_trees_are_walked_to_the_leaves(tmp_path):
    node: dict = {"leaf": {"score": 1.0}}
    for depth in range(8):
        node = {f"level_{depth}": node, "_stats": {"mean_abs_score": 1.0, "n_leaves": 1, "present_leaves": 1}}
    _write(tmp_path / "P", {"DEEP": node})
    tree = build_ontology(tmp_path / "P")
    assert tree["summary"]["max_depth"] == 10
    assert tree["extremes"][0]["label"] == "Leaf"


def test_a_folder_without_inputs_returns_an_empty_but_valid_view(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    tree = build_ontology(empty)
    assert tree["domains"] == []
    assert tree["has_deviation_map"] is False
    assert tree["summary"]["domain_count"] == 0


# --- feature folding ---------------------------------------------------------


def test_generated_data_records_each_measurement_once(tmp_path):
    """
    The two engine files describe the same leaf twice. Ids and labels differ
    completely, so the pairing has to come from position plus agreeing z-scores.
    """
    _write(
        tmp_path / "P",
        {
            "TOPO": {
                "network": {
                    "lesion_ratio_p83": {"score": 0.384},
                    "lesion_ratio_p84": {"score": 1.243},
                    "_stats": {"mean_abs_score": 0.814, "n_leaves": 2, "present_leaves": 2},
                }
            }
        },
        {
            "TOPO": {
                "network": {
                    "_leaves": [
                        {"feature": "Lesion overlap: 7Networks_LH_SomMot_1", "z_score": 0.384, "value": "0.12"},
                        {"feature": "Lesion overlap: 7Networks_LH_SomMot_2", "z_score": 1.243, "value": "0.41"},
                    ]
                }
            }
        },
    )
    tree = build_ontology(tmp_path / "P")
    network = tree["domains"][0]["children"][0]
    assert network["features"] == []
    assert [c["id"] for c in network["children"]] == ["lesion_ratio_p83", "lesion_ratio_p84"]
    assert network["children"][0]["value"] == "0.12"
    assert tree["summary"]["feature_count"] == 0
    assert tree["summary"]["leaf_count"] == 2


def test_disagreeing_scores_fall_back_to_name_matching(tmp_path):
    """Positional pairing must not fire when the two files do not line up."""
    _write(
        tmp_path / "P",
        {"D": {"group": {"education_years": {"score": 2.2}, "age": {"score": 1.0}}}},
        {
            "D": {
                "group": {
                    "_leaves": [
                        {"feature": "Age", "z_score": 1.0, "value": "54 years"},
                        {"feature": "Education (years)", "z_score": 2.2, "value": "12 years"},
                    ]
                }
            }
        },
    )
    group = build_ontology(tmp_path / "P")["domains"][0]["children"][0]
    assert group["features"] == []
    by_id = {c["id"]: c for c in group["children"]}
    assert by_id["age"]["value"] == "54 years"
    assert by_id["education_years"]["value"] == "12 years"


def test_different_granularity_keeps_both_levels(tmp_path):
    """Hand authored data puts features below the deviation leaves; keep both."""
    _write(
        tmp_path / "P",
        {"D": {"sphingolipids": {"score": 1.7}}},
        {"D": {"sphingolipids": {"ceramides": {"_leaves": [{"feature": "Cer(d18:1/16:0)", "z_score": 2.4}]}}}},
    )
    node = build_ontology(tmp_path / "P")["domains"][0]["children"][0]
    assert node["score"] == 1.7
    assert [c["id"] for c in node["children"]] == ["ceramides"]
    assert node["children"][0]["features"][0]["feature"] == "Cer(d18:1/16:0)"


def test_direction_rolls_up_through_the_whole_subtree(tmp_path):
    """Averaging only direct leaf children leaves every branch without a sign."""
    _write(
        tmp_path / "P",
        {"D": {"branch": {"sub": {"a": {"score": -2.0}, "b": {"score": -1.0}}}}},
    )
    domain = build_ontology(tmp_path / "P")["domains"][0]
    branch = domain["children"][0]
    assert branch["subtree_signed_mean"] == -1.5
    assert domain["subtree_signed_mean"] == -1.5
    # The absolute magnitude and the signed direction are separate readings.
    assert domain["mean_abs_score"] == 1.5


def test_the_roll_up_accumulator_never_reaches_the_client(tmp_path):
    _write(tmp_path / "P", {"D": {"a": {"score": 1.0}}})
    tree = build_ontology(tmp_path / "P")

    def check(node):
        assert "_signed" not in node
        for child in node["children"]:
            check(child)

    for domain in tree["domains"]:
        check(domain)
