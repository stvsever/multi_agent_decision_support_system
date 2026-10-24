"""Information loss between a participant record and the models that read it.

A leaf may be summarised by a tool, but it may never silently disappear: every
leaf either reaches a model in full or is passed on verbatim.
"""

from src.full_stack.backend.tools.unimodal_compressor import UnimodalCompressor
from src.full_stack.backend.utils.core.data_loader import _feature_id
from src.full_stack.backend.utils.core.multimodal_coverage import flatten_features
from src.full_stack.backend.utils.toon import json_to_toon


def test_toon_keeps_value_unit_deviation_and_qualifiers_of_every_leaf():
    leaves = {
        "_leaves": [
            {"feature": "Left hippocampal volume", "value": 3.12, "unit": "cm3", "z_score": -1.2, "ref_range": "3.0 to 4.2"},
            {"feature": "MoCA total", "value": 24, "unit": "points"},
            {"feature": "Sex", "value": "Female"},
            {"feature": "Ceramide", "value": -1.5, "z_score": -1.5},
        ]
    }
    text = json_to_toon(leaves)
    assert "Left hippocampal volume = 3.12 cm3 (z -1.20) [ref range 3.0 to 4.2]" in text
    assert "MoCA total = 24 points" in text
    assert "Sex = Female" in text
    # A z that stood in for a missing value is printed once, as the deviation.
    assert "Ceramide (z -1.50)" in text


def test_long_feature_ids_stay_unique():
    prefix = "functional connectivity between the left dorsal attention network and the right "
    a, b = _feature_id(prefix + "frontoparietal control network"), _feature_id(prefix + "default mode network")
    assert a != b
    assert _feature_id("MoCA total") == "moca_total"


class _Settings:
    class models:
        backend = "openrouter"
        tool_model = "deepseek/deepseek-v4-flash-0731"
        tool_max_tokens = 4000

    @staticmethod
    def effective_context_window(model, role=None):
        return 8192  # small enough to force truncation


def test_the_compressor_reports_only_the_leaves_its_prompt_holds():
    leaves = [{"feature": f"Regional measure {i:04d} with a descriptive label", "value": i, "z_score": 0.1} for i in range(4000)]
    domain = {"BRAIN": {"Morphometry": {"_leaves": leaves}}}
    tool = UnimodalCompressor.__new__(UnimodalCompressor)
    tool.settings = _Settings()
    payload = {"input_domains": ["BRAIN", "COGNITION"], "target_condition": "x", "domain_data": domain, "hierarchical_deviation": {}}
    prompt = tool._build_prompt(payload)
    seen = set(payload["_seen_feature_keys"])
    all_keys = {k for k, _ in flatten_features({"BRAIN": domain["BRAIN"]}, domain_hint="BRAIN")}
    assert 0 < len(seen) < len(all_keys)  # truncated, and the report says so
    for key, feat in flatten_features({"BRAIN": domain["BRAIN"]}, domain_hint="BRAIN"):
        assert (key in seen) == (feat["feature"] in prompt)
